import json
import sqlite3
from datetime import timedelta
from typing import Any, Dict, List, Optional

from dateutil import parser as dateutil_parser

from app.agent.pricing import estimate_stage1_cost_usd
from app.agent.runner import execute_run, start_run
from app.db import repo_analysis_runs, repo_events, repo_feedback, repo_files
from app.db.models import EventScope

__all__ = [
    "start_run",
    "execute_run",
    "trigger",
    "available_scopes",
    "scope_option",
    "resolve_scope",
    "NoNewDataError",
    "InvalidScopeError",
    "WINDOW_HOURS",
]

# The "last 24 hours" window. A single fixed value rather than a free-form
# duration: this is a PoC and every extra knob is another thing to explain.
WINDOW_HOURS = 24

UNSPECIFIED_CLUSTER_LABEL = "unspecified (uploaded logs)"


class NoNewDataError(Exception):
    """Raised by trigger() when nothing has changed within the requested scope
    since the last completed run over that same scope — so a new run would just
    re-derive the same result at the cost of another round of model calls."""


class InvalidScopeError(Exception):
    """Raised when the requested scope can't be resolved to any events."""


def _cluster_label(cluster: Optional[str]) -> str:
    return cluster if cluster else UNSPECIFIED_CLUSTER_LABEL


def _scope_size(conn: sqlite3.Connection, scope: EventScope) -> Dict[str, Any]:
    """How big a run over this scope is, in events and in money.

    The estimate is shown before the run rather than only recorded after it: an
    autosupport EMS log can hold an order of magnitude more events than a
    cluster fetch, and Stage 1 is a single call whose cost is linear in that
    count. See pricing.estimate_stage1_cost_usd for how rough it is."""
    event_count = repo_events.count_events(conn, scope)
    return {"event_count": event_count, "estimated_cost_usd": estimate_stage1_cost_usd(event_count)}


def resolve_scope(
    conn: sqlite3.Connection, mode: str, cluster: Optional[str], file_id: Optional[int] = None
) -> EventScope:
    """Turn a scope request into concrete event filters.

    Every mode is single-cluster by construction — see EventScope for why there
    is no all-clusters option."""
    if mode == "file":
        # One specific ingested file, for the Analyze button on the Files page.
        # `recent_pull` only ever resolves to the newest file for a cluster, so
        # without this an older bundle could not be analyzed at all.
        #
        # The cluster comes from the FILE, not from the caller: the file is the
        # authority on where its events came from, and taking a caller-supplied
        # cluster here would let a scope be built whose filter matches nothing.
        if file_id is None:
            raise InvalidScopeError("mode 'file' requires a file_id.")
        file_record = repo_files.get_file(conn, file_id)
        if file_record is None:
            raise InvalidScopeError(f"No such file: {file_id}.")
        return EventScope(
            mode=mode,
            cluster=file_record.cluster,
            file_id=file_record.id,
            label=f"file {file_record.filename} on {_cluster_label(file_record.cluster)}",
        )

    if mode == "recent_pull":
        # "One pull" is expressible because a cluster fetch writes exactly one
        # file (routes_clusters), so the most recently ingested file for this
        # cluster IS its most recent pull.
        file_record = repo_files.get_most_recent_ingested(conn, cluster)
        if file_record is None:
            raise InvalidScopeError(f"No ingested files for cluster {_cluster_label(cluster)}.")
        return EventScope(
            mode=mode,
            cluster=cluster,
            file_id=file_record.id,
            label=f"most recent pull ({file_record.filename}) on {_cluster_label(cluster)}",
        )

    if mode == "last_24h":
        # Anchored to the newest event in the data, NOT wall-clock now: this
        # tool is normally pointed at logs collected earlier, and a wall-clock
        # window would return nothing for any historical file. See
        # repo_events.get_latest_event_time.
        latest = repo_events.get_latest_event_time(conn, cluster)
        if latest is None:
            raise InvalidScopeError(f"No timestamped events for cluster {_cluster_label(cluster)}.")
        try:
            since = (dateutil_parser.isoparse(latest) - timedelta(hours=WINDOW_HOURS)).isoformat()
        except (ValueError, TypeError) as exc:
            raise InvalidScopeError(f"Could not parse latest event time {latest!r}: {exc}")
        return EventScope(
            mode=mode,
            cluster=cluster,
            since=since,
            label=f"last {WINDOW_HOURS}h of activity on {_cluster_label(cluster)} (through {latest})",
        )

    raise InvalidScopeError(f"Unknown scope mode {mode!r}; expected 'recent_pull', 'last_24h' or 'file'.")


def available_scopes(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """The concrete run options to show in the UI: two per cluster present in
    the data. Enumerated server-side rather than offering a cluster picker plus
    a mode picker, so the user makes one choice instead of two."""
    options: List[Dict[str, Any]] = []
    for entry in repo_events.list_clusters(conn):
        cluster = entry["cluster"]
        for mode in ("recent_pull", "last_24h"):
            try:
                scope = resolve_scope(conn, mode, cluster)
            except InvalidScopeError:
                # e.g. a cluster whose events all lack timestamps has no
                # meaningful 24h window; just don't offer that option.
                continue
            options.append(
                {
                    "mode": mode,
                    "cluster": cluster,
                    "cluster_label": _cluster_label(cluster),
                    "label": scope.label,
                    **_scope_size(conn, scope),
                }
            )
    return options


def scope_option(conn: sqlite3.Connection, mode: str, cluster: Optional[str], file_id: Optional[int] = None):
    """One scope option in the same shape available_scopes returns.

    Exists so the Files page's "Analyze this file" link resolves its option
    server-side too, instead of the frontend hand-building a scope (and its
    cluster label, and its cost estimate) that only this module should own."""
    scope = resolve_scope(conn, mode, cluster, file_id)
    return {
        "mode": scope.mode,
        "cluster": scope.cluster,
        "cluster_label": _cluster_label(scope.cluster),
        "file_id": scope.file_id,
        "label": scope.label,
        **_scope_size(conn, scope),
    }


def trigger(
    conn: sqlite3.Connection,
    mode: str = "last_24h",
    cluster: Optional[str] = None,
    file_id: Optional[int] = None,
) -> int:
    """Synchronously record a new analysis run over the requested scope; the
    caller schedules execute_run(run_id) (e.g. via FastAPI BackgroundTasks).

    Raises NoNewDataError when there is nothing new to analyze, and
    InvalidScopeError when the scope resolves to no events."""
    if repo_events.count_events(conn) == 0:
        raise NoNewDataError("No EMS events have been ingested yet.")

    scope = resolve_scope(conn, mode, cluster, file_id)
    scoped_total = repo_events.count_events(conn, scope)
    if scoped_total == 0:
        raise InvalidScopeError(f"No events in scope: {scope.label}.")

    # Compared against the last run OVER THE SAME SCOPE, not the last run
    # overall. Keyed on scope because runs are no longer totally ordered by
    # event count: analyzing cluster B never changes cluster A's corpus, so a
    # global "have the numbers changed" check would either refuse a legitimate
    # first run on B or wave through a pointless repeat on A.
    last_run = repo_analysis_runs.get_last_completed_run_for_scope(
        conn, scope.mode, scope.cluster, scope.file_id
    )
    if (
        last_run is not None
        and scoped_total == last_run.events_considered
        and not repo_feedback.has_feedback_since(conn, last_run.completed_at)
    ):
        raise NoNewDataError(
            f"No new events in this scope since run #{last_run.id} "
            f"({scoped_total} events already analyzed: {scope.label}) and no new feedback to reconsider."
        )

    return start_run(conn, scope)
