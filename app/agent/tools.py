import functools
import json
import sqlite3
import threading
import time
from typing import Annotated, Any, Callable, Dict, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from app.agent.findings import (
    INVESTIGATING_STATUS,
    REFUTED_OUTCOME,
    VALID_CATEGORIES,
    VALID_HYPOTHESIS_STATUSES,
    VALID_SEVERITIES,
    is_suppressed,
)
from app.db import repo_agent_steps, repo_catalog, repo_events
from app.db.models import EventRecord
from app.ontap import cluster_state


# Hard server-side ceilings on what any single tool call can return.
#
# The Stage 2 cost cap is only evaluated AFTER a model call, so whatever a
# tool returns is guaranteed to reach the model at least once at full price,
# and then rides along in `messages` for every remaining turn. That makes an
# unbounded tool result a hole straight through STAGE2_COST_CAP_USD: before
# these clamps, `limit` came straight from the model with no ceiling, and
# get_event_context passed no limit at all, so a single
# `window_minutes=1440` call on a busy node could return that node's entire
# day — six figures of tokens, past the cap, unstoppable. The model can ask
# for more than this; it does not get more than this.
MAX_TOOL_RESULT_EVENTS = 100
MAX_CONTEXT_WINDOW_MINUTES = 240


def _clamp(value: Optional[int], ceiling: int, default: int) -> int:
    if not isinstance(value, int) or value < 1:
        return default
    return min(value, ceiling)


def _format_events(events: List[EventRecord], truncated_at: Optional[int] = None) -> str:
    """Render events as compact JSON. `truncated_at` tells the model the
    result hit a server-side ceiling, so it narrows its filters instead of
    concluding that's all the events there are."""
    if not events:
        return "No matching events."
    rows = [
        {
            "event_id": e.id,
            "time": e.event_time,
            "node": e.node,
            "event_name": e.event_name,
            "severity": e.severity,
            "message": e.message,
        }
        for e in events
    ]
    # Compact separators, not indent=2: this is the payload that dominates
    # Stage 2's context, and pretty-printing buys the model nothing.
    payload = json.dumps(rows, separators=(",", ":"))
    if truncated_at is not None and len(events) >= truncated_at:
        payload += (
            f"\n(truncated to the first {truncated_at} matching events, in time order — "
            "narrow the node, severity, event name, or time range to see the rest)"
        )
    return payload


class InvestigationContext:
    """Mutable pointer to what the agent is currently investigating, so tools
    don't each have to take state parameters.

    Carries two things: where we are (candidate + turn), for the agent_steps
    audit trail, and which cluster this investigation's evidence came from,
    which is the host the live-state tools query. That host is never
    configuration — a configured default would be used for every investigation
    regardless of which cluster the candidate belongs to.

    Safe as shared mutable state because one investigation is one graph running
    on one thread, and build_tools serializes tool invocation anyway."""

    def __init__(
        self,
        candidate_id: Optional[int] = None,
        cluster: Optional[str] = None,
        scoped: bool = True,
    ) -> None:
        self.candidate_id = candidate_id
        # The cluster this investigation is scoped to, doubling as the REST
        # host: routes_clusters passes the same value straight to
        # fetch_ems_events. None means the events came from an uploaded log
        # file, which has no live cluster behind it.
        self.cluster = cluster
        # Whether `cluster` is a statement about this investigation at all.
        # A run created before scoping existed has scope_cluster NULL, which is
        # indistinguishable from the unspecified (uploaded-logs) pseudo-cluster
        # — so without this flag those legacy investigations would silently
        # narrow their evidence to the uploaded-logs pool. False means "query
        # every cluster", which is exactly what those runs did.
        self.scoped = scoped
        self.iteration = 0
        self.step_index = 0

    def next_index(self) -> int:
        self.step_index += 1
        return self.step_index

    def event_cluster(self):
        """The cluster filter for this investigation's event queries, or
        ANY_CLUSTER when there is nothing to scope to."""
        return self.cluster if self.scoped else repo_events.ANY_CLUSTER


def build_tools(conn: sqlite3.Connection, trace: Optional[InvestigationContext] = None) -> List[Any]:
    """Build the agent's toolset bound to this analysis run's DB connection.

    `trace`, when given, records every tool call to agent_steps so the
    investigation is auditable afterwards — without it the agent's reasoning
    exists only in the in-memory message list and is discarded when the graph
    returns."""

    # Serializes every tool invocation on this toolset's shared connection.
    #
    # LangGraph's ToolNode dispatches a turn's tool calls to a THREADPOOL
    # (prebuilt/tool_node.py:_run_one, via concurrent.futures), so when a model
    # emits two tool calls in one turn they execute on different threads
    # against this one sqlite3 connection. `check_same_thread=False` in
    # app/db/session.py silences Python's same-thread check but does NOT make a
    # connection thread-safe: concurrent use corrupts its internal state and
    # takes the interpreter down with a native SIGSEGV — no Python traceback,
    # nothing catchable in application code.
    #
    # This was reproducible at roughly 1 run in 15 of the test suite once
    # tracing made every tool a WRITER; before that the tools were read-only,
    # which is unsafe in the same way but far less likely to corrupt anything.
    # The lock covers the whole call, not just the trace write, because the
    # tool's own queries share the same connection.
    #
    # Cost: parallel tool calls run one at a time. Most are local SQLite reads
    # taking microseconds, so this is not a meaningful loss — and the cost cap
    # bounds turns, not wall-clock. The exception is get_cluster_state, which
    # holds the lock across a paginated REST walk that can run for seconds on a
    # large cluster; nothing contends for it (one investigation is one graph),
    # so this delays only a sibling tool call in the same turn.
    db_lock = threading.Lock()

    # The cluster every event query below is restricted to. Node names are not
    # unique across clusters (ONTAP names them <cluster>-01/-02, and the
    # simulator defaults to "cluster1"), so an unscoped query can hand the
    # agent another cluster's events as context for this node and let it
    # correlate a pattern that never happened. Stage 1 is scoped for exactly
    # this reason; Stage 2 must be too.
    def event_cluster():
        return trace.event_cluster() if trace is not None else repo_events.ANY_CLUSTER

    def traced(func: Callable) -> Callable:
        """Serialize a tool call on the shared connection, and record its name,
        arguments, timing and a truncated result to agent_steps.

        functools.wraps matters beyond tidiness here: @tool builds its argument
        schema from the wrapped signature via inspect.signature, which follows
        __wrapped__ — without it every tool would present as (*args, **kwargs)
        to the model."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with db_lock:
                if trace is None or trace.candidate_id is None:
                    return func(*args, **kwargs)
                started = time.monotonic()
                index = trace.next_index()
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - re-raised immediately below
                    repo_agent_steps.record_step(
                        conn,
                        candidate_id=trace.candidate_id,
                        iteration=trace.iteration,
                        step_index=index,
                        tool_name=getattr(func, "__name__", "unknown"),
                        tool_args=kwargs or None,
                        error=f"{type(exc).__name__}: {exc}",
                        duration_ms=int((time.monotonic() - started) * 1000),
                    )
                    raise
                # Command results (record_hypothesis, conclude_investigation)
                # carry a state update rather than text; summarize their update
                # keys so the trace still shows what the call did.
                if isinstance(result, Command):
                    summary = f"state update: {sorted(result.update.keys())}"
                else:
                    summary = str(result)
                repo_agent_steps.record_step(
                    conn,
                    candidate_id=trace.candidate_id,
                    iteration=trace.iteration,
                    step_index=index,
                    tool_name=getattr(func, "__name__", "unknown"),
                    tool_args=kwargs or None,
                    result_summary=summary,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
                return result

        return wrapper

    @tool
    @traced
    def query_events(
        node: Optional[str] = None,
        severity: Optional[str] = None,
        event_name_contains: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50,
    ) -> str:
        """Search ingested EMS events. Filter by node name, severity, a
        substring of the dotted event name, and/or an ISO 8601 time range.
        Returns up to `limit` matching events (id, time, node, event_name,
        severity, message) as JSON, capped at 100 regardless of `limit`. Use
        the returned event_id values as evidence when calling
        record_hypothesis."""
        capped = _clamp(limit, MAX_TOOL_RESULT_EVENTS, 50)
        events = repo_events.query_events(
            conn,
            node=node,
            severity=severity,
            event_name_contains=event_name_contains,
            start_time=start_time,
            end_time=end_time,
            limit=capped,
            cluster=event_cluster(),
        )
        return _format_events(events, truncated_at=capped)

    @tool
    @traced
    def get_event_context(event_id: int, window_minutes: int = 30) -> str:
        """Fetch other events on the same node within +/- window_minutes of
        the given event_id, to help correlate a potential incident with what
        else was happening around the same time. window_minutes is capped at
        240 and at most 100 events are returned; narrow the window rather
        than widening it if you get a truncated result."""
        window = _clamp(window_minutes, MAX_CONTEXT_WINDOW_MINUTES, 30)
        events = repo_events.get_events_near(
            conn, event_id, window, limit=MAX_TOOL_RESULT_EVENTS, cluster=event_cluster()
        )
        return _format_events(events, truncated_at=MAX_TOOL_RESULT_EVENTS)

    @tool
    @traced
    def search_events_by_name_pattern(pattern: str, limit: int = 50) -> str:
        """Search events whose dotted event_name contains the given
        substring (e.g. 'disk' to find all disk.* events). Returns at most
        100 events regardless of `limit`."""
        capped = _clamp(limit, MAX_TOOL_RESULT_EVENTS, 50)
        events = repo_events.query_events(
            conn, event_name_contains=pattern, limit=capped, cluster=event_cluster()
        )
        return _format_events(events, truncated_at=capped)

    @tool
    @traced
    def check_suppression(category: str, event_names: List[str], node: Optional[str] = None) -> str:
        """Check whether a candidate finding (category + the set of event
        names it's based on + the node it's on) has previously been
        dismissed by a user as unimportant. ALWAYS call this before calling
        record_hypothesis for a hypothesis you intend to finalize. Returns
        'suppressed' or 'not_suppressed'."""
        suppressed = is_suppressed(conn, category, event_names, node)
        return "suppressed" if suppressed else "not_suppressed"

    @tool
    @traced
    def record_hypothesis(
        category: str,
        severity: str,
        title: str,
        description: str,
        node: Optional[str],
        event_ids: List[int],
        event_names: List[str],
        confidence: float,
        status: str,
        recommendation: Optional[str] = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Record or update a hypothesis about a storage risk or anomaly.
        category must be one of: availability_risk, performance_issue,
        predictive_failure. severity must be one of: critical, high, medium,
        low. status must be 'investigating' (still gathering evidence) or
        'ready' (confident enough to finalize as a finding — call
        check_suppression first). event_ids are the supporting evidence
        event_id values from query_events/get_event_context."""
        # Validated here because this is the only boundary where model-authored
        # values enter the system. An off-enum category is not a cosmetic
        # problem: the signature is computed FROM the category, so such a
        # finding produces a signature Stage 1 can never reproduce — it can
        # never be suppressed, never match an open finding, and dismissing it
        # suppresses nothing.
        #
        # Answered with a corrective ToolMessage rather than an exception: the
        # model can fix it on the next turn, whereas raising would end an
        # investigation the user has already paid for.
        normalized_category = (category or "").strip().lower()
        normalized_severity = (severity or "").strip().lower()
        if normalized_category not in VALID_CATEGORIES or normalized_severity not in VALID_SEVERITIES:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                f"Not recorded: category must be one of {sorted(VALID_CATEGORIES)} "
                                f"and severity one of {sorted(VALID_SEVERITIES)}; got "
                                f"category={category!r}, severity={severity!r}. Call "
                                "record_hypothesis again with valid values."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )
        # An unrecognized status is normalized rather than rejected, and falls
        # back to 'investigating' deliberately: that direction keeps the
        # hypothesis as an explicitly unconfirmed partial instead of promoting
        # a typo like "Ready" into an open finding.
        normalized_status = (status or "").strip().lower()
        if normalized_status not in VALID_HYPOTHESIS_STATUSES:
            normalized_status = INVESTIGATING_STATUS
        hypothesis: Dict[str, Any] = {
            "category": normalized_category,
            "severity": normalized_severity,
            "title": title,
            "description": description,
            "recommendation": recommendation,
            "node": node,
            "evidence_event_ids": event_ids,
            "event_names": event_names,
            "confidence": confidence,
            "status": normalized_status,
        }
        # Contribute only the new hypothesis, not a rebuilt full list — the
        # `hypotheses` state field has an operator.add reducer (see state.py)
        # so LangGraph can safely merge this with any other record_hypothesis
        # call the model made in parallel within the same step.
        return Command(
            update={
                "hypotheses": [hypothesis],
                "messages": [
                    ToolMessage(
                        content=f"Recorded hypothesis: {title} ({normalized_status})",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    @tool
    @traced
    def conclude_investigation(
        summary: str,
        outcome: str = REFUTED_OUTCOME,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Call this once you have no more open hypotheses worth
        investigating further, ending the investigation loop early instead of
        running to the iteration cap.

        `outcome` is 'confirmed' if you finalized one or more findings, or
        'refuted' if you determined this candidate is not a real issue. Only say
        'confirmed' if you actually called record_hypothesis with status='ready'
        — concluding 'confirmed' without one leaves your theory recorded as an
        unconfirmed lead rather than a finding.
        `summary` is one or two sentences stating WHAT YOU CONCLUDED AND WHY,
        citing the specific evidence that decided it (e.g. "Both
        takeover-disabled events were part of a planned halt on cluster1-02;
        HA takeover is currently possible on both nodes, so this is not an
        availability risk."). A refuted candidate is shown to the user with
        this summary as the explanation, so "no issue found" on its own is not
        good enough — say what ruled it out."""
        return Command(
            update={
                "ready_to_conclude": True,
                "conclusion_summary": summary,
                "conclusion_outcome": (outcome or REFUTED_OUTCOME).strip().lower(),
                "messages": [
                    ToolMessage(content=f"Investigation concluded ({outcome}).", tool_call_id=tool_call_id)
                ],
            }
        )

    @tool
    @traced
    def lookup_event_definition(
        event_names: Optional[List[str]] = None, name_pattern: Optional[str] = None
    ) -> str:
        """Look up what an ONTAP EMS event actually MEANS, from the official
        event catalog: its documented description, the reason it fires, its
        catalog severity, and NetApp's own corrective action for it. Pass
        `event_names` for exact names (e.g. ['cf.fsm.takeoverOfPartnerDisabled'],
        up to 15 at a time), or `name_pattern` to find events whose name
        contains a substring (e.g. 'takeover'). Prefer this over inferring an
        event's meaning from its name or message text, and use the returned
        corrective_action to ground a finding's recommendation."""
        if name_pattern:
            records = repo_catalog.search(conn, name_pattern)
        else:
            records = repo_catalog.lookup(conn, event_names or [])
        if not records:
            asked = name_pattern or ", ".join(event_names or []) or "(nothing)"
            return (
                f"No catalog entry for {asked}. The catalog covers standard ONTAP EMS messages; "
                "an unmatched name may be from a different ONTAP version. Rely on the event's "
                "message text instead."
            )
        return json.dumps(records, separators=(",", ":"))

    @tool
    @traced
    def get_event_rate_baseline(event_name: str, node: Optional[str] = None) -> str:
        """Check how often an event normally fires in the ingested history, to
        judge whether the occurrences you're looking at are actually unusual.
        Returns total occurrences, the time span they cover, a mean per-day
        rate, the busiest single day, and — when `node` is given — how often
        the same event fired elsewhere in the cluster, which is how you tell an
        isolated node problem from normal cluster-wide chatter. Note the
        baseline covers only ingested events, so a short log makes everything
        look frequent."""
        return json.dumps(
            repo_events.get_event_rate_baseline(conn, event_name, node, cluster=event_cluster()),
            separators=(",", ":"),
        )

    @tool
    @traced
    def get_cluster_state(area: str, node: Optional[str] = None, name: Optional[str] = None) -> str:
        """Read the CURRENT state of the live ONTAP cluster, to corroborate or
        refute what the logs suggest — the logs say what happened, this says
        what is true now. `area` must be one of:
          - 'aggregates': per-aggregate capacity, fullest first (confirms space
            pressure behind a volume/space event). This is space consumed, NOT
            how busy the aggregate is.
          - 'disks': disk state histogram plus every disk in a bad state
            (confirms a predictive-failure or RAID event)
          - 'ha': per-node HA state, whether takeover is currently possible
            (confirms a failover-risk event)
          - 'volumes': volumes that are offline/restricted or >=90% full
        `node` narrows aggregates/disks to one node; `name` filters volumes by
        substring. If no cluster is configured or it can't be reached this
        returns available:false with a reason — treat that as "no live evidence
        either way", not as evidence of a problem, and continue from the logs.

        Counts are complete (every page is fetched), but the listed items are
        trimmed to the most relevant few: compare 'unhealthy_count' /
        'matching_count' against 'listed' before concluding you have seen
        everything. An 'incomplete_reason' key means the read stopped early and
        the counts are a lower bound.

        There is deliberately NO performance area — no latency, IOPS or
        throughput is available from this tool. Do not infer performance from
        capacity."""
        area_normalized = (area or "").strip().lower()
        # The host is this investigation's own cluster, never a configured
        # default: a default would point every investigation at the same
        # cluster regardless of where its evidence came from.
        host = trace.cluster if trace else None
        if area_normalized in {"aggregates", "disks", "ha", "volumes"} and not cluster_state.credentials_configured():
            return json.dumps(
                {
                    "available": False,
                    "reason": "No ONTAP credentials are configured (set ONTAP_USER and "
                    "ONTAP_PASSWORD). Continue the investigation from the ingested logs alone.",
                },
                separators=(",", ":"),
            )
        if area_normalized == "aggregates":
            result = cluster_state.get_aggregate_capacity(host, node=node)
        elif area_normalized == "disks":
            result = cluster_state.get_disk_health(host, node=node)
        elif area_normalized == "ha":
            result = cluster_state.get_node_ha_status(host)
        elif area_normalized == "volumes":
            result = cluster_state.get_volume_state(host, name=name)
        else:
            return json.dumps(
                {
                    "available": False,
                    "reason": f"unknown area '{area}'; valid areas are aggregates, disks, ha, volumes",
                },
                separators=(",", ":"),
            )
        return json.dumps(result, separators=(",", ":"), default=str)

    return [
        query_events,
        get_event_context,
        search_events_by_name_pattern,
        lookup_event_definition,
        get_event_rate_baseline,
        get_cluster_state,
        check_suppression,
        record_hypothesis,
        conclude_investigation,
    ]
