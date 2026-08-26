import json
import logging
import sqlite3
from typing import Optional

from app.agent import stage1
from app.agent.graph import build_graph
from app.config import settings
from app.db import repo_analysis_runs, repo_candidates, repo_events
from app.db.models import EventScope
from app.db.session import session

logger = logging.getLogger(__name__)


def start_run(conn: sqlite3.Connection, scope: Optional[EventScope] = None) -> int:
    """Create the analysis_runs row synchronously so callers get a run_id
    immediately; the actual Stage 1 scan runs separately via execute_run. The
    scope is persisted on the row, which is how it reaches that background
    task — which receives only the run_id."""
    run = repo_analysis_runs.start_run(conn, scope)
    return run.id


def execute_run(run_id: int) -> None:
    """Run Stage 1 (the broad, deduplicated scan producing ranked
    candidates) for `run_id` to completion and record the outcome. Opens
    its own DB connection since this runs as a background task outside the
    request that created the run. Does not investigate anything — that's
    execute_investigation, human-triggered per candidate."""
    with session() as conn:
        try:
            # Rebuilt from the row rather than passed in: this runs on a
            # background task that only gets the run_id. None means a run from
            # before scoping existed, which analyzed everything.
            scope = repo_analysis_runs.get_scope_for_run(conn, run_id)
            result = stage1.run_stage1(conn, run_id, scope=scope)

            # Snapshot the full scope now, at completion — not just a count —
            # so the run history can later show exactly which events/files/
            # nodes this run covered, even after further logs are ingested.
            scope_stats = repo_events.get_scope_summary_stats(conn, scope)
            repo_analysis_runs.complete_run(
                conn,
                run_id,
                events_considered=scope_stats["total"],
                iterations=1,
                candidates_generated=result.candidates_generated,
                candidates_auto_suppressed=result.candidates_auto_suppressed,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cache_creation_input_tokens=result.cache_creation_input_tokens,
                cache_read_input_tokens=result.cache_read_input_tokens,
                scope_json=json.dumps(scope_stats),
            )
        except stage1.Stage1Error as exc:
            # Failed after the model call was billed — record the cost, or
            # the run history shows real spend as $0.0000.
            logger.exception("analysis run %s failed", run_id)
            repo_analysis_runs.fail_run(conn, run_id, str(exc), **(exc.usage or {}))
        except Exception as exc:  # noqa: BLE001 - a bad run must not crash the background task
            logger.exception("analysis run %s failed", run_id)
            repo_analysis_runs.fail_run(conn, run_id, str(exc))


def execute_investigation(candidate_id: int) -> None:
    """Run the Stage 2 investigate/tool-call loop for a single candidate to
    completion (or until it hits the cost cap / iteration cap) and record
    the outcome. Opens its own DB connection, same rationale as execute_run."""
    # Accumulated by the graph's investigate node as it goes. The graph's own
    # state is only readable once invoke() returns, so without this an
    # exception part-way through the loop would throw away every token it had
    # already spent and report the investigation as free.
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "iterations": 0,
    }
    with session() as conn:
        try:
            candidate = repo_candidates.get_candidate(conn, candidate_id)
            graph = build_graph(conn, usage_sink=usage)
            initial_state = {
                "messages": [],
                "analysis_run_id": candidate.analysis_run_id,
                "candidate_id": candidate_id,
                "iteration": 0,
                "max_iterations": settings.max_agent_iterations,
                "hypotheses": [],
                "finalized_findings": [],
                "findings_suppressed": 0,
                "ready_to_conclude": False,
                "conclusion_summary": "",
                "conclusion_outcome": "",
                "stop_reason": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
            final_state = graph.invoke(initial_state, config={"recursion_limit": 50})
            repo_candidates.complete_investigation(
                conn,
                candidate_id,
                input_tokens=final_state.get("input_tokens", 0),
                output_tokens=final_state.get("output_tokens", 0),
                cache_creation_input_tokens=final_state.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=final_state.get("cache_read_input_tokens", 0),
                iterations=final_state.get("iteration", 0),
            )
        except Exception as exc:  # noqa: BLE001 - a bad investigation must not crash the background task
            logger.exception("candidate investigation %s failed", candidate_id)
            repo_candidates.fail_investigation(conn, candidate_id, str(exc), **usage)


def run_analysis_sync(scope: Optional[EventScope] = None) -> int:
    """Convenience entry point for CLI/manual use: creates and runs a Stage 1
    scan in one call, returning the run_id once it's finished."""
    with session() as conn:
        run_id = start_run(conn, scope)
    execute_run(run_id)
    return run_id
