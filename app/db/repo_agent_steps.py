import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Tool arguments and result summaries are stored truncated. These rows exist to
# make an investigation auditable, not to be a second copy of the evidence —
# and an un-truncated result_summary would put a 100-event JSON blob in the
# database for every single query the agent makes.
MAX_STORED_ARGS_CHARS = 1000
MAX_STORED_RESULT_CHARS = 2000


def _truncate(text: Optional[str], limit: int) -> Optional[str]:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def record_step(
    conn: sqlite3.Connection,
    candidate_id: int,
    iteration: int,
    step_index: int,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]] = None,
    result_summary: Optional[str] = None,
    error: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> None:
    """Append one tool call to the investigation's audit trail.

    Never raises: a failure to record a trace row must not take down the
    investigation it is only observing. A lost trace row is a lost trace row;
    a lost investigation is money."""
    try:
        conn.execute(
            """
            INSERT INTO agent_steps
                (candidate_id, iteration, step_index, tool_name, tool_args,
                 result_summary, error, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                iteration,
                step_index,
                tool_name,
                _truncate(json.dumps(tool_args, default=str), MAX_STORED_ARGS_CHARS) if tool_args else None,
                _truncate(result_summary, MAX_STORED_RESULT_CHARS),
                _truncate(error, MAX_STORED_RESULT_CHARS),
                duration_ms,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    except sqlite3.Error:  # noqa: BLE001 - tracing must never break the run
        pass


def list_steps_for_candidate(conn: sqlite3.Connection, candidate_id: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM agent_steps WHERE candidate_id = ? ORDER BY id",
        (candidate_id,),
    ).fetchall()
    steps = []
    for row in rows:
        args = None
        if row["tool_args"]:
            try:
                args = json.loads(row["tool_args"])
            except ValueError:
                # Truncated mid-JSON by MAX_STORED_ARGS_CHARS — show the raw
                # text rather than dropping the row.
                args = {"_raw": row["tool_args"]}
        steps.append(
            {
                "id": row["id"],
                "iteration": row["iteration"],
                "step_index": row["step_index"],
                "tool_name": row["tool_name"],
                "tool_args": args,
                "result_summary": row["result_summary"],
                "error": row["error"],
                "duration_ms": row["duration_ms"],
                "created_at": row["created_at"],
            }
        )
    return steps
