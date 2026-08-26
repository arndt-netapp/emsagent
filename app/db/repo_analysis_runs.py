import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from app.db.models import EventScope


@dataclass
class AnalysisRun:
    id: int
    status: str
    started_at: str
    completed_at: Optional[str]
    events_considered: int
    iterations: int
    candidates_generated: int
    candidates_auto_suppressed: int
    error_message: Optional[str]
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    scope_json: Optional[str]
    # The scope this run was requested for, recorded at start. Defaulted so
    # rows created before scoping existed still construct; None reads as
    # "unscoped", which is exactly what those runs were.
    scope_mode: Optional[str] = None
    scope_cluster: Optional[str] = None
    scope_file_id: Optional[int] = None
    scope_since: Optional[str] = None
    scope_label: Optional[str] = None


def _row_to_run(row: sqlite3.Row) -> AnalysisRun:
    return AnalysisRun(
        id=row["id"],
        status=row["status"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        events_considered=row["events_considered"],
        iterations=row["iterations"],
        candidates_generated=row["candidates_generated"],
        candidates_auto_suppressed=row["candidates_auto_suppressed"],
        error_message=row["error_message"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cache_creation_input_tokens=row["cache_creation_input_tokens"],
        cache_read_input_tokens=row["cache_read_input_tokens"],
        scope_json=row["scope_json"],
        scope_mode=row["scope_mode"],
        scope_cluster=row["scope_cluster"],
        scope_file_id=row["scope_file_id"],
        scope_since=row["scope_since"],
        scope_label=row["scope_label"],
    )


def start_run(conn: sqlite3.Connection, scope: Optional[EventScope] = None) -> AnalysisRun:
    """Record the run and the scope it was requested for. The scope is stored
    now, not at completion, because execute_run happens on a background task
    that gets only the run_id — the row is how the scope reaches it."""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO analysis_runs
            (status, started_at, scope_mode, scope_cluster, scope_file_id, scope_since, scope_label)
        VALUES ('running', ?, ?, ?, ?, ?, ?)
        """,
        (
            now,
            scope.mode if scope else None,
            scope.cluster if scope else None,
            scope.file_id if scope else None,
            scope.since if scope else None,
            scope.label if scope else None,
        ),
    )
    return get_run(conn, cur.lastrowid)


def get_scope_for_run(conn: sqlite3.Connection, run_id: int) -> Optional[EventScope]:
    """Rebuild the EventScope a run was started with. Returns None for runs
    predating scoping, which analyzed every event in the database — passing
    None downstream reproduces exactly that behavior."""
    run = get_run(conn, run_id)
    if run is None or not run.scope_mode:
        return None
    return EventScope(
        mode=run.scope_mode,
        cluster=run.scope_cluster,
        file_id=run.scope_file_id,
        since=run.scope_since,
        label=run.scope_label or run.scope_mode,
    )


def get_last_completed_run_for_scope(
    conn: sqlite3.Connection, mode: str, cluster: Optional[str], file_id: Optional[int] = None
) -> Optional[AnalysisRun]:
    """Most recent completed run over this exact scope.

    `IS ?` throughout rather than `= ?`: the unspecified pseudo-cluster is NULL
    and must match itself, and the same holds for scope_file_id on the modes
    that don't use one. Including file_id matters for mode='file', where two
    runs over two different files would otherwise look like the same scope and
    the second would be refused as a no-op."""
    row = conn.execute(
        """
        SELECT * FROM analysis_runs
        WHERE status = 'completed' AND scope_mode = ?
          AND scope_cluster IS ? AND scope_file_id IS ?
        ORDER BY id DESC LIMIT 1
        """,
        (mode, cluster, file_id),
    ).fetchone()
    return _row_to_run(row) if row else None


def get_run(conn: sqlite3.Connection, run_id: int) -> Optional[AnalysisRun]:
    row = conn.execute("SELECT * FROM analysis_runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_run(row) if row else None


def list_runs(conn: sqlite3.Connection, limit: int = 50, offset: int = 0) -> List[AnalysisRun]:
    rows = conn.execute(
        "SELECT * FROM analysis_runs ORDER BY started_at DESC LIMIT ? OFFSET ?", (limit, offset)
    ).fetchall()
    return [_row_to_run(r) for r in rows]


def get_last_completed_run(conn: sqlite3.Connection) -> Optional[AnalysisRun]:
    row = conn.execute(
        "SELECT * FROM analysis_runs WHERE status = 'completed' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return _row_to_run(row) if row else None


def complete_run(
    conn: sqlite3.Connection,
    run_id: int,
    events_considered: int,
    iterations: int,
    candidates_generated: int,
    candidates_auto_suppressed: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    scope_json: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        UPDATE analysis_runs
        SET status = 'completed', completed_at = ?, events_considered = ?, iterations = ?,
            candidates_generated = ?, candidates_auto_suppressed = ?, input_tokens = ?, output_tokens = ?,
            cache_creation_input_tokens = ?, cache_read_input_tokens = ?, scope_json = ?
        WHERE id = ?
        """,
        (
            now,
            events_considered,
            iterations,
            candidates_generated,
            candidates_auto_suppressed,
            input_tokens,
            output_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
            scope_json,
            run_id,
        ),
    )


def fail_run(
    conn: sqlite3.Connection,
    run_id: int,
    error_message: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> None:
    """Record a failed run, including any tokens it burned before failing.
    A Stage 1 call that reaches the API and then fails afterwards — a
    truncated response, a schema mismatch — has already been billed for the
    whole corpus; leaving the counters at their DEFAULT 0 (as this used to)
    displays that as $0.0000. stage1.Stage1Error carries the usage across."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        UPDATE analysis_runs
        SET status = 'failed', completed_at = ?, error_message = ?, input_tokens = ?, output_tokens = ?,
            cache_creation_input_tokens = ?, cache_read_input_tokens = ?
        WHERE id = ?
        """,
        (
            now,
            error_message,
            input_tokens,
            output_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
            run_id,
        ),
    )
