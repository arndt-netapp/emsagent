import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.db.models import Candidate


@dataclass
class CandidateInput:
    rank: int
    category: str
    node: Optional[str]
    rationale: str
    confidence: Optional[float]
    refs: List[int]
    status: str = "pending"
    # Refs resolved to durable event ids at generation time — see
    # compaction.resolve_leads for why this can't be re-derived later.
    leads: List[Dict[str, Any]] = field(default_factory=list)


def _row_to_candidate(row: sqlite3.Row) -> Candidate:
    return Candidate(
        id=row["id"],
        analysis_run_id=row["analysis_run_id"],
        rank=row["rank"],
        category=row["category"],
        node=row["node"],
        rationale=row["rationale"],
        confidence=row["confidence"],
        refs=json.loads(row["refs"]),
        leads=json.loads(row["leads"]) if row["leads"] else [],
        status=row["status"],
        discard_reason=row["discard_reason"],
        investigation_input_tokens=row["investigation_input_tokens"],
        investigation_output_tokens=row["investigation_output_tokens"],
        investigation_cache_creation_input_tokens=row["investigation_cache_creation_input_tokens"],
        investigation_cache_read_input_tokens=row["investigation_cache_read_input_tokens"],
        investigation_iterations=row["investigation_iterations"],
        investigation_started_at=row["investigation_started_at"],
        investigation_completed_at=row["investigation_completed_at"],
        investigation_error=row["investigation_error"],
        created_at=row["created_at"],
    )


def bulk_insert_candidates(
    conn: sqlite3.Connection, analysis_run_id: int, candidates: List[CandidateInput]
) -> List[Candidate]:
    now = datetime.now(timezone.utc).isoformat()
    ids = []
    for c in candidates:
        cur = conn.execute(
            """
            INSERT INTO candidates
                (analysis_run_id, rank, category, node, rationale, confidence, refs, leads, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_run_id,
                c.rank,
                c.category,
                c.node,
                c.rationale,
                c.confidence,
                json.dumps(c.refs),
                json.dumps(c.leads),
                c.status,
                now,
            ),
        )
        ids.append(cur.lastrowid)
    return [get_candidate(conn, i) for i in ids]


def get_candidate(conn: sqlite3.Connection, candidate_id: int) -> Optional[Candidate]:
    row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    return _row_to_candidate(row) if row else None


def list_candidates_for_run(conn: sqlite3.Connection, analysis_run_id: int) -> List[Candidate]:
    rows = conn.execute(
        "SELECT * FROM candidates WHERE analysis_run_id = ? ORDER BY rank", (analysis_run_id,)
    ).fetchall()
    return [_row_to_candidate(r) for r in rows]


def start_investigation(conn: sqlite3.Connection, candidate_id: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "UPDATE candidates SET status = 'investigating', investigation_started_at = ? WHERE id = ? AND status = 'pending'",
        (now, candidate_id),
    )
    if cur.rowcount == 0:
        raise ValueError(f"candidate {candidate_id} is not in a 'pending' state")


def complete_investigation(
    conn: sqlite3.Connection,
    candidate_id: int,
    input_tokens: int,
    output_tokens: int,
    iterations: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        UPDATE candidates
        SET status = 'investigated', investigation_completed_at = ?,
            investigation_input_tokens = ?, investigation_output_tokens = ?, investigation_iterations = ?,
            investigation_cache_creation_input_tokens = ?, investigation_cache_read_input_tokens = ?
        WHERE id = ?
        """,
        (now, input_tokens, output_tokens, iterations, cache_creation_input_tokens, cache_read_input_tokens, candidate_id),
    )


def fail_investigation(
    conn: sqlite3.Connection,
    candidate_id: int,
    error_message: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    iterations: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> None:
    """Record a failed investigation, including whatever tokens it burned
    before failing. Those turns were billed by Anthropic whether or not the
    loop reached a conclusion, so leaving the columns NULL (as this used to)
    reports real spend as no spend — the opposite of what a cost-capped
    feature should do. Callers get the counts from build_graph's usage_sink,
    which survives an exception out of the graph."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        UPDATE candidates
        SET status = 'investigated', investigation_completed_at = ?, investigation_error = ?,
            investigation_input_tokens = ?, investigation_output_tokens = ?, investigation_iterations = ?,
            investigation_cache_creation_input_tokens = ?, investigation_cache_read_input_tokens = ?
        WHERE id = ?
        """,
        (
            now,
            error_message,
            input_tokens,
            output_tokens,
            iterations,
            cache_creation_input_tokens,
            cache_read_input_tokens,
            candidate_id,
        ),
    )


def discard(conn: sqlite3.Connection, candidate_id: int, reason: Optional[str] = None) -> None:
    cur = conn.execute(
        "UPDATE candidates SET status = 'discarded', discard_reason = ? WHERE id = ? AND status = 'pending'",
        (reason, candidate_id),
    )
    if cur.rowcount == 0:
        raise ValueError(f"candidate {candidate_id} is not in a 'pending' state")
