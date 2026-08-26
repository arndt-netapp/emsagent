import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.db.models import Finding

logger = logging.getLogger(__name__)

# Ceiling on how many event ids one finding can cite. Deliberately equal to
# tools.MAX_TOOL_RESULT_EVENTS — the most events a single tool call can return —
# rather than imported from it: the repo layer must not depend on the agent
# layer, which already depends on this module. See insert_evidence for what an
# unbounded list actually breaks.
MAX_EVIDENCE_IDS = 100


def _row_to_finding(row: sqlite3.Row) -> Finding:
    return Finding(
        id=row["id"],
        analysis_run_id=row["analysis_run_id"],
        candidate_id=row["candidate_id"],
        category=row["category"],
        severity=row["severity"],
        title=row["title"],
        description=row["description"],
        recommendation=row["recommendation"],
        node=row["node"],
        signature=row["signature"],
        pattern_signature=row["pattern_signature"],
        status=row["status"],
        confidence=row["confidence"],
        created_at=row["created_at"],
        dismissed_at=row["dismissed_at"],
    )


def insert_finding(
    conn: sqlite3.Connection,
    analysis_run_id: Optional[int],
    category: str,
    severity: str,
    title: str,
    description: str,
    recommendation: Optional[str],
    node: Optional[str],
    signature: str,
    pattern_signature: str,
    confidence: Optional[float],
    evidence_event_ids: List[int],
    candidate_id: Optional[int] = None,
    status: str = "open",
) -> Finding:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO findings
            (analysis_run_id, candidate_id, category, severity, title, description, recommendation,
             node, signature, pattern_signature, status, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis_run_id,
            candidate_id,
            category,
            severity,
            title,
            description,
            recommendation,
            node,
            signature,
            pattern_signature,
            status,
            confidence,
            now,
        ),
    )
    finding_id = cur.lastrowid
    insert_evidence(conn, finding_id, evidence_event_ids)
    return get_finding(conn, finding_id)


def insert_evidence(conn: sqlite3.Connection, finding_id: int, event_ids: List[int], note: Optional[str] = None) -> None:
    """Cite events as evidence, silently dropping ids that don't exist.

    These ids come from the MODEL, and `finding_evidence.event_id` is a foreign
    key with `PRAGMA foreign_keys = ON` — so one hallucinated id used to raise
    IntegrityError from inside `insert_finding`, *after* the findings row was
    written. That was expensive in three compounding ways: the investigation
    died and its whole Stage 2 spend bought nothing; the half-written findings
    row was committed anyway (runner.execute_investigation catches inside its
    `with session()`, so nothing rolls back); and because that orphan had
    status='open', `find_open_by_signature` then blocked the real version of
    that finding from ever being created.

    Dropping a bad citation instead mirrors how stage1.py already handles
    hallucinated `refs`: keep the result, lose the bogus reference, log it.
    Non-integer ids simply fail to match and are dropped the same way.

    The list is also CAPPED, for the same reason and against the same failure.
    One `?` per id with no ceiling means a model citing more ids than SQLite's
    variable limit (999 on older builds, 32766 since 3.32) raises from this
    query — again after the findings row exists, again committed, again leaving
    an orphan `status='open'` row that blocks the real finding. MAX_EVIDENCE_IDS
    is the number of events a single tool call can even return
    (tools.MAX_TOOL_RESULT_EVENTS), so no honestly-sourced citation reaches it."""
    if not event_ids:
        return
    deduped = list(dict.fromkeys(event_ids))
    if len(deduped) > MAX_EVIDENCE_IDS:
        logger.warning(
            "finding %s cited %s event ids; keeping the first %s",
            finding_id,
            len(deduped),
            MAX_EVIDENCE_IDS,
        )
        deduped = deduped[:MAX_EVIDENCE_IDS]
    placeholders = ",".join("?" for _ in deduped)
    known = {
        row["id"]
        for row in conn.execute(f"SELECT id FROM events WHERE id IN ({placeholders})", deduped)
    }
    kept = [event_id for event_id in deduped if event_id in known]
    dropped = [event_id for event_id in deduped if event_id not in known]
    if dropped:
        logger.warning(
            "finding %s cited %s event id(s) that do not exist; dropping them: %s",
            finding_id,
            len(dropped),
            dropped,
        )
    if not kept:
        return
    conn.executemany(
        "INSERT INTO finding_evidence (finding_id, event_id, note) VALUES (?, ?, ?)",
        [(finding_id, event_id, note) for event_id in kept],
    )


def get_finding(conn: sqlite3.Connection, finding_id: int) -> Optional[Finding]:
    row = conn.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone()
    return _row_to_finding(row) if row else None


def find_open_by_signature(conn: sqlite3.Connection, signature: str) -> Optional[Finding]:
    row = conn.execute(
        "SELECT * FROM findings WHERE signature = ? AND status = 'open' LIMIT 1", (signature,)
    ).fetchone()
    return _row_to_finding(row) if row else None


def get_finding_evidence(conn: sqlite3.Connection, finding_id: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT e.*, fe.note AS evidence_note
        FROM finding_evidence fe
        JOIN events e ON e.id = fe.event_id
        WHERE fe.finding_id = ?
        ORDER BY e.event_time, e.sequence_num
        """,
        (finding_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_findings(
    conn: sqlite3.Connection,
    status: Optional[str] = None,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Finding]:
    clauses = []
    params: List[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if category:
        clauses.append("category = ?")
        params.append(category)
    if severity:
        clauses.append("severity = ?")
        params.append(severity)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.extend([limit, offset])
    rows = conn.execute(
        f"SELECT * FROM findings {where} ORDER BY created_at DESC LIMIT ? OFFSET ?", params
    ).fetchall()
    return [_row_to_finding(r) for r in rows]


def list_findings_for_candidate(conn: sqlite3.Connection, candidate_id: int) -> List[Finding]:
    rows = conn.execute(
        "SELECT * FROM findings WHERE candidate_id = ? ORDER BY created_at", (candidate_id,)
    ).fetchall()
    return [_row_to_finding(r) for r in rows]


def dismiss(conn: sqlite3.Connection, finding_id: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE findings SET status = 'dismissed', dismissed_at = ? WHERE id = ?", (now, finding_id))
