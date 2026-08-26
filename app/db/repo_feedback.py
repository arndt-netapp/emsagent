import sqlite3
from datetime import datetime, timezone
from typing import Optional

from app.db.models import FeedbackRecord

_VALID_SCOPES = {"node", "global"}


def _row_to_feedback(row: sqlite3.Row) -> FeedbackRecord:
    return FeedbackRecord(
        id=row["id"],
        finding_id=row["finding_id"],
        signature=row["signature"],
        pattern_signature=row["pattern_signature"],
        scope=row["scope"],
        reason=row["reason"],
        created_at=row["created_at"],
    )


def insert_feedback(
    conn: sqlite3.Connection,
    finding_id: int,
    signature: str,
    pattern_signature: str,
    scope: str = "node",
    reason: Optional[str] = None,
) -> FeedbackRecord:
    if scope not in _VALID_SCOPES:
        raise ValueError(f"invalid feedback scope: {scope}")
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO feedback (finding_id, signature, pattern_signature, scope, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (finding_id, signature, pattern_signature, scope, reason, now),
    )
    row = conn.execute("SELECT * FROM feedback WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _row_to_feedback(row)


def has_feedback_since(conn: sqlite3.Connection, since_iso: str) -> bool:
    """Whether any feedback (dismissal) was recorded after the given
    timestamp — used to decide whether a re-run over an unchanged event set
    could still produce a different (suppressed) result."""
    row = conn.execute("SELECT 1 FROM feedback WHERE created_at > ? LIMIT 1", (since_iso,)).fetchone()
    return row is not None


def recent_dismissals(conn: sqlite3.Connection, limit: int = 25) -> list:
    """The most recent dismissals, joined to the finding they rejected, for
    Stage 1's "previously dismissed, and why" prompt block.

    The signature tables already stop a dismissed pattern from being re-raised,
    but that is a post-hoc filter: the model still spends reasoning on the
    pattern and still ranks it. Showing it what a human rejected — and the
    stated reason, which until now was written to `feedback.reason` and never
    read by anything — lets it generalize ("this cluster's owner doesn't care
    about backup warnings") instead of re-proposing near-misses that dodge an
    exact signature match.

    Bounded by `limit` because this rides in the system block of the most
    expensive call in the system."""
    rows = conn.execute(
        """
        SELECT f.reason, f.scope, f.created_at,
               fi.title, fi.category, fi.node
        FROM feedback f
        JOIN findings fi ON fi.id = f.finding_id
        ORDER BY f.created_at DESC, f.id DESC
        LIMIT ?
        """,
        (max(1, limit),),
    ).fetchall()
    return [
        {
            "title": r["title"],
            "category": r["category"],
            "node": r["node"],
            "scope": r["scope"],
            "reason": r["reason"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def check_suppression(conn: sqlite3.Connection, signature: str, pattern_signature: str) -> bool:
    """A candidate finding is suppressed if a prior dismissal exactly matches its
    node-scoped signature, or matches its pattern_signature at global scope."""
    row = conn.execute(
        """
        SELECT 1 FROM feedback
        WHERE (scope = 'node' AND signature = ?)
           OR (scope = 'global' AND pattern_signature = ?)
        LIMIT 1
        """,
        (signature, pattern_signature),
    ).fetchone()
    return row is not None
