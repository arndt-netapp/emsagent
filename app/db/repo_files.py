import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

from app.db.models import FileRecord

_VALID_STATUSES = {"pending", "processing", "processed", "failed"}


def _row_to_file(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        id=row["id"],
        filename=row["filename"],
        cluster=row["cluster"],
        filepath=row["filepath"],
        file_hash=row["file_hash"],
        file_size_bytes=row["file_size_bytes"],
        detected_format=row["detected_format"],
        status=row["status"],
        error_message=row["error_message"],
        discovered_at=row["discovered_at"],
        ingested_at=row["ingested_at"],
        event_count=row["event_count"],
        duplicates_skipped=row["duplicates_skipped"] or 0,
        severity_filter=row["severity_filter"],
        severity_skipped=row["severity_skipped"] or 0,
    )


def find_by_hash(conn: sqlite3.Connection, file_hash: str) -> Optional[FileRecord]:
    row = conn.execute("SELECT * FROM files WHERE file_hash = ?", (file_hash,)).fetchone()
    return _row_to_file(row) if row else None


def insert_file(
    conn: sqlite3.Connection,
    filename: str,
    filepath: str,
    file_hash: str,
    file_size_bytes: int,
    cluster: Optional[str] = None,
    severity_filter: Optional[str] = None,
) -> FileRecord:
    """`cluster` is set for cluster pulls, which know where their events came
    from. It stays None for a file dropped in the watch directory: the text log
    format carries no cluster identity, so inventing one would be a guess.

    `severity_filter` is the floor `ingest_file` will enforce on this file's
    events. It is recorded on the row rather than passed to ingestion, so the
    file always states the rule its own event rows were selected by — whether
    ingestion happens immediately (upload, cluster fetch) or later through
    `ingest_pending_files`."""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO files (cluster, filename, filepath, file_hash, file_size_bytes,
                           status, discovered_at, severity_filter)
        VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
        """,
        (cluster, filename, filepath, file_hash, file_size_bytes, now, severity_filter),
    )
    return get_file(conn, cur.lastrowid)


def get_file(conn: sqlite3.Connection, file_id: int) -> Optional[FileRecord]:
    row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    return _row_to_file(row) if row else None


def list_files(conn: sqlite3.Connection, limit: int = 100, offset: int = 0) -> List[FileRecord]:
    rows = conn.execute(
        "SELECT * FROM files ORDER BY discovered_at DESC LIMIT ? OFFSET ?", (limit, offset)
    ).fetchall()
    return [_row_to_file(r) for r in rows]


def get_pending(conn: sqlite3.Connection) -> List[FileRecord]:
    rows = conn.execute("SELECT * FROM files WHERE status = 'pending' ORDER BY discovered_at").fetchall()
    return [_row_to_file(r) for r in rows]


def mark_status(
    conn: sqlite3.Connection,
    file_id: int,
    status: str,
    error_message: Optional[str] = None,
    detected_format: Optional[str] = None,
    event_count: Optional[int] = None,
    duplicates_skipped: Optional[int] = None,
    severity_skipped: Optional[int] = None,
) -> None:
    if status not in _VALID_STATUSES:
        raise ValueError(f"invalid file status: {status}")
    ingested_at = datetime.now(timezone.utc).isoformat() if status == "processed" else None
    conn.execute(
        """
        UPDATE files
        SET status = ?,
            error_message = ?,
            detected_format = COALESCE(?, detected_format),
            event_count = COALESCE(?, event_count),
            duplicates_skipped = COALESCE(?, duplicates_skipped),
            severity_skipped = COALESCE(?, severity_skipped),
            ingested_at = COALESCE(?, ingested_at)
        WHERE id = ?
        """,
        (
            status,
            error_message,
            detected_format,
            event_count,
            duplicates_skipped,
            severity_skipped,
            ingested_at,
            file_id,
        ),
    )


def get_most_recent_ingested(conn: sqlite3.Connection, cluster: Optional[str]) -> Optional[FileRecord]:
    """The most recently ingested file for one cluster — i.e. its most recent
    pull, since a cluster fetch writes exactly one file per fetch.

    Ordered by ingested_at, not discovered_at: a file dropped in the watch
    directory can be discovered long before it is ingested, and it is the
    ingestion that put its events in the database."""
    row = conn.execute(
        """
        SELECT * FROM files
        WHERE cluster IS ? AND status = 'processed' AND event_count > 0
        ORDER BY ingested_at DESC, id DESC
        LIMIT 1
        """,
        (cluster,),
    ).fetchone()
    return _row_to_file(row) if row else None
