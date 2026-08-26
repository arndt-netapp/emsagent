import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.db import repo_events, repo_files
from app.db.models import FileRecord
from app.parsing.ems_parser import parse_file
from app.severity import partition as partition_by_severity

logger = logging.getLogger(__name__)


def _to_utc_iso(event_time: Optional[datetime]) -> Optional[str]:
    """Render a parsed timestamp as an ISO string in UTC.

    Every downstream comparison of `event_time` is a TEXT comparison — SQLite
    has no date type, so `ORDER BY event_time`, `_scope_clause`'s
    `event_time >= ?`, the dedup `BETWEEN`, and `MIN`/`MAX` all sort strings.
    Lexicographic order over ISO strings is chronological order only if every
    string shares one offset, and the two parsers do not: `ems_text` carries
    whatever ONTAP sent (normally Z/+00:00) while `autosupport_format`
    preserves the bundle's local offset (e.g. -0700). Mix a fetch and a bundle
    in one cluster and compaction's adjacency pass runs over a wrong ordering,
    fabricating the counts and time spans it exists to get right.

    Naive datetimes are left EXACTLY as they are, and that guard is the whole
    subtlety here: `astimezone()` on a naive datetime silently assumes the
    machine's local zone, so "normalizing" one would shift it by whatever the
    developer's box happens to be set to. No offset means no offset."""
    if event_time is None:
        return None
    if event_time.tzinfo is None:
        return event_time.isoformat()
    return event_time.astimezone(timezone.utc).isoformat()


def _to_event_dicts(parsed_events) -> List[Dict[str, Any]]:
    return [
        {
            "raw_line": e.raw_line,
            "event_time": _to_utc_iso(e.event_time),
            "node": e.node,
            "event_name": e.event_name,
            "severity": e.severity,
            "message": e.message,
            "sequence_num": i,
            "parse_confidence": e.parse_confidence,
        }
        for i, e in enumerate(parsed_events, start=1)
    ]


def _drop_already_ingested(
    conn: sqlite3.Connection, cluster: Optional[str], event_dicts: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """Remove events a PREVIOUS ingestion already stored for this cluster.

    Cluster fetches overlap by design — "give me the last 500 events" run twice
    returns mostly the same events — and without this every re-fetch inserts
    them again. That is worse than a wasted row: compaction collapses adjacent
    identical events into a count, so a duplicated event reports as having
    fired twice, manufacturing exactly the event-storm signal Stage 1 is
    looking for, and skewing get_event_rate_baseline with it.

    Duplicates are only dropped against events already in the database, never
    within this batch. That distinction is deliberate: a genuine storm is a run
    of identical events at the same second on the same node, and the log format
    preserves nothing to tell those apart from re-fetched copies. Deduplicating
    within a file would flatten a real 50-event storm to a single row and lose
    the signal; deduplicating across files removes only what a prior fetch
    already recorded. A UNIQUE index would have had the same flattening
    problem, which is why there isn't one."""
    if not event_dicts:
        return event_dicts, 0
    times = [e.get("event_time") for e in event_dicts if e.get("event_time")]
    existing = repo_events.existing_event_keys(
        conn, cluster, min(times) if times else None, max(times) if times else None
    )
    if not existing:
        return event_dicts, 0
    kept = []
    skipped = 0
    for event in event_dicts:
        key = (
            event.get("node"),
            event.get("event_time"),
            event["event_name"],
            event.get("severity"),
            event.get("message"),
        )
        if key in existing:
            skipped += 1
            continue
        kept.append(event)
    # Renumber so sequence_num stays dense within the file; compaction orders
    # by (event_time, sequence_num) and gaps would be meaningless noise.
    for i, event in enumerate(kept, start=1):
        event["sequence_num"] = i
    return kept, skipped


def ingest_file(conn: sqlite3.Connection, file_record: FileRecord) -> FileRecord:
    repo_files.mark_status(conn, file_record.id, "processing")
    try:
        format_name, parsed_events = parse_file(Path(file_record.filepath))
        # The floor is read off the file row, not passed in: whoever registered
        # the file chose it, and enforcing it here means the row's stated
        # filter is true of its events no matter which path ingested them.
        #
        # Applied on this path even for a cluster fetch, whose events were
        # already filtered cluster-side by the same floor. The duplication is
        # the point — it costs a rank lookup per event and makes
        # severity_filter a guarantee rather than a record of what was
        # requested, so an ONTAP version that ignores or mishandles the query
        # can't leave the row claiming a filter that never applied.
        parsed_events, severity_skipped = partition_by_severity(
            parsed_events, file_record.severity_filter
        )
        if severity_skipped:
            logger.info(
                "file %s: dropped %s event(s) below severity %s",
                file_record.filename,
                severity_skipped,
                file_record.severity_filter,
            )
        event_dicts = _to_event_dicts(parsed_events)
        event_dicts, duplicates = _drop_already_ingested(conn, file_record.cluster, event_dicts)
        if duplicates:
            logger.info(
                "file %s: skipped %s event(s) already ingested for cluster %s",
                file_record.filename,
                duplicates,
                file_record.cluster or "unspecified",
            )
        # Cluster identity travels from the file to every event row it
        # produced; this is the only point where the two are connected.
        repo_events.bulk_insert_events(conn, file_record.id, event_dicts, cluster=file_record.cluster)
        repo_files.mark_status(
            conn,
            file_record.id,
            "processed",
            detected_format=format_name,
            event_count=len(event_dicts),
            duplicates_skipped=duplicates,
            severity_skipped=severity_skipped,
        )
    except Exception as exc:  # noqa: BLE001 - a bad file must not abort the batch
        repo_files.mark_status(conn, file_record.id, "failed", error_message=str(exc))
    return repo_files.get_file(conn, file_record.id)


def ingest_pending_files(conn: sqlite3.Connection, file_ids: Optional[List[int]] = None) -> Dict[str, int]:
    pending = repo_files.get_pending(conn)
    if file_ids is not None:
        pending = [f for f in pending if f.id in file_ids]

    processed = 0
    failed = 0
    event_count = 0
    for file_record in pending:
        result = ingest_file(conn, file_record)
        if result.status == "processed":
            processed += 1
            event_count += result.event_count
        else:
            failed += 1
    return {"processed": processed, "failed": failed, "event_count": event_count}
