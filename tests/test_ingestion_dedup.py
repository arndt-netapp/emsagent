"""Cross-fetch event deduplication.

Cluster fetches overlap by design: "give me the last 500 events" run twice
returns mostly the same events. Without dedup each re-fetch re-inserts them,
and because compaction collapses adjacent identical events into a count, a
duplicated event reports as having fired twice — manufacturing exactly the
event-storm signal Stage 1 exists to find.
"""

import sqlite3
from pathlib import Path

import pytest

from app.agent import compaction
from app.db import repo_events, repo_files
from app.services.ingestion_service import _drop_already_ingested, ingest_file

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "app" / "db" / "schema.sql"

LINE = "{time} [{node}: {name}:error]: {name}: Failover monitor: takeover disabled."


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text())
    return connection


def _write_log(tmp_path, name, events):
    path = tmp_path / name
    path.write_text(
        "\n".join(LINE.format(time=t, node=n, name=e) for t, n, e in events) + "\n",
        encoding="utf-8",
    )
    return path


def _ingest(conn, tmp_path, name, events, cluster="prod-a"):
    path = _write_log(tmp_path, name, events)
    record = repo_files.insert_file(conn, name, str(path), name, path.stat().st_size, cluster=cluster)
    return ingest_file(conn, record)


T1 = "2026-08-18T10:00:00+00:00"
T2 = "2026-08-18T10:05:00+00:00"
NODE = "cluster1-01"
EVENT_A = "cf.fsm.takeoverOfPartnerDisabled"
EVENT_B = "monitor.globalStatus.critical"


def test_overlapping_refetch_does_not_duplicate_events(conn, tmp_path):
    """The normal re-fetch case: the second pull repeats everything from the
    first and adds one new event."""
    _ingest(conn, tmp_path, "pull1.log", [(T1, NODE, EVENT_A)])
    second = _ingest(conn, tmp_path, "pull2.log", [(T1, NODE, EVENT_A), (T2, NODE, EVENT_B)])

    assert second.event_count == 1
    assert second.duplicates_skipped == 1
    assert repo_events.count_events(conn) == 2


def test_duplicates_do_not_inflate_compacted_counts(conn, tmp_path):
    """The reason dedup matters: a duplicated event otherwise reports as having
    fired twice, and 'fired twice in a row' is the pattern Stage 1 escalates."""
    _ingest(conn, tmp_path, "pull1.log", [(T1, NODE, EVENT_A)])
    _ingest(conn, tmp_path, "pull2.log", [(T1, NODE, EVENT_A)], cluster="prod-a")

    groups = {g.event_name: g for g in compaction.build_compact_corpus(conn)}
    assert groups[EVENT_A].count == 1


def test_genuine_storm_within_one_fetch_is_preserved(conn, tmp_path):
    """Dedup is only ever against events ALREADY in the database, never within
    a batch. A real storm is a run of identical events at the same second on
    the same node, and the log format preserves nothing that distinguishes
    those from re-fetched copies — so deduplicating inside a file would flatten
    a real 50-event storm to one row and destroy the signal."""
    storm = [(T1, NODE, "disk.smart.error")] * 5
    result = _ingest(conn, tmp_path, "storm.log", storm)

    assert result.event_count == 5
    assert result.duplicates_skipped == 0
    groups = {g.event_name: g for g in compaction.build_compact_corpus(conn)}
    assert groups["disk.smart.error"].count == 5


def test_dedup_is_per_cluster(conn, tmp_path):
    """Two clusters can legitimately emit the same event at the same second —
    ONTAP even names their nodes identically. Cross-cluster dedup would drop
    the second cluster's real event."""
    _ingest(conn, tmp_path, "a.log", [(T1, NODE, EVENT_A)], cluster="prod-a")
    second = _ingest(conn, tmp_path, "b.log", [(T1, NODE, EVENT_A)], cluster="prod-b")

    assert second.event_count == 1
    assert second.duplicates_skipped == 0


def test_untimestamped_events_are_still_deduped(conn, tmp_path):
    """Events whose timestamp didn't parse can't be range-filtered, so the key
    query includes them explicitly rather than treating them as non-colliding."""
    events = [
        {"raw_line": "r", "event_time": None, "node": NODE, "event_name": EVENT_A,
         "severity": "error", "message": "m", "sequence_num": 1}
    ]
    record = repo_files.insert_file(conn, "a.log", "/w/a.log", "h1", 0, cluster="prod-a")
    repo_events.bulk_insert_events(conn, record.id, events, cluster="prod-a")

    kept, skipped = _drop_already_ingested(conn, "prod-a", list(events))
    assert kept == []
    assert skipped == 1


def test_sequence_numbers_stay_dense_after_dropping(conn, tmp_path):
    """compaction orders by (event_time, sequence_num); gaps left by dropped
    duplicates would be meaningless noise in that ordering."""
    _ingest(conn, tmp_path, "pull1.log", [(T1, NODE, EVENT_A)])
    _ingest(conn, tmp_path, "pull2.log", [(T1, NODE, EVENT_A), (T2, NODE, EVENT_B)])

    kept = [e for e in repo_events.get_all_events_ordered(conn) if e.event_name == EVENT_B]
    assert kept[0].sequence_num == 1


def test_first_ingestion_is_unaffected(conn, tmp_path):
    """An empty database means no key set to compare against — the fast path
    must not change behavior."""
    result = _ingest(conn, tmp_path, "pull1.log", [(T1, NODE, EVENT_A), (T2, NODE, EVENT_B)])
    assert result.event_count == 2
    assert result.duplicates_skipped == 0
