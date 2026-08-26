import sqlite3
from pathlib import Path

import pytest

from app.agent import compaction
from app.db import repo_events, repo_files

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "app" / "db" / "schema.sql"


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text())
    return connection


def _insert(conn, file_id, seq, node, event_name, severity, event_time):
    repo_events.bulk_insert_events(
        conn,
        file_id,
        [
            {
                "raw_line": "raw",
                "event_time": event_time,
                "node": node,
                "event_name": event_name,
                "severity": severity,
                "message": "msg",
                "sequence_num": seq,
            }
        ],
    )


def test_no_duplicates_produces_one_group_per_event(conn):
    f = repo_files.insert_file(conn, "seed.log", "seed://1", "hash1", 0)
    _insert(conn, f.id, 1, "node1", "disk.smart.error", "warning", "2026-08-17T00:00:00+00:00")
    _insert(conn, f.id, 2, "node1", "disk.predictiveFailure", "alert", "2026-08-17T00:01:00+00:00")
    conn.commit()

    groups = compaction.build_compact_corpus(conn)

    assert len(groups) == 2
    assert all(g.count == 1 for g in groups)


def test_consecutive_identical_events_on_same_node_collapse(conn):
    f = repo_files.insert_file(conn, "seed.log", "seed://1", "hash1", 0)
    for i in range(1, 51):
        _insert(conn, f.id, i, "node1", "cf.fsm.takeoverOfPartnerDisabled", "error", f"2026-08-17T00:{i:02d}:00+00:00")
    conn.commit()

    groups = compaction.build_compact_corpus(conn)

    assert len(groups) == 1
    assert groups[0].count == 50
    assert groups[0].first_time == "2026-08-17T00:01:00+00:00"
    assert groups[0].last_time == "2026-08-17T00:50:00+00:00"


def test_different_severity_breaks_the_run(conn):
    f = repo_files.insert_file(conn, "seed.log", "seed://1", "hash1", 0)
    _insert(conn, f.id, 1, "node1", "disk.smart.error", "warning", "2026-08-17T00:00:00+00:00")
    _insert(conn, f.id, 2, "node1", "disk.smart.error", "alert", "2026-08-17T00:01:00+00:00")
    _insert(conn, f.id, 3, "node1", "disk.smart.error", "warning", "2026-08-17T00:02:00+00:00")
    conn.commit()

    groups = compaction.build_compact_corpus(conn)

    assert [g.count for g in groups] == [1, 1, 1]


def test_different_event_on_same_node_breaks_the_run(conn):
    f = repo_files.insert_file(conn, "seed.log", "seed://1", "hash1", 0)
    _insert(conn, f.id, 1, "node1", "cf.fsm.takeoverOfPartnerDisabled", "error", "2026-08-17T00:00:00+00:00")
    _insert(conn, f.id, 2, "node1", "perf.ccma.off", "alert", "2026-08-17T00:01:00+00:00")
    _insert(conn, f.id, 3, "node1", "cf.fsm.takeoverOfPartnerDisabled", "error", "2026-08-17T00:02:00+00:00")
    conn.commit()

    groups = compaction.build_compact_corpus(conn)

    assert [g.count for g in groups] == [1, 1, 1]
    assert [g.event_name for g in groups] == [
        "cf.fsm.takeoverOfPartnerDisabled",
        "perf.ccma.off",
        "cf.fsm.takeoverOfPartnerDisabled",
    ]


def test_other_node_interleaving_does_not_break_the_run(conn):
    f = repo_files.insert_file(conn, "seed.log", "seed://1", "hash1", 0)
    _insert(conn, f.id, 1, "node1", "cf.fsm.takeoverOfPartnerDisabled", "error", "2026-08-17T00:00:00+00:00")
    _insert(conn, f.id, 2, "node2", "perf.ccma.off", "alert", "2026-08-17T00:00:30+00:00")
    _insert(conn, f.id, 3, "node1", "cf.fsm.takeoverOfPartnerDisabled", "error", "2026-08-17T00:01:00+00:00")
    conn.commit()

    groups = compaction.build_compact_corpus(conn)

    # node1's run collapses to one group of count=2 despite node2's event
    # being interleaved in time between the two node1 firings.
    node1_groups = [g for g in groups if g.node == "node1"]
    assert len(node1_groups) == 1
    assert node1_groups[0].count == 2
    assert any(g.node == "node2" for g in groups)


def test_render_compact_corpus_format(conn):
    f = repo_files.insert_file(conn, "seed.log", "seed://1", "hash1", 0)
    _insert(conn, f.id, 1, "node1", "disk.smart.error", "warning", "2026-08-17T00:00:00+00:00")
    conn.commit()

    groups = compaction.build_compact_corpus(conn)
    text = compaction.render_compact_corpus(groups)

    lines = text.split("\n")
    assert lines[0] == "ref|first_time|last_time|node|event_name|severity|count"
    assert lines[1] == "1|2026-08-17T00:00:00+00:00|2026-08-17T00:00:00+00:00|node1|disk.smart.error|warning|1"
