import gzip
import json
import sqlite3
from pathlib import Path

import pytest

from app.db import repo_catalog, repo_files

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "app" / "db" / "schema.sql"

CATALOG_RECORDS = [
    {
        "name": "cf.fsm.takeoverOfPartnerDisabled",
        "severity": "error",
        "description": "This message occurs when the failover monitor determines that takeover of the partner is disabled.",
        "corrective_action": "Use the 'storage failover show' command to check the failover state.",
        "snmp_trap_type": "severity_based",
        "deprecated": False,
    },
    {
        "name": "disk.predictiveFailure",
        "severity": "alert",
        "description": "This message occurs when a disk reports a predictive failure.",
        "corrective_action": "Replace the disk.",
        "snmp_trap_type": "standard",
        "deprecated": False,
    },
    {
        "name": "callhome.reboot",
        "severity": "notice",
        "description": "x" * 500,
        "corrective_action": "None required.",
        "snmp_trap_type": "built_in",
        "deprecated": True,
    },
]


@pytest.fixture
def catalog_fixture(tmp_path):
    path = tmp_path / "ems_catalog.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"records": CATALOG_RECORDS, "num_records": len(CATALOG_RECORDS)}, f)
    return path


@pytest.fixture
def conn(catalog_fixture):
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text())
    repo_catalog.load_catalog(connection, catalog_fixture)
    return connection


def test_load_populates_catalog(conn):
    assert repo_catalog.count_catalog(conn) == 3


def test_load_is_idempotent(conn, catalog_fixture):
    """load_catalog runs on every init_db, i.e. every connection. If it
    re-inserted each time, opening a connection would rewrite 8000 rows."""
    assert repo_catalog.load_catalog(conn, catalog_fixture) == 0
    assert repo_catalog.count_catalog(conn) == 3


def test_load_missing_fixture_is_not_fatal(tmp_path):
    """The app is fully usable without a catalog — the agent just loses its
    event definitions. A missing fixture must not break database init."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text())
    assert repo_catalog.load_catalog(connection, tmp_path / "nope.json.gz") == 0


def test_lookup_is_case_insensitive(conn):
    """ONTAP's catalog casing isn't always how a name appears in a log line,
    and a case-sensitive miss is indistinguishable from 'no such event' —
    which would teach the model the catalog is useless."""
    records = repo_catalog.lookup(conn, ["CF.FSM.TAKEOVEROFPARTNERDISABLED"])
    assert len(records) == 1
    assert records[0]["name"] == "cf.fsm.takeoverOfPartnerDisabled"
    assert "storage failover show" in records[0]["corrective_action"]


def test_lookup_unknown_name_returns_empty(conn):
    assert repo_catalog.lookup(conn, ["not.a.real.event"]) == []


def test_lookup_clamps_to_max_results(conn):
    """The clamp is what keeps a definition lookup from becoming the same
    unbounded-tool-result hole through the cost cap that event queries were."""
    names = [f"bogus.event.{i}" for i in range(50)] + ["disk.predictiveFailure"]
    records = repo_catalog.lookup(conn, names)
    # The real name sits past the clamp, so it is correctly not returned.
    assert len(records) <= repo_catalog.MAX_LOOKUP_RESULTS
    assert records == []


def test_lookup_dedupes_before_clamping(conn):
    repeated = ["disk.predictiveFailure"] * 40
    assert len(repo_catalog.lookup(conn, repeated)) == 1


def test_search_matches_substring(conn):
    records = repo_catalog.search(conn, "takeover")
    assert [r["name"] for r in records] == ["cf.fsm.takeoverOfPartnerDisabled"]


def test_search_respects_ceiling(conn):
    assert len(repo_catalog.search(conn, ".", limit=999)) <= repo_catalog.MAX_LOOKUP_RESULTS


def test_glossary_is_scoped_to_ingested_events(conn):
    """The glossary rides in Stage 1's system block, so it must be bounded by
    how many DISTINCT event names the corpus contains, not by the catalog's
    8000 entries."""
    file_record = repo_files.insert_file(conn, "seed.log", "seed://1", "hash1", 0)
    conn.execute(
        """
        INSERT INTO events (file_id, raw_line, event_time, node, event_name, severity, message,
                            sequence_num, parse_confidence, created_at)
        VALUES (?, 'raw', '2026-08-10T01:00:00+00:00', 'node1', 'disk.predictiveFailure', 'alert',
                'msg', 1, 'high', '2026-01-01T00:00:00+00:00')
        """,
        (file_record.id,),
    )
    entries = repo_catalog.glossary_for_events(conn)
    assert [e["name"] for e in entries] == ["disk.predictiveFailure"]


def test_glossary_truncates_long_descriptions(conn):
    file_record = repo_files.insert_file(conn, "seed.log", "seed://2", "hash2", 0)
    conn.execute(
        """
        INSERT INTO events (file_id, raw_line, event_time, node, event_name, severity, message,
                            sequence_num, parse_confidence, created_at)
        VALUES (?, 'raw', '2026-08-10T01:00:00+00:00', 'node1', 'callhome.reboot', 'notice',
                'msg', 1, 'high', '2026-01-01T00:00:00+00:00')
        """,
        (file_record.id,),
    )
    entries = repo_catalog.glossary_for_events(conn, max_chars=80)
    assert len(entries[0]["description"]) <= 80
    assert entries[0]["description"].endswith("…")
