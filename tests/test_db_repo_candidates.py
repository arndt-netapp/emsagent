import sqlite3
from pathlib import Path

import pytest

from app.db import repo_candidates

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "app" / "db" / "schema.sql"


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text())
    connection.execute("INSERT INTO analysis_runs (status, started_at) VALUES ('completed', '2026-01-01T00:00:00+00:00')")
    connection.commit()
    return connection


def _insert_one(conn, rank=1, refs=None):
    [c] = repo_candidates.bulk_insert_candidates(
        conn,
        1,
        [
            repo_candidates.CandidateInput(
                rank=rank,
                category="predictive_failure",
                node="node1",
                rationale="test rationale",
                confidence=0.7,
                refs=refs or [1, 2],
            )
        ],
    )
    return c


def test_bulk_insert_preserves_rank_order_and_round_trips_refs(conn):
    candidates = repo_candidates.bulk_insert_candidates(
        conn,
        1,
        [
            repo_candidates.CandidateInput(rank=2, category="performance_issue", node="node2", rationale="r2", confidence=0.5, refs=[3]),
            repo_candidates.CandidateInput(rank=1, category="predictive_failure", node="node1", rationale="r1", confidence=0.9, refs=[1, 2]),
        ],
    )
    conn.commit()

    fetched = repo_candidates.list_candidates_for_run(conn, 1)
    assert [c.rank for c in fetched] == [1, 2]
    assert fetched[0].refs == [1, 2]
    assert fetched[1].refs == [3]
    assert candidates[0].status == "pending"


def test_start_investigation_transitions_pending_to_investigating(conn):
    c = _insert_one(conn)
    conn.commit()

    repo_candidates.start_investigation(conn, c.id)

    fetched = repo_candidates.get_candidate(conn, c.id)
    assert fetched.status == "investigating"
    assert fetched.investigation_started_at is not None


def test_start_investigation_raises_when_not_pending(conn):
    c = _insert_one(conn)
    conn.commit()
    repo_candidates.start_investigation(conn, c.id)

    with pytest.raises(ValueError):
        repo_candidates.start_investigation(conn, c.id)


def test_complete_investigation_sets_status_and_usage(conn):
    c = _insert_one(conn)
    conn.commit()
    repo_candidates.start_investigation(conn, c.id)

    repo_candidates.complete_investigation(conn, c.id, input_tokens=1000, output_tokens=2000, iterations=3)

    fetched = repo_candidates.get_candidate(conn, c.id)
    assert fetched.status == "investigated"
    assert fetched.investigation_input_tokens == 1000
    assert fetched.investigation_output_tokens == 2000
    assert fetched.investigation_iterations == 3
    assert fetched.investigation_completed_at is not None


def test_fail_investigation_records_error(conn):
    c = _insert_one(conn)
    conn.commit()
    repo_candidates.start_investigation(conn, c.id)

    repo_candidates.fail_investigation(conn, c.id, "boom")

    fetched = repo_candidates.get_candidate(conn, c.id)
    assert fetched.status == "investigated"
    assert fetched.investigation_error == "boom"


def test_fail_investigation_records_tokens_already_spent(conn):
    """An investigation that dies part-way through was still billed for the
    turns it completed. Leaving the columns NULL reported real spend as no
    spend, which is precisely backwards for a cost-capped feature."""
    c = _insert_one(conn)
    conn.commit()
    repo_candidates.start_investigation(conn, c.id)

    repo_candidates.fail_investigation(
        conn,
        c.id,
        "boom",
        input_tokens=1200,
        output_tokens=3400,
        iterations=2,
        cache_creation_input_tokens=500,
        cache_read_input_tokens=9000,
    )

    fetched = repo_candidates.get_candidate(conn, c.id)
    assert fetched.investigation_input_tokens == 1200
    assert fetched.investigation_output_tokens == 3400
    assert fetched.investigation_iterations == 2
    assert fetched.investigation_cache_creation_input_tokens == 500
    assert fetched.investigation_cache_read_input_tokens == 9000


def test_bulk_insert_round_trips_resolved_leads(conn):
    lead = {
        "ref": 1,
        "node": "node1",
        "event_name": "disk.predictiveFailure",
        "severity": "alert",
        "count": 3,
        "first_event_id": 10,
        "last_event_id": 12,
        "first_time": "2026-08-10T01:00:00-04:00",
        "last_time": "2026-08-10T01:02:00-04:00",
    }
    [c] = repo_candidates.bulk_insert_candidates(
        conn,
        1,
        [
            repo_candidates.CandidateInput(
                rank=1,
                category="predictive_failure",
                node="node1",
                rationale="r",
                confidence=0.9,
                refs=[1],
                leads=[lead],
            )
        ],
    )
    conn.commit()

    assert repo_candidates.get_candidate(conn, c.id).leads == [lead]


def test_candidate_without_stored_leads_reads_back_as_empty(conn):
    """Rows created before the `leads` column existed read back as [] rather
    than None, so graph.load_context's legacy-fallback check stays simple."""
    c = _insert_one(conn)
    conn.execute("UPDATE candidates SET leads = NULL WHERE id = ?", (c.id,))
    conn.commit()

    assert repo_candidates.get_candidate(conn, c.id).leads == []


def test_discard_transitions_pending_to_discarded_with_reason(conn):
    c = _insert_one(conn)
    conn.commit()

    repo_candidates.discard(conn, c.id, reason="already known")

    fetched = repo_candidates.get_candidate(conn, c.id)
    assert fetched.status == "discarded"
    assert fetched.discard_reason == "already known"


def test_discard_raises_when_not_pending(conn):
    c = _insert_one(conn)
    conn.commit()
    repo_candidates.discard(conn, c.id)

    with pytest.raises(ValueError):
        repo_candidates.discard(conn, c.id)


def test_get_candidate_returns_none_for_unknown_id(conn):
    assert repo_candidates.get_candidate(conn, 9999) is None
