import json
import sqlite3
from pathlib import Path

import pytest

from app.agent.tools import InvestigationContext, build_tools
from app.db import repo_agent_steps, repo_analysis_runs, repo_candidates, repo_files

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "app" / "db" / "schema.sql"


@pytest.fixture
def conn():
    # check_same_thread=False mirrors app/db/session.py. Load-bearing here:
    # LangGraph's ToolNode runs tools on worker threads, so a fixture without
    # it tests a connection the app never creates — and would mask the
    # cross-thread race the concurrency test below exists to pin down.
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_PATH.read_text())
    file_record = repo_files.insert_file(connection, "seed.log", "seed://1", "hash1", 0)
    connection.execute(
        """
        INSERT INTO events (file_id, raw_line, event_time, node, event_name, severity, message,
                            sequence_num, parse_confidence, created_at)
        VALUES (?, 'raw', '2026-08-10T01:00:00+00:00', 'node1', 'disk.smart.error', 'warning',
                'msg', 1, 'high', '2026-01-01T00:00:00+00:00')
        """,
        (file_record.id,),
    )
    connection.commit()
    return connection


@pytest.fixture
def candidate_id(conn):
    run = repo_analysis_runs.start_run(conn)
    repo_candidates.bulk_insert_candidates(
        conn,
        run.id,
        [
            repo_candidates.CandidateInput(
                rank=1,
                category="predictive_failure",
                node="node1",
                rationale="disk warnings",
                confidence=0.8,
                refs=[1],
                leads=[],
                status="pending",
            )
        ],
    )
    conn.commit()
    return repo_candidates.list_candidates_for_run(conn, run.id)[0].id


def test_record_and_list_roundtrip(conn, candidate_id):
    repo_agent_steps.record_step(
        conn,
        candidate_id=candidate_id,
        iteration=1,
        step_index=1,
        tool_name="query_events",
        tool_args={"node": "node1"},
        result_summary="2 events",
        duration_ms=12,
    )
    steps = repo_agent_steps.list_steps_for_candidate(conn, candidate_id)
    assert len(steps) == 1
    assert steps[0]["tool_name"] == "query_events"
    assert steps[0]["tool_args"] == {"node": "node1"}
    assert steps[0]["duration_ms"] == 12


def test_long_results_are_truncated(conn, candidate_id):
    """These rows exist to make an investigation auditable, not to be a second
    copy of the evidence — every query would otherwise store a 100-event blob."""
    repo_agent_steps.record_step(
        conn,
        candidate_id=candidate_id,
        iteration=1,
        step_index=1,
        tool_name="query_events",
        result_summary="x" * 10000,
    )
    stored = repo_agent_steps.list_steps_for_candidate(conn, candidate_id)[0]["result_summary"]
    assert len(stored) == repo_agent_steps.MAX_STORED_RESULT_CHARS
    assert stored.endswith("…")


def test_recording_never_raises(conn):
    """A failure to write a trace row must not take down the investigation it
    is only observing — a lost trace row is cheap, a lost investigation is not.
    candidate_id 99999 violates the foreign key."""
    repo_agent_steps.record_step(
        conn, candidate_id=99999, iteration=1, step_index=1, tool_name="query_events"
    )
    assert repo_agent_steps.list_steps_for_candidate(conn, 99999) == []


def test_truncated_args_still_list(conn, candidate_id):
    """Args truncated mid-JSON can't be parsed back; the row must still render
    rather than disappearing from the trace."""
    repo_agent_steps.record_step(
        conn,
        candidate_id=candidate_id,
        iteration=1,
        step_index=1,
        tool_name="query_events",
        tool_args={"pattern": "y" * 5000},
    )
    step = repo_agent_steps.list_steps_for_candidate(conn, candidate_id)[0]
    assert "_raw" in step["tool_args"]


def test_tools_write_a_trace_when_given_a_context(conn, candidate_id):
    """The end-to-end path: a traced tool call lands in agent_steps tagged with
    the candidate and turn it belongs to."""
    trace = InvestigationContext(candidate_id=candidate_id)
    trace.iteration = 3
    tools = {t.name: t for t in build_tools(conn, trace=trace)}
    tools["query_events"].invoke({"node": "node1"})

    steps = repo_agent_steps.list_steps_for_candidate(conn, candidate_id)
    assert len(steps) == 1
    assert steps[0]["tool_name"] == "query_events"
    assert steps[0]["iteration"] == 3
    assert steps[0]["tool_args"] == {"node": "node1"}
    assert "disk.smart.error" in steps[0]["result_summary"]


def test_command_returning_tools_are_traced_by_update_keys(conn, candidate_id):
    """record_hypothesis returns a Command carrying a state update rather than
    text; the trace should still show what it did."""
    trace = InvestigationContext(candidate_id=candidate_id)
    tools = {t.name: t for t in build_tools(conn, trace=trace)}
    tools["conclude_investigation"].invoke(
        {"type": "tool_call", "name": "conclude_investigation", "args": {"summary": "Nothing to pursue."}, "id": "call_1"}
    )
    step = repo_agent_steps.list_steps_for_candidate(conn, candidate_id)[0]
    assert "ready_to_conclude" in step["result_summary"]


def test_tools_work_without_a_trace_context(conn):
    """build_tools(conn) with no trace is still a supported call — tracing is
    an addition, not a requirement."""
    tools = {t.name: t for t in build_tools(conn)}
    assert "disk.smart.error" in tools["query_events"].invoke({"node": "node1"})


def test_concurrent_tool_calls_do_not_corrupt_the_connection(conn, candidate_id):
    """LangGraph's ToolNode dispatches a turn's tool calls to a threadpool, so
    two tool calls from one model turn hit this single sqlite3 connection from
    two threads. check_same_thread=False silences Python's check but does not
    make the connection thread-safe — before build_tools serialized them, this
    took the whole interpreter down with a native SIGSEGV about 1 run in 15.

    Uses many more threads than a real turn would to make the race loud."""
    import threading

    trace = InvestigationContext(candidate_id=candidate_id)
    tools = {t.name: t for t in build_tools(conn, trace=trace)}
    errors = []

    def hammer():
        try:
            for _ in range(15):
                tools["query_events"].invoke({"node": "node1"})
        except Exception as exc:  # noqa: BLE001 - surfaced via the assert below
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    steps = repo_agent_steps.list_steps_for_candidate(conn, candidate_id)
    assert len(steps) == 8 * 15
    # Every write landed: none were silently swallowed by record_step's
    # never-raise guard because the connection was busy on another thread.
    assert all(s["tool_name"] == "query_events" for s in steps)
