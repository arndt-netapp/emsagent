"""Validates the LangGraph wiring (investigate <-> tools loop, termination,
suppression enforcement, the hard cost cap) against a scripted fake model,
without calling the real Anthropic API."""
import sqlite3
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent import compaction
from app.agent.findings import compute_pattern_signature, compute_signature
from app.agent.pricing import SONNET_5_OUTPUT_PRICE_PER_MTOK, STAGE2_COST_CAP_USD
from app.db import repo_candidates, repo_feedback, repo_files, repo_findings
from app.db.models import EventScope
from app.db.repo_findings import MAX_EVIDENCE_IDS

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "app" / "db" / "schema.sql"

CATEGORY = "predictive_failure"
EVENT_NAMES = ["disk.predictiveFailure"]
NODE = "node1"


class FakeBoundModel:
    def __init__(self, responses: List[AIMessage]):
        self._responses = responses
        self._i = 0

    def invoke(self, messages, **kwargs):
        response = self._responses[self._i]
        self._i += 1
        return response


class FakeChatAnthropic:
    """Stands in for langchain_anthropic.ChatAnthropic: same construction
    signature and .bind_tools(...) -> object with .invoke(...)."""

    def __init__(self, *args, **kwargs):
        pass

    def bind_tools(self, tools):
        return FakeBoundModel(_scripted_responses())


def _tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id}


def _usage_costing(fraction_of_cap):
    """A usage_metadata dict for one model turn costing
    `fraction_of_cap` x STAGE2_COST_CAP_USD, all of it in output tokens.

    Derived from the cap rather than hardcoded: these tests are about the
    cap's behaviour, and token counts tuned to one particular ceiling stop
    exercising it the moment the ceiling moves — a turn budget that no longer
    crosses the cap turns a cost-cap test into a test of whatever happens to
    run out first."""
    dollars = STAGE2_COST_CAP_USD * fraction_of_cap
    output_tokens = int(dollars / SONNET_5_OUTPUT_PRICE_PER_MTOK * 1_000_000)
    return {"input_tokens": 0, "output_tokens": output_tokens, "total_tokens": output_tokens}


def _scripted_responses() -> List[AIMessage]:
    return [
        AIMessage(content="", tool_calls=[_tool_call("query_events", {"node": NODE}, "c1")]),
        AIMessage(
            content="",
            tool_calls=[_tool_call("check_suppression", {"category": CATEGORY, "event_names": EVENT_NAMES, "node": NODE}, "c2")],
        ),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "record_hypothesis",
                    {
                        "category": CATEGORY,
                        "severity": "high",
                        "title": "Disk predictive failure on node1",
                        "description": "Escalating SMART warnings culminated in a predictive failure event.",
                        "node": NODE,
                        "event_ids": [1],
                        "event_names": EVENT_NAMES,
                        "confidence": 0.9,
                        "status": "ready",
                        "recommendation": "Replace the flagged disk.",
                    },
                    "c3",
                )
            ],
        ),
        AIMessage(content="", tool_calls=[_tool_call("conclude_investigation", {"summary": "Concluded after reviewing the evidence.", "outcome": "confirmed"}, "c4")]),
    ]


def _conclude(outcome, summary="Confirmed: SMART errors on node1 are escalating.", call_id="c2"):
    """One conclude_investigation turn. `outcome` is the axis that decides
    whether a refutation is written, so it is always stated explicitly."""
    return AIMessage(
        content="",
        tool_calls=[_tool_call("conclude_investigation", {"summary": summary, "outcome": outcome}, call_id)],
    )


@pytest.fixture
def conn():
    # check_same_thread=False mirrors app/db/session.py, and it is load-bearing
    # here rather than cosmetic: LangGraph's ToolNode executes tool functions on
    # a worker thread, not the thread that built the graph. Any tool that
    # touches the database — which now includes every tool, via the agent_steps
    # tracing wrapper — is therefore running cross-thread, and sqlite3's default
    # same-thread check rejects it. A fixture without this silently loses every
    # traced write (record_step swallows sqlite3.Error by design) and tests a
    # connection the app never actually creates.
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    # Foreign keys are OFF by default in SQLite and ON in app/db/session.py, so
    # a fixture without this is a connection the app never creates — and the
    # difference is not academic: finding_evidence.event_id is a foreign key,
    # so a hypothesis citing an invented event id fails in production and
    # passes here. Same class of mismatch as check_same_thread above.
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_PATH.read_text())
    file_record = repo_files.insert_file(connection, "seed.log", "seed://1", "hash1", 0)
    connection.execute(
        """
        INSERT INTO events (file_id, raw_line, event_time, node, event_name, severity, message, sequence_num, parse_confidence, created_at)
        VALUES (?, 'raw', '2026-08-10T01:10:00-04:00', ?, ?, 'alert', 'Disk predictively failed.', 1, 'high', '2026-01-01T00:00:00+00:00')
        """,
        (file_record.id, NODE, "disk.predictiveFailure"),
    )
    connection.commit()
    return connection


@pytest.fixture
def candidate_id(conn):
    """A pending candidate seeded on its own analysis run, as Stage 1 would
    have produced it — Stage 2's graph is always invoked against a specific
    candidate, never a bare scope."""
    cur = conn.execute("INSERT INTO analysis_runs (status, started_at) VALUES ('running', '2026-01-01T00:00:00+00:00')")
    run_id = cur.lastrowid
    # Leads resolved against the corpus as it stands now, exactly as
    # stage1.run_stage1 does — refs alone are positional and stop meaning
    # anything once the event set changes.
    groups_by_ref = {g.ref: g for g in compaction.build_compact_corpus(conn)}
    [candidate] = repo_candidates.bulk_insert_candidates(
        conn,
        run_id,
        [
            repo_candidates.CandidateInput(
                rank=1,
                category=CATEGORY,
                node=NODE,
                rationale="Disk predictive failure flagged during triage.",
                confidence=0.8,
                refs=[1],
                leads=compaction.resolve_leads(groups_by_ref, [1]),
            )
        ],
    )
    conn.commit()
    return candidate.id


def _run_graph(conn, candidate_id, responses=None, max_iterations=15):
    class _FakeChatAnthropicWithResponses(FakeChatAnthropic):
        def bind_tools(self, tools):
            return FakeBoundModel(responses if responses is not None else _scripted_responses())

    with patch("app.agent.graph.ChatAnthropic", _FakeChatAnthropicWithResponses):
        from app.agent.graph import build_graph

        graph = build_graph(conn)
        initial_state = {
            "messages": [],
            "analysis_run_id": 1,
            "candidate_id": candidate_id,
            "iteration": 0,
            "max_iterations": max_iterations,
            "hypotheses": [],
            "finalized_findings": [],
            "findings_suppressed": 0,
            "ready_to_conclude": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        return graph.invoke(initial_state, config={"recursion_limit": 50})


def test_graph_persists_finding_when_not_suppressed(conn, candidate_id):
    final_state = _run_graph(conn, candidate_id)

    assert final_state["ready_to_conclude"] is True
    assert final_state["iteration"] == 4
    assert len(final_state["finalized_findings"]) == 1
    assert final_state["findings_suppressed"] == 0

    findings = repo_findings.list_findings(conn)
    assert len(findings) == 1
    assert findings[0].category == CATEGORY
    assert findings[0].candidate_id == candidate_id


def _seed_dismissal(conn, category=CATEGORY, event_names=EVENT_NAMES, node=NODE) -> int:
    """A human dismissal, seeded the way the app actually produces one.

    These used to pass `finding_id=0` as a placeholder, on the grounds that
    suppression only keys off the signatures. That is true of the lookup but
    not of the schema: feedback.finding_id is a foreign key, and with
    foreign_keys ON (as in production, and now in this fixture) a placeholder
    cannot be inserted at all. routes_findings only ever writes feedback
    alongside dismissing a real finding, so this mirrors it."""
    signature = compute_signature(category, event_names, node)
    pattern_signature = compute_pattern_signature(category, event_names)
    finding = repo_findings.insert_finding(
        conn,
        analysis_run_id=None,
        category=category,
        severity="high",
        title="Previously dismissed by a human",
        description="Dismissed in an earlier run.",
        recommendation=None,
        node=node,
        signature=signature,
        pattern_signature=pattern_signature,
        confidence=None,
        evidence_event_ids=[],
    )
    repo_findings.dismiss(conn, finding.id)
    repo_feedback.insert_feedback(
        conn, finding_id=finding.id, signature=signature, pattern_signature=pattern_signature
    )
    return finding.id


def _findings_besides(conn, dismissed_id):
    """Everything the run under test produced, ignoring the seeded dismissal."""
    return [f for f in repo_findings.list_findings(conn) if f.id != dismissed_id]


def test_graph_suppresses_previously_dismissed_finding(conn, candidate_id):
    dismissed_id = _seed_dismissal(conn)

    final_state = _run_graph(conn, candidate_id)

    assert final_state["findings_suppressed"] == 1
    assert final_state["finalized_findings"] == []
    assert _findings_besides(conn, dismissed_id) == []


def test_graph_handles_parallel_record_hypothesis_calls(conn, candidate_id):
    """Regression test: a single model turn issuing two record_hypothesis
    tool calls at once (parallel tool use) must not raise LangGraph's
    INVALID_CONCURRENT_GRAPH_UPDATE — both hypotheses should be recorded."""
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "record_hypothesis",
                    {
                        "category": CATEGORY,
                        "severity": "high",
                        "title": "Disk predictive failure on node1",
                        "description": "First hypothesis.",
                        "node": NODE,
                        "event_ids": [1],
                        "event_names": EVENT_NAMES,
                        "confidence": 0.9,
                        "status": "investigating",
                        "recommendation": None,
                    },
                    "c1",
                ),
                _tool_call(
                    "record_hypothesis",
                    {
                        "category": CATEGORY,
                        "severity": "medium",
                        "title": "Second candidate hypothesis",
                        "description": "Second hypothesis, recorded in the same turn as the first.",
                        "node": NODE,
                        "event_ids": [1],
                        "event_names": EVENT_NAMES,
                        "confidence": 0.5,
                        "status": "investigating",
                        "recommendation": None,
                    },
                    "c2",
                ),
            ],
        ),
        AIMessage(content="", tool_calls=[_tool_call("conclude_investigation", {"summary": "Concluded after reviewing the evidence.", "outcome": "confirmed"}, "c3")]),
    ]

    final_state = _run_graph(conn, candidate_id, responses=responses)

    assert final_state["ready_to_conclude"] is True
    assert len(final_state["hypotheses"]) == 2
    assert {h["title"] for h in final_state["hypotheses"]} == {
        "Disk predictive failure on node1",
        "Second candidate hypothesis",
    }


def test_load_context_does_not_include_full_corpus(conn, candidate_id):
    """Regression test: Stage 2's system prompt must not scale with total
    event count — dumping the full compact corpus there would let a single
    first call blow past STAGE2_COST_CAP_USD before the cap-check routing
    ever runs (see graph.py's load_context)."""
    conn.execute(
        """
        INSERT INTO events (file_id, raw_line, event_time, node, event_name, severity, message, sequence_num, parse_confidence, created_at)
        VALUES (1, 'raw', '2026-08-10T02:00:00-04:00', 'node2', 'perf.ccma.off', 'alert', 'Unrelated event.', 2, 'high', '2026-01-01T00:00:00+00:00')
        """
    )
    conn.commit()

    responses = [AIMessage(content="", tool_calls=[_tool_call("conclude_investigation", {"summary": "Concluded after reviewing the evidence.", "outcome": "confirmed"}, "c1")])]
    final_state = _run_graph(conn, candidate_id, responses=responses)

    system_text = final_state["messages"][0].content
    # The candidate's own lead (event_names=EVENT_NAMES) must be present...
    assert EVENT_NAMES[0] in "".join(str(m.content) for m in final_state["messages"][:2])
    # ...but the unrelated event elsewhere in the corpus must not have been
    # dumped into the system prompt (the node list itself, from the O(1)
    # scope summary, legitimately includes "node2" — that's fine, it's
    # bounded by distinct node count, not event count).
    assert "perf.ccma.off" not in system_text


def _seed_scoped_candidate(conn, cluster):
    """A candidate on a run scoped to one cluster, as analysis_service.trigger
    would have created it — the conn/candidate_id fixtures predate scoping, so a
    scope-sensitive test has to build its own."""
    file_record = repo_files.insert_file(conn, f"{cluster}.log", f"seed://{cluster}", f"hash-{cluster}", 0)
    conn.execute(
        """
        INSERT INTO events (file_id, cluster, raw_line, event_time, node, event_name, severity, message, sequence_num, parse_confidence, created_at)
        VALUES (?, ?, 'raw', '2026-08-10T01:10:00-04:00', ?, ?, 'alert', 'Disk predictively failed.', 1, 'high', '2026-01-01T00:00:00+00:00')
        """,
        (file_record.id, cluster, f"{cluster}-01", "disk.predictiveFailure"),
    )
    cur = conn.execute(
        """
        INSERT INTO analysis_runs (status, started_at, scope_mode, scope_cluster, scope_label)
        VALUES ('completed', '2026-01-01T00:00:00+00:00', 'last_24h', ?, ?)
        """,
        (cluster, f"last 24h of activity on {cluster}"),
    )
    scope = EventScope(mode="last_24h", cluster=cluster, label=f"last 24h of activity on {cluster}")
    groups_by_ref = {g.ref: g for g in compaction.build_compact_corpus(conn, scope)}
    [candidate] = repo_candidates.bulk_insert_candidates(
        conn,
        cur.lastrowid,
        [
            repo_candidates.CandidateInput(
                rank=1,
                category=CATEGORY,
                node=f"{cluster}-01",
                rationale="Disk predictive failure flagged during triage.",
                confidence=0.8,
                refs=[1],
                leads=compaction.resolve_leads(groups_by_ref, [1]),
            )
        ],
    )
    conn.commit()
    return candidate.id


def test_scope_summary_is_scoped_to_the_run_not_the_whole_database(conn):
    """The scope summary opens Stage 2's system prompt by naming the corpus it
    is working over, and it must name THIS run's corpus.

    Unscoped (which it was) it reports every node, the combined event total and
    the full time range of every cluster ever ingested — while every evidence
    tool is restricted to the run's own cluster via
    InvestigationContext.event_cluster(). That hands the agent node names its
    own queries can never return a row for, and an event count that isn't its
    corpus. See repo_events.get_scope_summary_stats' docstring, which says
    exactly this."""
    _seed_scoped_candidate(conn, "clusterA")
    candidate_b = _seed_scoped_candidate(conn, "clusterB")

    final_state = _run_graph(conn, candidate_b, responses=[_conclude("refuted", call_id="c1")])

    system_text = final_state["messages"][0].content
    assert "clusterB-01" in system_text
    assert "clusterA-01" not in system_text
    # ...and the count is this cluster's one event, not both clusters' two.
    assert "1 events" in system_text
    assert "clusterB" in system_text  # the scope label, so the model knows it is one window


def test_leads_survive_events_being_ingested_out_of_order(conn, candidate_id):
    """Regression test: refs are positional over a chronological walk of every
    ingested event, so inserting an event that sorts BEFORE the candidate's
    own renumbers every ref after it. Stage 2 used to re-derive refs against
    the current event set, which silently pointed the whole investigation
    budget at unrelated events. Resolved leads carry event ids instead."""
    # The candidate was generated when ref 1 == the disk.predictiveFailure
    # event. Now an older log file arrives, taking over ref 1.
    conn.execute(
        """
        INSERT INTO events (file_id, raw_line, event_time, node, event_name, severity, message, sequence_num, parse_confidence, created_at)
        VALUES (1, 'raw', '2026-08-09T00:00:00-04:00', 'node9', 'unrelated.backfilled.event', 'info', 'Backfilled.', 0, 'high', '2026-01-01T00:00:00+00:00')
        """
    )
    conn.commit()

    responses = [AIMessage(content="", tool_calls=[_tool_call("conclude_investigation", {"summary": "Concluded after reviewing the evidence.", "outcome": "confirmed"}, "c1")])]
    final_state = _run_graph(conn, candidate_id, responses=responses)

    seed = str(final_state["messages"][1].content)
    assert "disk.predictiveFailure" in seed
    assert "unrelated.backfilled.event" not in seed


def test_graph_reports_usage_through_the_sink_for_the_failure_path(conn, candidate_id):
    """The graph's state is only readable once invoke() returns, so a crash
    mid-loop would otherwise discard every token already spent and report the
    investigation as free. The sink is what runner.execute_investigation
    records on its except branch."""
    usage_meta = {"input_tokens": 1000, "output_tokens": 400, "total_tokens": 1400}
    responses = [
        AIMessage(content="", tool_calls=[_tool_call("query_events", {"node": NODE}, "c1")], usage_metadata=usage_meta),
        AIMessage(content="", tool_calls=[_tool_call("conclude_investigation", {"summary": "Concluded after reviewing the evidence.", "outcome": "confirmed"}, "c2")], usage_metadata=usage_meta),
    ]
    sink = {}

    class _FakeChatAnthropicWithResponses(FakeChatAnthropic):
        def bind_tools(self, tools):
            return FakeBoundModel(responses)

    with patch("app.agent.graph.ChatAnthropic", _FakeChatAnthropicWithResponses):
        from app.agent.graph import build_graph

        build_graph(conn, usage_sink=sink).invoke(
            {
                "messages": [],
                "analysis_run_id": 1,
                "candidate_id": candidate_id,
                "iteration": 0,
                "max_iterations": 15,
                "hypotheses": [],
                "finalized_findings": [],
                "findings_suppressed": 0,
                "ready_to_conclude": False,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            config={"recursion_limit": 50},
        )

    assert sink["input_tokens"] == 2000
    assert sink["output_tokens"] == 800
    assert sink["iterations"] == 2


def test_graph_stops_at_cost_cap(conn, candidate_id):
    """Regression test: the loop must stop once cumulative cost crosses
    STAGE2_COST_CAP_USD, even if the model never calls
    conclude_investigation and iterations remain under max_iterations.

    The cap gates the edge back into a *model* call, not the tools node: the
    cap-crossing turn's tool calls were already generated and billed, so they
    do run (free, local DB work) before the loop stops. That's what lets a
    record_hypothesis(status='ready') issued on the final turn still reach
    persist_findings instead of being thrown away."""
    # Three turns at 40% of the cap each cross it; a fourth is scripted but
    # must never run.
    usage = _usage_costing(0.4)
    responses = [
        AIMessage(content="", tool_calls=[_tool_call("query_events", {"node": NODE}, f"c{i}")], usage_metadata=usage)
        for i in range(1, 5)
    ]

    final_state = _run_graph(conn, candidate_id, responses=responses)

    assert final_state["iteration"] == 3  # the scripted 4th model call never happened
    assert final_state["ready_to_conclude"] is False
    tool_messages = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 3  # including the cap-crossing turn's own call


def test_graph_persists_hypothesis_recorded_on_the_cap_crossing_turn(conn, candidate_id):
    """The turn that crosses the cost cap can be the one that finalizes the
    finding. Its tool calls are already paid for, so abandoning them would
    discard a completed investigation and bill for it anyway."""
    usage = _usage_costing(1.01)  # over the cap in a single turn
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "record_hypothesis",
                    {
                        "category": CATEGORY,
                        "severity": "high",
                        "title": "Disk predictive failure on node1",
                        "description": "Recorded on the same turn that crossed the cost cap.",
                        "node": NODE,
                        "event_ids": [1],
                        "event_names": EVENT_NAMES,
                        "confidence": 0.9,
                        "status": "ready",
                        "recommendation": "Replace the flagged disk.",
                    },
                    "c1",
                )
            ],
            usage_metadata=usage,
        ),
        AIMessage(content="", tool_calls=[_tool_call("conclude_investigation", {"summary": "Concluded after reviewing the evidence.", "outcome": "confirmed"}, "c2")], usage_metadata=usage),
    ]

    final_state = _run_graph(conn, candidate_id, responses=responses)

    assert final_state["iteration"] == 1
    assert final_state["ready_to_conclude"] is False  # stopped by cost, not by concluding
    assert len(final_state["finalized_findings"]) == 1
    assert len(repo_findings.list_findings(conn)) == 1


def test_investigate_unpacks_cache_tokens_from_blended_usage_metadata(conn, candidate_id):
    """Regression test: langchain_anthropic's usage_metadata['input_tokens']
    is base_input + cache_read + cache_creation blended into one figure —
    accumulating it as-is into state['input_tokens'] and pricing it all at
    the full input rate would overcharge cache reads (~0.1x) and cache
    writes (~1.25x) at the full 1x rate, making the cost cap trigger too
    early. The graph must unpack the blend back into its parts."""
    usage = {
        "input_tokens": 10000,  # blended total (langchain_anthropic's convention)
        "output_tokens": 100,
        "total_tokens": 10100,
        "input_token_details": {"cache_read": 9000, "cache_creation": 500},
    }
    responses = [AIMessage(content="", tool_calls=[_tool_call("conclude_investigation", {"summary": "Concluded after reviewing the evidence.", "outcome": "confirmed"}, "c1")], usage_metadata=usage)]

    final_state = _run_graph(conn, candidate_id, responses=responses)

    # base input = 10000 - 9000 (cache_read) - 500 (cache_creation) = 500
    assert final_state["input_tokens"] == 500
    assert final_state["cache_read_input_tokens"] == 9000
    assert final_state["cache_creation_input_tokens"] == 500
    # None of the 10000 blended tokens should be double-counted or dropped.
    assert final_state["input_tokens"] + final_state["cache_read_input_tokens"] + final_state["cache_creation_input_tokens"] == 10000


def test_graph_records_an_audit_trail_of_every_tool_call(conn, candidate_id):
    """The agent's reasoning otherwise exists only in the in-memory message
    list and is discarded the moment the graph returns — which is exactly when
    a user wants to check its work."""
    from app.db import repo_agent_steps

    _run_graph(conn, candidate_id)

    steps = repo_agent_steps.list_steps_for_candidate(conn, candidate_id)
    assert [s["tool_name"] for s in steps] == [
        "query_events",
        "check_suppression",
        "record_hypothesis",
        "conclude_investigation",
    ]
    # Tagged with the turn that produced them, so the UI can group the trace by
    # model call and show the loop structure rather than a flat list.
    assert [s["iteration"] for s in steps] == [1, 2, 3, 4]
    assert steps[0]["tool_args"] == {"node": NODE}
    assert all(s["error"] is None for s in steps)


def _refute(summary="Both events were a planned halt; HA takeover is currently possible."):
    return [_conclude("refuted", summary=summary, call_id="c1")]


def test_refuted_candidate_is_recorded_as_a_no_issue_result(conn, candidate_id):
    """Ruling a candidate out is a real Stage 2 outcome and is recorded as one,
    so every investigation result is visible in the same place instead of a
    refutation leaving nothing behind but a tool trace."""
    _run_graph(conn, candidate_id, responses=_refute())

    [record] = repo_findings.list_findings(conn)
    assert record.status == "no_issue"
    assert record.title == "No issue found"
    assert "planned halt" in record.description
    assert record.candidate_id == candidate_id


def test_a_refutation_never_blocks_a_later_real_finding(conn, candidate_id):
    """The escalation case, and the reason a refutation must not be an *open*
    finding: a pattern ruled out today can become a genuine problem next week.
    persist_hypothesis and Stage 1 both skip anything matching an open finding
    signature, so if a refutation counted as one, that later finding could
    never be created."""
    _run_graph(conn, candidate_id, responses=_refute())
    assert repo_findings.list_findings(conn)[0].status == "no_issue"

    # Same candidate signature, now genuinely confirmed.
    _run_graph(conn, candidate_id)

    statuses = sorted(f.status for f in repo_findings.list_findings(conn))
    assert statuses == ["no_issue", "open"]


def _hypothesis_call(call_id="c3", **overrides):
    """The scripted 'ready' hypothesis, with fields overridable per test."""
    args = {
        "category": CATEGORY,
        "severity": "high",
        "title": "Disk predictive failure on node1",
        "description": "Escalating SMART warnings culminated in a predictive failure event.",
        "node": NODE,
        "event_ids": [1],
        "event_names": EVENT_NAMES,
        "confidence": 0.9,
        "status": "ready",
        "recommendation": "Replace the flagged disk.",
    }
    args.update(overrides)
    return _tool_call("record_hypothesis", args, call_id)


def _one_hypothesis_then_conclude(**overrides):
    return [
        AIMessage(content="", tool_calls=[_hypothesis_call(**overrides)]),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "conclude_investigation",
                    {"summary": "Concluded after reviewing the evidence.", "outcome": "confirmed"},
                    "c4",
                )
            ],
        ),
    ]


def test_a_hallucinated_evidence_id_does_not_destroy_the_investigation(conn, candidate_id):
    """finding_evidence.event_id is a foreign key and foreign_keys is ON, so a
    single made-up event id used to raise IntegrityError from inside
    persist_findings — after the findings row had been written.

    That cost three things at once, and the third is the worst: the whole Stage
    2 spend bought nothing; the half-written findings row was committed anyway
    (runner catches inside its `with session()`, so nothing rolls back); and
    because that orphan had status='open', find_open_by_signature then blocked
    the REAL version of this finding from ever being created. Same class of
    invariant as test_a_refutation_never_blocks_a_later_real_finding."""
    final_state = _run_graph(
        conn,
        candidate_id,
        responses=_one_hypothesis_then_conclude(event_ids=[1, 999999]),
    )

    assert len(final_state["finalized_findings"]) == 1
    [finding] = repo_findings.list_findings(conn)
    assert finding.status == "open"
    # The real citation survives; only the invented one is dropped.
    evidence = repo_findings.get_finding_evidence(conn, finding.id)
    assert [e["id"] for e in evidence] == [1]


def test_a_flood_of_evidence_ids_does_not_destroy_the_investigation(conn, candidate_id):
    """The volume half of the hallucinated-evidence problem above.

    insert_evidence builds one `?` per cited id, so an unbounded list blows
    SQLite's variable limit (999 on older builds, 32766 since 3.32) — and it
    raises from inside insert_finding, i.e. AFTER the findings row is written,
    leaving exactly the committed `status='open'` orphan that then blocks the
    real finding. The list is capped instead, at the number of events a single
    tool call can even return, so no honest citation is affected."""
    real_ids = [
        conn.execute(
            """
            INSERT INTO events (file_id, raw_line, event_time, node, event_name, severity, message, sequence_num, parse_confidence, created_at)
            VALUES (1, 'raw', '2026-08-10T01:10:00-04:00', ?, ?, 'alert', 'Disk predictively failed.', ?, 'high', '2026-01-01T00:00:00+00:00')
            """,
            (NODE, "disk.predictiveFailure", i),
        ).lastrowid
        for i in range(2, MAX_EVIDENCE_IDS + 52)
    ]
    conn.commit()
    # Comfortably past every SQLite variable limit, so this would raise without
    # the cap on any build.
    cited = real_ids + list(range(10 ** 6, 10 ** 6 + 40000))

    final_state = _run_graph(
        conn, candidate_id, responses=_one_hypothesis_then_conclude(event_ids=cited)
    )

    assert len(final_state["finalized_findings"]) == 1
    [finding] = repo_findings.list_findings(conn)
    assert finding.status == "open"
    evidence = repo_findings.get_finding_evidence(conn, finding.id)
    assert [e["id"] for e in evidence] == real_ids[:MAX_EVIDENCE_IDS]


def test_a_hypothesis_with_an_invalid_category_is_not_recorded(conn, candidate_id):
    """A signature is computed FROM the category, so an off-enum one produces a
    signature Stage 1 can never reproduce: the finding could never be
    suppressed, never match an open finding, and dismissing it would suppress
    nothing. The model is told to try again rather than the run being killed."""
    final_state = _run_graph(
        conn,
        candidate_id,
        responses=_one_hypothesis_then_conclude(category="capacity_risk"),
    )

    assert final_state["hypotheses"] == []
    # Nothing was created from the invalid hypothesis — and nothing at all, in
    # fact: the script goes on to conclude outcome='confirmed', which is not a
    # refutation (see test_a_confirmed_conclusion_is_never_filed_as_no_issue),
    # and the rejected hypothesis left no working theory to keep as a partial.
    findings = repo_findings.list_findings(conn)
    assert findings == []
    corrective = [
        m for m in final_state["messages"] if "Not recorded" in str(getattr(m, "content", ""))
    ]
    assert corrective, "the model should be told why, so it can retry within the same budget"


def test_a_hypothesis_status_is_normalized_before_it_decides_anything(conn, candidate_id):
    """graph.persist_findings compares status to 'ready' exactly, so an
    unnormalized 'Ready' would quietly demote a finalized hypothesis to a
    partial — a confirmed finding reported as an unfinished one."""
    final_state = _run_graph(
        conn,
        candidate_id,
        responses=_one_hypothesis_then_conclude(status="  Ready  "),
    )

    assert [h["status"] for h in final_state["hypotheses"]] == ["ready"]
    assert [f.status for f in repo_findings.list_findings(conn)] == ["open"]


def test_refutation_is_not_written_when_the_budget_ran_out(conn, candidate_id):
    """Hitting the iteration cap is not a refutation — it means the budget ran
    out before the agent reached a view. Recording that as "no issue found"
    would state a conclusion nobody drew."""
    responses = [
        AIMessage(content="", tool_calls=[_tool_call("query_events", {"node": NODE}, f"c{i}")])
        for i in range(6)
    ]
    with patch("app.agent.graph.ChatAnthropic", FakeChatAnthropic):
        from app.agent.graph import build_graph

        graph = build_graph(conn)
        graph.invoke(
            {
                "messages": [],
                "analysis_run_id": 1,
                "candidate_id": candidate_id,
                "iteration": 0,
                "max_iterations": 2,
                "hypotheses": [],
                "finalized_findings": [],
                "findings_suppressed": 0,
                "ready_to_conclude": False,
                "conclusion_summary": "",
                "conclusion_outcome": "",
                "input_tokens": 0,
                "output_tokens": 0,
            },
            config={"recursion_limit": 50},
        )

    assert repo_findings.list_findings(conn) == []


def _working_hypothesis(description, call_id, severity="high", title="Disk predictive failure on node1"):
    return _tool_call(
        "record_hypothesis",
        {
            "category": CATEGORY,
            "severity": severity,
            "title": title,
            "description": description,
            "node": NODE,
            "event_ids": [1],
            "event_names": EVENT_NAMES,
            "confidence": 0.6,
            "status": "investigating",
            "recommendation": "Check the disk.",
        },
        call_id,
    )


def test_cost_cap_keeps_the_working_hypothesis_as_a_partial_finding(conn, candidate_id):
    """The point of the whole partial path: an investigation the cost cap cut
    short was billed in full, so whatever it had worked out by then has to
    survive. Before this, a hypothesis still at status='investigating' was
    dropped and the user saw nothing but a tool trace and a bill."""
    usage = _usage_costing(1.01)  # over the cap in a single turn
    responses = [
        AIMessage(content="", tool_calls=[_working_hypothesis("SMART warnings escalating on node1.", "c1")], usage_metadata=usage),
        AIMessage(content="", tool_calls=[_tool_call("query_events", {"node": NODE}, "c2")], usage_metadata=usage),
    ]

    final_state = _run_graph(conn, candidate_id, responses=responses)

    assert final_state["stop_reason"] == "cost_cap"
    assert final_state["ready_to_conclude"] is False
    [record] = repo_findings.list_findings(conn)
    assert record.status == "partial"
    assert record.severity == "high"  # the agent's own severity, not downgraded
    assert record.candidate_id == candidate_id
    assert "SMART warnings escalating" in record.description
    assert "cost cap" in record.description  # why it is unconfirmed, stated on the record
    # The evidence it had gathered comes with it — a lead with no events behind
    # it is not a result anyone can follow up.
    assert [e["id"] for e in repo_findings.get_finding_evidence(conn, record.id)] == [1]


def test_partial_finding_is_not_an_open_finding(conn, candidate_id):
    """Same escalation argument as a refutation: an investigation that ran out
    of money must not make the confirmed version of the same pattern
    uncreatable later. persist_hypothesis and Stage 1's pre-filter both skip
    anything matching an OPEN finding signature."""
    usage = _usage_costing(1.01)  # over the cap in a single turn
    _run_graph(
        conn,
        candidate_id,
        responses=[
            AIMessage(content="", tool_calls=[_working_hypothesis("Partial view.", "c1")], usage_metadata=usage),
            AIMessage(content="", tool_calls=[_tool_call("query_events", {"node": NODE}, "c2")], usage_metadata=usage),
        ],
    )
    assert repo_findings.list_findings(conn)[0].status == "partial"

    # The same pattern, later investigated to a proper conclusion.
    _run_graph(conn, candidate_id)

    assert sorted(f.status for f in repo_findings.list_findings(conn)) == ["open", "partial"]


def test_partial_finding_collapses_revisions_of_one_hypothesis(conn, candidate_id):
    """record_hypothesis is 'record or update', and `hypotheses` is append-only,
    so an agent refining one theory over three turns leaves three entries. That
    is one lead, not three findings — and the user should get the latest
    version of it."""
    usage = _usage_costing(0.4)  # the third turn crosses the cap
    responses = [
        AIMessage(content="", tool_calls=[_working_hypothesis("First pass.", "c1")], usage_metadata=usage),
        AIMessage(content="", tool_calls=[_working_hypothesis("Refined with more evidence.", "c2")], usage_metadata=usage),
        AIMessage(
            content="",
            tool_calls=[_working_hypothesis("A genuinely different theory.", "c3", title="Aggregate pressure")],
            usage_metadata=usage,
        ),
        AIMessage(content="", tool_calls=[_tool_call("query_events", {"node": NODE}, "c4")], usage_metadata=usage),
    ]

    _run_graph(conn, candidate_id, responses=responses)

    findings = repo_findings.list_findings(conn, status="partial")
    # Same (category, node, event_names) for all three, so they collapse to the
    # most recent one rather than piling up per turn.
    assert len(findings) == 1
    assert "A genuinely different theory." in findings[0].description


def test_turn_limit_also_keeps_the_working_hypothesis(conn, candidate_id):
    """The iteration cap is the same situation as the cost cap — budget
    exhausted mid-investigation — so it keeps the partial too, and says so."""
    responses = [
        AIMessage(content="", tool_calls=[_working_hypothesis("Half-checked theory.", "c1")]),
        AIMessage(content="", tool_calls=[_tool_call("query_events", {"node": NODE}, "c2")]),
        AIMessage(content="", tool_calls=[_tool_call("query_events", {"node": NODE}, "c3")]),
    ]

    final_state = _run_graph(conn, candidate_id, responses=responses, max_iterations=2)

    assert final_state["stop_reason"] == "iteration_cap"
    [record] = repo_findings.list_findings(conn)
    assert record.status == "partial"
    assert "turn limit" in record.description


def test_a_concluded_investigation_never_leaves_a_partial(conn, candidate_id):
    """A partial is a budget artifact. An agent that reached a conclusion said
    everything it had to say — keeping its superseded working notes alongside
    the real result would double-report one investigation."""
    responses = [
        AIMessage(content="", tool_calls=[_working_hypothesis("Working theory, later finalized.", "c1")]),
        _scripted_responses()[2],  # the same hypothesis at status='ready'
        _scripted_responses()[3],  # conclude_investigation
    ]

    final_state = _run_graph(conn, candidate_id, responses=responses)

    assert final_state["stop_reason"] == "concluded"
    assert [f.status for f in repo_findings.list_findings(conn)] == ["open"]


def test_a_partial_never_re_raises_a_dismissed_pattern(conn, candidate_id):
    """Running out of budget is not a licence to re-surface something a human
    already dismissed — suppression applies to a partial exactly as it does to
    a finalized hypothesis."""
    dismissed_id = _seed_dismissal(conn)
    usage = _usage_costing(1.01)  # over the cap in a single turn

    final_state = _run_graph(
        conn,
        candidate_id,
        responses=[
            AIMessage(content="", tool_calls=[_working_hypothesis("Suppressed pattern.", "c1")], usage_metadata=usage),
            AIMessage(content="", tool_calls=[_tool_call("query_events", {"node": NODE}, "c2")], usage_metadata=usage),
        ],
    )

    assert final_state["findings_suppressed"] == 1
    assert _findings_besides(conn, dismissed_id) == []


def test_suppressed_hypothesis_is_not_recorded_as_a_refutation(conn, candidate_id):
    """The agent DID find something; it was just already known. Recording that
    as "no issue found" would misreport the outcome."""
    dismissed_id = _seed_dismissal(conn)

    final_state = _run_graph(conn, candidate_id)

    assert final_state["findings_suppressed"] == 1
    assert _findings_besides(conn, dismissed_id) == []


def test_a_confirmed_conclusion_is_never_filed_as_no_issue(conn, candidate_id):
    """The refutation gate keys on the agent's stated OUTCOME, not merely on
    "it concluded and nothing was finalized".

    An agent that records a working hypothesis and then concludes
    outcome='confirmed' without re-recording it at status='ready' has not ruled
    anything out. Filing that as "No issue found" — which is what happened
    before this gate — states the exact opposite of what it concluded, at
    severity=info, on a record routes_findings won't even let the user dismiss.
    The theory is kept as an unconfirmed partial instead, since it was never put
    through the finalize step."""
    responses = [
        AIMessage(content="", tool_calls=[_working_hypothesis("SMART warnings escalating on node1.", "c1")]),
        _conclude("confirmed"),
    ]

    final_state = _run_graph(conn, candidate_id, responses=responses)

    assert final_state["stop_reason"] == "concluded"
    [record] = repo_findings.list_findings(conn)
    assert record.status == "partial"
    assert record.status != "no_issue"
    assert record.title == "Disk predictive failure on node1"  # the agent's, not "No issue found"
    assert "SMART warnings escalating" in record.description
    # The note says which of the two ways to get a partial this was, so it isn't
    # misread as a budget failure.
    assert "never finalized" in record.description


def test_a_confirmed_conclusion_with_nothing_recorded_leaves_nothing(conn, candidate_id):
    """The same gate, with no working hypothesis to fall back on. Better to
    leave the tool trace and no finding than to invent a conclusion: the agent
    said 'confirmed', so "no issue found" is wrong, and it never stated what it
    confirmed, so there is nothing to record as a lead either."""
    final_state = _run_graph(conn, candidate_id, responses=[_conclude("confirmed", call_id="c1")])

    assert final_state["stop_reason"] == "concluded"
    assert repo_findings.list_findings(conn) == []


def test_a_refuted_conclusion_still_wins_over_a_working_hypothesis(conn, candidate_id):
    """The other direction, pinned so the outcome gate doesn't quietly turn
    every unfinalized theory into a partial: an agent that recorded a theory and
    then explicitly ruled the candidate out has superseded its own working
    notes. Keeping both would report one investigation as two results, and would
    re-raise as a lead the very thing the agent just refuted."""
    responses = [
        AIMessage(content="", tool_calls=[_working_hypothesis("Maybe the disk is failing.", "c1")]),
        _conclude("refuted", summary="The SMART counters reset after a planned firmware update."),
    ]

    _run_graph(conn, candidate_id, responses=responses)

    [record] = repo_findings.list_findings(conn)
    assert record.status == "no_issue"
    assert "firmware update" in record.description
