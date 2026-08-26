import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.agent.findings import compute_pattern_signature, compute_signature
from app.agent.tools import (
    MAX_CONTEXT_WINDOW_MINUTES,
    MAX_TOOL_RESULT_EVENTS,
    InvestigationContext,
    build_tools,
)
from app.db import repo_feedback, repo_files

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "app" / "db" / "schema.sql"


def _events_payload(result: str):
    """Tool results carry a human-readable truncation notice after the JSON
    when a server-side ceiling was hit; strip it before parsing."""
    return json.loads(result.split("\n(truncated")[0])


ANCHOR = datetime.fromisoformat("2026-08-10T01:00:00-04:00")


def _insert_burst(conn, count, node="node1", start_minutes=0, event_name="disk.smart.error"):
    """A run of events one minute apart, as an event storm would arrive.
    Same UTC offset as the fixture's seeded events — get_events_near compares
    timestamps as strings, so mixed offsets would compare nonsensically."""
    rows = [
        (
            1,
            "raw",
            (ANCHOR + timedelta(minutes=start_minutes + i)).isoformat(),
            node,
            event_name,
            "warning",
            f"Burst event {i}.",
            1000 + start_minutes + i,
            "high",
            "2026-01-01T00:00:00+00:00",
        )
        for i in range(count)
    ]
    conn.executemany(
        """
        INSERT INTO events (file_id, raw_line, event_time, node, event_name, severity, message,
                            sequence_num, parse_confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text())
    file_record = repo_files.insert_file(connection, "seed.log", "seed://1", "hash1", 0)
    connection.execute(
        """
        INSERT INTO events (file_id, raw_line, event_time, node, event_name, severity, message, sequence_num, parse_confidence, created_at)
        VALUES (?, 'raw', ?, ?, ?, ?, ?, ?, 'high', '2026-01-01T00:00:00+00:00')
        """,
        (
            file_record.id,
            "2026-08-10T01:00:00-04:00",
            "node1",
            "disk.smart.error",
            "warning",
            "Disk 0a.00.3 SMART warning.",
            1,
        ),
    )
    connection.execute(
        """
        INSERT INTO events (file_id, raw_line, event_time, node, event_name, severity, message, sequence_num, parse_confidence, created_at)
        VALUES (?, 'raw', ?, ?, ?, ?, ?, ?, 'high', '2026-01-01T00:00:00+00:00')
        """,
        (
            file_record.id,
            "2026-08-10T01:10:00-04:00",
            "node1",
            "disk.predictiveFailure",
            "alert",
            "Disk 0a.00.3 predictively failed.",
            2,
        ),
    )
    connection.commit()
    return connection


@pytest.fixture
def tools(conn):
    return {t.name: t for t in build_tools(conn)}


def test_query_events_filters_by_node(tools):
    result = tools["query_events"].invoke({"node": "node1"})
    payload = json.loads(result)
    assert len(payload) == 2
    assert {e["event_name"] for e in payload} == {"disk.smart.error", "disk.predictiveFailure"}


def test_query_events_no_match_returns_message(tools):
    result = tools["query_events"].invoke({"node": "does-not-exist"})
    assert result == "No matching events."


def test_search_events_by_name_pattern(tools):
    result = tools["search_events_by_name_pattern"].invoke({"pattern": "predictiveFailure"})
    payload = json.loads(result)
    assert len(payload) == 1
    assert payload[0]["event_name"] == "disk.predictiveFailure"


def test_get_event_context_returns_nearby_events_on_same_node(tools):
    result = tools["get_event_context"].invoke({"event_id": 2, "window_minutes": 30})
    payload = json.loads(result)
    assert len(payload) == 1
    assert payload[0]["event_id"] == 1


def test_query_events_limit_is_clamped_regardless_of_what_the_model_asks_for(tools, conn):
    """The Stage 2 cost cap is only evaluated after a model call, so any tool
    result is guaranteed to reach the model at least once at full price and
    then ride along in the conversation for every remaining turn. A `limit`
    taken straight from the model was therefore a hole straight through
    STAGE2_COST_CAP_USD."""
    _insert_burst(conn, 300)

    result = tools["query_events"].invoke({"node": "node1", "limit": 100000})

    assert len(_events_payload(result)) == MAX_TOOL_RESULT_EVENTS
    assert "truncated" in result


def test_search_events_by_name_pattern_limit_is_clamped(tools, conn):
    _insert_burst(conn, 300)

    result = tools["search_events_by_name_pattern"].invoke({"pattern": "disk", "limit": 99999})

    assert len(_events_payload(result)) == MAX_TOOL_RESULT_EVENTS


def test_query_events_rejects_a_nonsense_limit(tools, conn):
    _insert_burst(conn, 10)

    result = tools["query_events"].invoke({"node": "node1", "limit": 0})

    assert len(_events_payload(result)) > 0  # falls back to the default, not an empty result


def test_get_event_context_caps_both_its_window_and_its_row_count(tools, conn):
    """get_event_context passed no limit at all, so a single call with a wide
    window could return a busy node's entire day — the worst of the cap
    bypasses, since the model chooses window_minutes itself."""
    # 300 events at one-minute spacing starting 1 minute after the anchor,
    # i.e. spilling well past the 240-minute ceiling.
    _insert_burst(conn, 300, start_minutes=11)

    result = tools["get_event_context"].invoke({"event_id": 1, "window_minutes": 100000})
    payload = _events_payload(result)

    assert len(payload) == MAX_TOOL_RESULT_EVENTS
    # Nothing beyond the clamped window, even though it was asked for.
    for event in payload:
        delta = abs(datetime.fromisoformat(event["time"]) - ANCHOR)
        assert delta <= timedelta(minutes=MAX_CONTEXT_WINDOW_MINUTES)


def test_event_results_are_compact_json(tools):
    """Pretty-printing the payload that dominates Stage 2's context buys the
    model nothing and is resent on every subsequent turn."""
    result = tools["query_events"].invoke({"node": "node1"})

    assert "\n  " not in result
    assert '{"event_id"' in result


def _seed_second_cluster(conn):
    """The same node name on a different cluster — not contrived: ONTAP names
    nodes <cluster>-01/-02 and the simulator defaults to "cluster1", which is
    exactly why compaction keys adjacency on (cluster, node)."""
    other = repo_files.insert_file(conn, "other.log", "seed://2", "hash2", 0, cluster="cluster-b")
    conn.executemany(
        """
        INSERT INTO events (file_id, cluster, raw_line, event_time, node, event_name, severity,
                            message, sequence_num, parse_confidence, created_at)
        VALUES (?, 'cluster-b', 'raw', ?, 'node1', ?, ?, ?, ?, 'high', '2026-01-01T00:00:00+00:00')
        """,
        [
            (other.id, "2026-08-10T01:01:00-04:00", "disk.smart.error", "warning", "Other cluster.", 1),
            (other.id, "2026-08-10T01:02:00-04:00", "disk.smart.error", "warning", "Other cluster.", 2),
        ],
    )
    conn.commit()


@pytest.fixture
def scoped_tools(conn):
    """Tools bound to an investigation of the unspecified (uploaded-logs)
    cluster, which is what the seeded events belong to."""
    _seed_second_cluster(conn)
    trace = InvestigationContext(candidate_id=None, cluster=None, scoped=True)
    return {t.name: t for t in build_tools(conn, trace=trace)}


def test_query_events_does_not_reach_into_another_cluster(scoped_tools):
    """Node names are not unique across clusters, so an unscoped query hands
    the agent another cluster's events under this cluster's node name and lets
    it correlate a pattern that never happened. Stage 1 is scoped for exactly
    this reason; Stage 2 must be too."""
    payload = _events_payload(scoped_tools["query_events"].invoke({"node": "node1"}))

    assert len(payload) == 2
    assert all("Other cluster." != e["message"] for e in payload)


def test_get_event_context_does_not_reach_into_another_cluster(scoped_tools):
    """The worst of the three: "what else was happening on this node" is only
    a meaningful question within one cluster."""
    payload = _events_payload(scoped_tools["get_event_context"].invoke({"event_id": 2, "window_minutes": 30}))

    assert [e["event_id"] for e in payload] == [1]


def test_rate_baseline_counts_only_this_cluster(scoped_tools):
    """`elsewhere_in_cluster` naming another cluster's nodes is a false
    statement, not a loose one — and a corpus-wide denominator spanning every
    cluster makes mean_per_day arithmetically wrong as well."""
    baseline = json.loads(
        scoped_tools["get_event_rate_baseline"].invoke({"event_name": "disk.smart.error", "node": "node1"})
    )

    assert baseline["total_occurrences"] == 1  # not 3
    assert baseline["elsewhere_in_cluster"]["total_occurrences"] == 0
    assert baseline["elsewhere_in_cluster"]["distinct_other_nodes"] == 0


def test_an_unscoped_legacy_investigation_still_sees_everything(conn):
    """A run created before scoping existed has scope_cluster NULL, which is
    indistinguishable from the unspecified pseudo-cluster. Those runs analyzed
    every cluster, so their investigations must not silently narrow to the
    uploaded-logs pool."""
    _seed_second_cluster(conn)
    trace = InvestigationContext(candidate_id=None, cluster=None, scoped=False)
    tools = {t.name: t for t in build_tools(conn, trace=trace)}

    payload = _events_payload(tools["query_events"].invoke({"node": "node1"}))

    assert len(payload) == 4


def test_check_suppression_before_and_after_feedback(tools, conn):
    category = "predictive_failure"
    event_names = ["disk.predictiveFailure"]
    node = "node1"

    before = tools["check_suppression"].invoke({"category": category, "event_names": event_names, "node": node})
    assert before == "not_suppressed"

    signature = compute_signature(category, event_names, node)
    pattern_signature = compute_pattern_signature(category, event_names)
    repo_feedback.insert_feedback(conn, finding_id=1, signature=signature, pattern_signature=pattern_signature)

    after = tools["check_suppression"].invoke({"category": category, "event_names": event_names, "node": node})
    assert after == "suppressed"


def test_record_hypothesis_appends_to_state(tools):
    record_hypothesis = tools["record_hypothesis"]
    result = record_hypothesis.func(
        category="predictive_failure",
        severity="high",
        title="Disk 0a.00.3 predictive failure",
        description="Escalating SMART warnings culminated in a predictive failure.",
        node="node1",
        event_ids=[1, 2],
        event_names=["disk.smart.error", "disk.predictiveFailure"],
        confidence=0.85,
        status="ready",
        recommendation="Replace disk 0a.00.3.",
        tool_call_id="tc1",
    )
    hypotheses = result.update["hypotheses"]
    assert len(hypotheses) == 1
    assert hypotheses[0]["status"] == "ready"
    assert hypotheses[0]["evidence_event_ids"] == [1, 2]
    assert result.update["messages"][0].tool_call_id == "tc1"


def test_conclude_investigation_records_why(tools):
    """A refuted candidate is shown to the user with this summary as the
    explanation, so the reasoning has to be captured rather than left implicit
    in the tool trace."""
    conclude = tools["conclude_investigation"]
    result = conclude.func(summary="Planned halt, HA is healthy.", outcome="Refuted", tool_call_id="tc2")
    assert result.update["ready_to_conclude"] is True
    assert result.update["conclusion_summary"] == "Planned halt, HA is healthy."
    assert result.update["conclusion_outcome"] == "refuted"


# --- Grounding tools: catalog definitions, baselines, live cluster state -----


@pytest.fixture
def catalog_conn(conn):
    """The `conn` fixture builds a bare schema, so the catalog table is empty.
    Seed just the two events the fixture ingests."""
    conn.executemany(
        """
        INSERT INTO ems_catalog (name, severity, description, corrective_action, snmp_trap_type, deprecated)
        VALUES (?, ?, ?, ?, ?, 0)
        """,
        [
            (
                "disk.predictiveFailure",
                "alert",
                "This message occurs when a disk reports a predictive failure.",
                "Replace the disk at the earliest opportunity.",
                "standard",
            ),
            (
                "disk.smart.error",
                "warning",
                "This message occurs when a disk reports a SMART error.",
                "Monitor the disk.",
                "standard",
            ),
        ],
    )
    conn.commit()
    return conn


@pytest.fixture
def catalog_tools(catalog_conn):
    return {t.name: t for t in build_tools(catalog_conn)}


def test_lookup_event_definition_returns_corrective_action(catalog_tools):
    """The corrective action is the whole point: it's what grounds a finding's
    recommendation in NetApp's documentation rather than the model's priors."""
    result = catalog_tools["lookup_event_definition"].invoke(
        {"event_names": ["disk.predictiveFailure"]}
    )
    payload = json.loads(result)
    assert payload[0]["corrective_action"] == "Replace the disk at the earliest opportunity."


def test_lookup_event_definition_by_pattern(catalog_tools):
    payload = json.loads(catalog_tools["lookup_event_definition"].invoke({"name_pattern": "smart"}))
    assert [r["name"] for r in payload] == ["disk.smart.error"]


def test_lookup_event_definition_miss_explains_itself(catalog_tools):
    """A bare empty result would read as 'this event doesn't exist'. The model
    needs to know the catalog simply doesn't cover it and to fall back to the
    message text."""
    result = catalog_tools["lookup_event_definition"].invoke({"event_names": ["nope.not.real"]})
    assert "No catalog entry" in result
    assert "message text" in result


def test_event_rate_baseline_counts_occurrences(tools):
    payload = json.loads(
        tools["get_event_rate_baseline"].invoke({"event_name": "disk.smart.error"})
    )
    assert payload["total_occurrences"] == 1
    assert payload["busiest_day"]["count"] == 1


def test_event_rate_baseline_reports_other_nodes(conn, tools):
    """'Only this node sees it' vs. 'the whole cluster sees it' is the
    difference between an isolated fault and normal chatter."""
    _insert_burst(conn, 3, node="node2", start_minutes=100, event_name="disk.smart.error")
    payload = json.loads(
        tools["get_event_rate_baseline"].invoke({"event_name": "disk.smart.error", "node": "node1"})
    )
    assert payload["total_occurrences"] == 1
    assert payload["elsewhere_in_cluster"]["total_occurrences"] == 3
    assert payload["elsewhere_in_cluster"]["distinct_other_nodes"] == 1


def test_cluster_state_without_credentials_is_not_an_error(tools, monkeypatch):
    """With no credentials the agent must be told there's no live evidence
    either way — not handed an error it might read as a fault."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ontap_user", None)
    payload = json.loads(tools["get_cluster_state"].invoke({"area": "disks"}))
    assert payload["available"] is False
    assert "ONTAP_USER" in payload["reason"]


def test_cluster_state_queries_the_investigations_own_cluster(conn, monkeypatch):
    """The host comes from the run's cluster scope, never from configuration.
    A configured default would be used for EVERY investigation, so with two
    clusters registered, investigating a candidate from cluster B would query
    cluster A and report its disk health as corroborating evidence for B."""
    from app.agent import tools as tools_module
    from app.agent.tools import InvestigationContext
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ontap_user", "admin")
    monkeypatch.setattr(app_settings, "ontap_password", "secret")
    seen = {}

    def fake_disk_health(host, node=None):
        seen["host"] = host
        return {"available": True, "disk_count": 0}

    monkeypatch.setattr(tools_module.cluster_state, "get_disk_health", fake_disk_health)

    context = InvestigationContext(candidate_id=None, cluster="prod-b.example.com")
    scoped = {t.name: t for t in build_tools(conn, trace=context)}
    scoped["get_cluster_state"].invoke({"area": "disks"})

    assert seen["host"] == "prod-b.example.com"


def test_cluster_state_reports_no_host_for_uploaded_logs(conn, monkeypatch):
    """Events from a dropped log file have no cluster behind them, so there is
    nothing to query — and nothing to silently fall back to."""
    from app.agent.tools import InvestigationContext
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ontap_user", "admin")
    monkeypatch.setattr(app_settings, "ontap_password", "secret")

    context = InvestigationContext(candidate_id=None, cluster=None)
    scoped = {t.name: t for t in build_tools(conn, trace=context)}
    payload = json.loads(scoped["get_cluster_state"].invoke({"area": "ha"}))

    assert payload["available"] is False
    assert "uploaded log file" in payload["reason"]


def test_cluster_state_rejects_unknown_area(tools):
    payload = json.loads(tools["get_cluster_state"].invoke({"area": "weather"}))
    assert payload["available"] is False
    assert "aggregates" in payload["reason"]


def test_cluster_state_dispatches_to_the_right_area(tools, monkeypatch):
    from app.agent import tools as tools_module
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ontap_user", "admin")
    monkeypatch.setattr(app_settings, "ontap_password", "secret")
    monkeypatch.setattr(
        tools_module.cluster_state,
        "get_node_ha_status",
        lambda host, **kwargs: {"available": True, "nodes": [{"name": "n1"}]},
    )
    payload = json.loads(tools["get_cluster_state"].invoke({"area": "HA"}))
    assert payload["nodes"][0]["name"] == "n1"
