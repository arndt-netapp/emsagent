import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.config import settings

SAMPLE_LOG = (
    "2026-08-10T00:00:12-04:00 [node1: callhome.reboot:notice]: callhome.reboot: System rebooted.\n"
    "2026-08-10T01:00:02-04:00 [node1: disk.smart.error:warning]: disk.smart.error: SMART warning.\n"
    "2026-08-10T01:15:33-04:00 [node1: disk.predictiveFailure:alert]: disk.predictiveFailure: predictively failed.\n"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "database_path", tmp_path / "test.db")
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    monkeypatch.setattr(settings, "ems_watch_dir", watch_dir)

    from app.main import app

    return TestClient(app)


# --- Stage 1 fakes (raw anthropic SDK shape, patched onto app.agent.stage1.anthropic.Anthropic) ---


class FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int, cache_creation_input_tokens=0, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class FakeTextBlock:
    def __init__(self, text: str):
        self.text = text
        self.type = "text"


class FakeStage1Message:
    def __init__(self, payload: Dict[str, Any], input_tokens=100, output_tokens=50, stop_reason="end_turn"):
        self.content = [FakeTextBlock(json.dumps(payload))]
        self.usage = FakeUsage(input_tokens, output_tokens)
        self.stop_reason = stop_reason


class FakeMessagesClient:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        return self._response


class FakeAnthropicClient:
    def __init__(self, response):
        self.messages = FakeMessagesClient(response)


def _stage1_candidate_payload(refs=(3,), category="predictive_failure", node="node1", rationale="Predictive failure flagged."):
    return {
        "candidates": [
            {
                "id": "C1",
                "rank": 1,
                "category": category,
                "node": node,
                "refs": list(refs),
                "rationale": rationale,
                "confidence": 0.8,
            }
        ]
    }


def _stage1_no_candidates_payload():
    return {"candidates": []}


def _patch_stage1(payload):
    return patch(
        "app.agent.stage1.anthropic.Anthropic",
        lambda **kwargs: FakeAnthropicClient(FakeStage1Message(payload)),
    )


# --- Stage 2 fakes (langchain_anthropic.ChatAnthropic shape, patched onto app.agent.graph.ChatAnthropic) ---


def _tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id}


def _stage2_scripted_responses() -> List[AIMessage]:
    return [
        AIMessage(content="", tool_calls=[_tool_call("query_events", {"node": "node1"}, "c1")]),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "check_suppression",
                    {"category": "predictive_failure", "event_names": ["disk.predictiveFailure"], "node": "node1"},
                    "c2",
                )
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "record_hypothesis",
                    {
                        "category": "predictive_failure",
                        "severity": "high",
                        "title": "Disk predictive failure on node1",
                        "description": "SMART warnings escalated into a predictive failure on node1.",
                        "node": "node1",
                        "event_ids": [2, 3],
                        "event_names": ["disk.predictiveFailure"],
                        "confidence": 0.9,
                        "status": "ready",
                        "recommendation": "Replace flagged disk.",
                    },
                    "c3",
                )
            ],
        ),
        AIMessage(content="", tool_calls=[_tool_call("conclude_investigation", {"summary": "Concluded after reviewing the evidence.", "outcome": "confirmed"}, "c4")]),
    ]


class FakeBoundModel:
    def __init__(self, responses):
        self._responses = responses
        self._i = 0

    def invoke(self, messages, **kwargs):
        response = self._responses[self._i]
        self._i += 1
        return response


class FakeChatAnthropic:
    def __init__(self, *args, **kwargs):
        pass

    def bind_tools(self, tools):
        return FakeBoundModel(_stage2_scripted_responses())


def _upload_and_ingest_sample(client, filename="sample.log"):
    """Upload ingests on its own — there is no separate ingest step to drive."""
    files = {"file": (filename, SAMPLE_LOG, "text/plain")}
    r = client.post("/api/files/upload", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "processed"
    assert body["event_count"] == 3
    return body


def test_file_upload_ingest_and_list(client):
    _upload_and_ingest_sample(client)

    r = client.get("/api/files")
    assert r.status_code == 200
    files = r.json()
    assert len(files) == 1
    assert files[0]["status"] == "processed"
    assert files[0]["event_count"] == 3

    r = client.get(f"/api/files/{files[0]['id']}")
    assert r.status_code == 200

    r = client.get("/api/files/9999")
    assert r.status_code == 404


def test_analysis_run_produces_candidates_investigate_dismiss_and_auto_suppress_on_rerun(client):
    _upload_and_ingest_sample(client)

    # Stage 1: broad scan produces one ranked candidate.
    with _patch_stage1(_stage1_candidate_payload()):
        r = client.post("/api/analysis/runs")
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        r = client.get(f"/api/analysis/runs/{run_id}")
        assert r.status_code == 200
        run = r.json()
        assert run["status"] == "completed"
        assert run["candidates_generated"] == 1
        assert run["candidates_auto_suppressed"] == 0
        assert run["scope"]["total"] == 3
        assert run["scope"]["files"] == ["sample.log"]
        assert run["scope"]["nodes"] == ["node1"]

    # With nothing new ingested and nothing dismissed yet, an immediate
    # re-run is refused rather than spending another round of model calls
    # to re-derive the same result.
    r = client.post("/api/analysis/runs")
    assert r.status_code == 409

    r = client.get(f"/api/analysis/runs/{run_id}/candidates")
    assert r.status_code == 200
    candidates = r.json()
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["status"] == "pending"
    assert candidate["category"] == "predictive_failure"
    assert candidate["node"] == "node1"

    # No findings exist yet — Stage 2 hasn't run for this candidate.
    assert client.get("/api/findings").json() == []

    # Stage 2: human selects this candidate for a real investigation.
    with patch("app.agent.graph.ChatAnthropic", FakeChatAnthropic):
        r = client.post(f"/api/candidates/{candidate['id']}/investigate")
        assert r.status_code == 200
        assert r.json()["status"] == "investigating"

        r = client.get(f"/api/candidates/{candidate['id']}")
        assert r.status_code == 200
        detail = r.json()
        assert detail["status"] == "investigated"
        assert len(detail["findings"]) == 1
        assert detail["investigation_input_tokens"] is not None

    r = client.get("/api/findings")
    findings = r.json()
    assert len(findings) == 1
    finding_id = findings[0]["id"]
    assert findings[0]["candidate_id"] == candidate["id"]

    r = client.get(f"/api/findings/{finding_id}")
    assert r.status_code == 200
    assert len(r.json()["evidence"]) == 2

    r = client.post(f"/api/findings/{finding_id}/dismiss", json={"reason": "known issue", "scope": "node"})
    assert r.status_code == 200
    assert r.json()["status"] == "dismissed"

    r = client.get("/api/findings", params={"status": "open"})
    assert r.json() == []
    r = client.get("/api/findings", params={"status": "dismissed"})
    assert len(r.json()) == 1

    # Re-running Stage 1 (now allowed — a dismissal was just recorded) should
    # auto-suppress the same candidate at triage time rather than persist it
    # as 'pending' and wait for a human to investigate a known non-issue.
    with _patch_stage1(_stage1_candidate_payload()):
        r = client.post("/api/analysis/runs")
        assert r.status_code == 200
        run_id_2 = r.json()["run_id"]
        run_2 = client.get(f"/api/analysis/runs/{run_id_2}").json()
        assert run_2["candidates_generated"] == 1
        assert run_2["candidates_auto_suppressed"] == 1

    candidates_2 = client.get(f"/api/analysis/runs/{run_id_2}/candidates").json()
    assert candidates_2[0]["status"] == "auto_suppressed"

    r = client.get("/api/findings")
    assert len(r.json()) == 1  # still just the one (dismissed) finding, no duplicate


def test_investigate_requires_pending_candidate(client):
    _upload_and_ingest_sample(client)
    with _patch_stage1(_stage1_candidate_payload()):
        run_id = client.post("/api/analysis/runs").json()["run_id"]
    candidate_id = client.get(f"/api/analysis/runs/{run_id}/candidates").json()[0]["id"]

    with patch("app.agent.graph.ChatAnthropic", FakeChatAnthropic):
        r = client.post(f"/api/candidates/{candidate_id}/investigate")
        assert r.status_code == 200

        r = client.post(f"/api/candidates/{candidate_id}/investigate")
        assert r.status_code == 409

    r = client.post("/api/candidates/9999/investigate")
    assert r.status_code == 404


def test_discard_candidate_no_llm_call(client):
    _upload_and_ingest_sample(client)
    with _patch_stage1(_stage1_candidate_payload()):
        run_id = client.post("/api/analysis/runs").json()["run_id"]
    candidate_id = client.get(f"/api/analysis/runs/{run_id}/candidates").json()[0]["id"]

    r = client.post(f"/api/candidates/{candidate_id}/discard", json={"reason": "known false positive"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "discarded"
    assert body["discard_reason"] == "known false positive"

    # Already discarded — a second discard is rejected, not silently accepted.
    r = client.post(f"/api/candidates/{candidate_id}/discard", json={})
    assert r.status_code == 409

    assert client.get("/api/findings").json() == []


def test_losing_a_race_to_investigate_is_a_409_not_a_500(client):
    """The status check in candidate_service is not the guard — the repo's
    conditional UPDATE is, and it raises a bare ValueError when the row moved
    in between (two clicks, two tabs). The route catches only LookupError and
    InvalidCandidateStateError, so that path used to escape as a 500 with a
    traceback while the identical non-race case returned a clean 409.

    Simulated by moving the row and then making the pre-check see the stale
    'pending' it would have seen a moment earlier."""
    _upload_and_ingest_sample(client)
    with _patch_stage1(_stage1_candidate_payload()):
        run_id = client.post("/api/analysis/runs").json()["run_id"]
    candidate = client.get(f"/api/analysis/runs/{run_id}/candidates").json()[0]

    assert client.post(f"/api/candidates/{candidate['id']}/discard", json={}).status_code == 200

    from app.db.models import Candidate

    stale = Candidate(
        id=candidate["id"],
        analysis_run_id=run_id,
        rank=1,
        category="predictive_failure",
        node="node1",
        rationale="stale read",
        confidence=0.8,
        refs=[3],
        leads=[],
        status="pending",  # what the racing request saw
        discard_reason=None,
        investigation_input_tokens=None,
        investigation_output_tokens=None,
        investigation_cache_creation_input_tokens=None,
        investigation_cache_read_input_tokens=None,
        investigation_iterations=None,
        investigation_started_at=None,
        investigation_completed_at=None,
        investigation_error=None,
        created_at="2026-01-01T00:00:00+00:00",
    )
    with patch("app.services.candidate_service.repo_candidates.get_candidate", return_value=stale):
        r = client.post(f"/api/candidates/{candidate['id']}/investigate")
        assert r.status_code == 409

        r = client.post(f"/api/candidates/{candidate['id']}/discard", json={})
        assert r.status_code == 409


def test_dismiss_nonexistent_finding_404(client):
    r = client.post("/api/findings/9999/dismiss", json={"scope": "node"})
    assert r.status_code == 404


def test_trigger_run_with_no_events_is_refused(client):
    r = client.post("/api/analysis/runs")
    assert r.status_code == 409
    assert "no ems events" in r.json()["detail"].lower()


def test_scope_endpoint_reflects_ingested_events(client):
    r = client.get("/api/analysis/scope")
    assert r.json()["total"] == 0

    _upload_and_ingest_sample(client)

    r = client.get("/api/analysis/scope")
    assert r.status_code == 200
    scope = r.json()
    assert scope["total"] == 3
    assert scope["nodes"] == ["node1"]
    assert scope["files"] == ["sample.log"]


def test_stage1_produces_no_candidates_when_nothing_flagged(client):
    _upload_and_ingest_sample(client)

    with _patch_stage1(_stage1_no_candidates_payload()):
        r = client.post("/api/analysis/runs")
        run_id = r.json()["run_id"]
        run = client.get(f"/api/analysis/runs/{run_id}").json()
        assert run["candidates_generated"] == 0

    assert client.get(f"/api/analysis/runs/{run_id}/candidates").json() == []


def test_cluster_fetch_events_ingests_immediately(client):
    records = [
        {
            "node": {"name": "clusternode1"},
            "time": "2026-08-10T00:00:12-04:00",
            "message": {"name": "callhome.reboot", "severity": "notice"},
            "log_message": "callhome.reboot: System rebooted.",
        }
    ]

    def fake_fetch(host, user, password, count, severity=None, log_message=None, verify_tls=False, page_size=100):
        assert host == "cluster.example.com"
        for r in records[:count]:
            yield r

    with patch("app.api.routes_clusters.fetch_ems_events", fake_fetch):
        r = client.post(
            "/api/clusters/fetch-events",
            json={"cluster": "cluster.example.com", "username": "admin", "password": "pw", "count": 1},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "processed"
    assert body["event_count"] == 1


def test_cluster_fetch_events_502_on_client_error(client):
    from app.ontap.client import OntapClientError

    def fake_fetch(*args, **kwargs):
        raise OntapClientError("connection refused")
        yield  # pragma: no cover - makes this a generator

    with patch("app.api.routes_clusters.fetch_ems_events", fake_fetch):
        r = client.post(
            "/api/clusters/fetch-events",
            json={"cluster": "cluster.example.com", "username": "admin", "password": "pw", "count": 1},
        )
    assert r.status_code == 502


def test_cluster_fetch_keeps_the_events_that_arrived_before_a_mid_walk_failure(client):
    """A 10,000-event fetch is dozens of sequential paginated requests, and a
    cluster closing the connection near the end used to lose every event
    already retrieved. The ones that arrived are perfectly good; only a fetch
    that got nothing at all is an error."""
    from app.ontap.client import OntapClientError

    def fake_fetch(host, user, password, count, severity=None, log_message=None, verify_tls=False, page_size=500):
        yield CLUSTER_EVENT_1
        yield CLUSTER_EVENT_2
        raise OntapClientError(
            "GET https://cluster.example.com/... failed: ConnectionError: "
            "('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))"
        )

    with patch("app.api.routes_clusters.fetch_ems_events", fake_fetch):
        r = client.post(
            "/api/clusters/fetch-events",
            json={"cluster": "cluster.example.com", "username": "admin", "password": "pw", "count": 5000},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "processed"
    assert body["event_count"] == 2
    # The shortfall is stated, not silently passed off as "the cluster only
    # had 2 events" — the difference between 2 and the 5000 asked for is the
    # whole thing the user needs to know.
    assert "stopped after 2 of the 5000" in body["error_message"]
    assert "RemoteDisconnected" in body["error_message"]


def test_cluster_fetch_that_dies_before_any_event_is_still_a_502(client, tmp_path):
    """Nothing fetched is a real failure, and it must not leave a zero-byte
    log file behind for the watch directory to pick up as a new bundle."""
    from app.config import settings
    from app.ontap.client import OntapClientError

    def fake_fetch(*args, **kwargs):
        raise OntapClientError("connection refused")
        yield  # pragma: no cover - makes this a generator

    with patch("app.api.routes_clusters.fetch_ems_events", fake_fetch):
        r = client.post(
            "/api/clusters/fetch-events",
            json={"cluster": "cluster.example.com", "username": "admin", "password": "pw", "count": 5000},
        )

    assert r.status_code == 502
    assert list(settings.ems_watch_dir.glob("ems_fetch_*.log")) == []


def _fake_cluster_fetch(records):
    def fake_fetch(host, user, password, count, severity=None, log_message=None, verify_tls=False, page_size=100):
        for r in records[:count]:
            yield r

    return fake_fetch


CLUSTER_EVENT_1 = {
    "node": {"name": "cluster1-01"},
    "time": "2026-08-10T00:00:12-04:00",
    "message": {"name": "callhome.reboot", "severity": "notice"},
    "log_message": "callhome.reboot: System rebooted.",
}
CLUSTER_EVENT_2 = {
    "node": {"name": "cluster1-01"},
    "time": "2026-08-10T00:05:00-04:00",
    "message": {"name": "disk.smart.error", "severity": "error"},
    "log_message": "disk.smart.error: SMART error on disk 0a.00.3.",
}


def _fetch(client, records, count=10):
    with patch("app.api.routes_clusters.fetch_ems_events", _fake_cluster_fetch(records)):
        return client.post(
            "/api/clusters/fetch-events",
            json={"cluster": "cluster.example.com", "username": "admin", "password": "pw", "count": count},
        )


def test_cluster_fetch_identical_refetch_is_409_not_500(client):
    """Re-fetching a cluster that produced no new events writes byte-identical
    content, whose hash collides with the UNIQUE constraint on files.file_hash.
    That surfaced as an unhandled IntegrityError and a 500."""
    assert _fetch(client, [CLUSTER_EVENT_1]).status_code == 200

    second = _fetch(client, [CLUSTER_EVENT_1])
    assert second.status_code == 409
    assert "nothing new" in second.json()["detail"]


def test_cluster_fetch_overlapping_refetch_only_ingests_new_events(client):
    """The normal case: the second pull repeats the first's events and adds
    one. Only the new event should land, or compaction would report the
    repeated one as having fired twice."""
    assert _fetch(client, [CLUSTER_EVENT_1]).status_code == 200

    second = _fetch(client, [CLUSTER_EVENT_1, CLUSTER_EVENT_2])
    assert second.status_code == 200
    body = second.json()
    assert body["event_count"] == 1
    assert body["duplicates_skipped"] == 1

    scope = client.get("/api/analysis/scope").json()
    assert scope["total"] == 2


def test_cluster_fetch_tags_events_with_their_cluster(client):
    """Without this the events land in the 'unspecified' bucket and become
    uncorrelatable with — or worse, correlated against — another cluster."""
    assert _fetch(client, [CLUSTER_EVENT_1]).status_code == 200

    options = client.get("/api/analysis/scope-options").json()
    assert {o["cluster"] for o in options} == {"cluster.example.com"}


def test_finding_detail_carries_its_investigation_trace(client):
    """The trace belongs next to the consequential action. Dismissing a finding
    happens on the Findings page, so that page must be able to show how the
    conclusion was reached — not just the Analysis page you stop visiting once
    the run is done."""
    from app.agent.findings import compute_pattern_signature, compute_signature
    from app.db import repo_agent_steps, repo_analysis_runs, repo_candidates, repo_findings
    from app.db.session import session as db_session

    with db_session() as conn:
        run = repo_analysis_runs.start_run(conn)
        repo_candidates.bulk_insert_candidates(
            conn,
            run.id,
            [
                repo_candidates.CandidateInput(
                    rank=3,
                    category="availability_risk",
                    node="node1",
                    rationale="takeover disabled",
                    confidence=0.9,
                    refs=[1],
                    leads=[],
                    status="pending",
                )
            ],
        )
        candidate_id = repo_candidates.list_candidates_for_run(conn, run.id)[0].id
        repo_agent_steps.record_step(
            conn,
            candidate_id=candidate_id,
            iteration=1,
            step_index=1,
            tool_name="lookup_event_definition",
            tool_args={"event_names": ["cf.fsm.takeoverOfPartnerDisabled"]},
            result_summary="[{...}]",
        )
        repo_candidates.start_investigation(conn, candidate_id)
        repo_candidates.complete_investigation(
            conn,
            candidate_id,
            input_tokens=2100,
            output_tokens=680,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            iterations=4,
        )
        names = ["cf.fsm.takeoverOfPartnerDisabled"]
        finding = repo_findings.insert_finding(
            conn,
            analysis_run_id=run.id,
            category="availability_risk",
            severity="high",
            title="HA takeover disabled",
            description="d",
            recommendation=None,
            node="node1",
            signature=compute_signature("availability_risk", names, "node1"),
            pattern_signature=compute_pattern_signature("availability_risk", names),
            confidence=0.9,
            evidence_event_ids=[],
            candidate_id=candidate_id,
        )

    detail = client.get(f"/api/findings/{finding.id}").json()
    assert [s["tool_name"] for s in detail["steps"]] == ["lookup_event_definition"]
    assert detail["analysis_run_id"] == run.id
    assert detail["candidate_rank"] == 3
    assert detail["investigation_iterations"] == 4
    assert detail["investigation_cost_usd"] > 0


def test_finding_without_a_candidate_has_no_trace(client):
    """Findings predating candidates (or created by any path that sets no
    candidate_id) simply have no trace — an empty list, not an error."""
    from app.agent.findings import compute_pattern_signature, compute_signature
    from app.db import repo_findings
    from app.db.session import session as db_session

    with db_session() as conn:
        names = ["disk.smart.error"]
        finding = repo_findings.insert_finding(
            conn,
            analysis_run_id=None,
            category="predictive_failure",
            severity="low",
            title="Orphan finding",
            description="d",
            recommendation=None,
            node=None,
            signature=compute_signature("predictive_failure", names, None),
            pattern_signature=compute_pattern_signature("predictive_failure", names),
            confidence=0.5,
            evidence_event_ids=[],
        )

    detail = client.get(f"/api/findings/{finding.id}").json()
    assert detail["steps"] == []
    assert detail["investigation_cost_usd"] is None


def test_no_issue_record_cannot_be_dismissed(client):
    """Dismissing writes suppression feedback. Agreeing that something is not a
    problem is a different signal from a human overriding the agent, and
    conflating them would poison the dismissal reasons Stage 1 is shown."""
    from app.agent.findings import persist_refutation
    from app.db import repo_analysis_runs, repo_candidates
    from app.db.session import session as db_session

    with db_session() as conn:
        run = repo_analysis_runs.start_run(conn)
        repo_candidates.bulk_insert_candidates(
            conn,
            run.id,
            [
                repo_candidates.CandidateInput(
                    rank=1,
                    category="availability_risk",
                    node="node1",
                    rationale="takeover disabled",
                    confidence=0.7,
                    refs=[1],
                    leads=[],
                    status="pending",
                )
            ],
        )
        candidate_id = repo_candidates.list_candidates_for_run(conn, run.id)[0].id
        record = persist_refutation(
            conn,
            analysis_run_id=run.id,
            candidate_id=candidate_id,
            category="availability_risk",
            node="node1",
            event_names=["cf.fsm.takeoverOfPartnerDisabled"],
            summary="Planned halt; HA takeover is currently possible on both nodes.",
        )

    r = client.post(f"/api/findings/{record.id}/dismiss", json={"scope": "node"})
    assert r.status_code == 409
    assert "nothing to dismiss" in r.json()["detail"]

    listed = client.get("/api/findings?status=no_issue").json()
    assert [f["id"] for f in listed] == [record.id]


def test_partial_finding_is_listable_and_dismissible(client):
    """The opposite call from a refutation, and deliberately so: a partial is a
    claimed risk nobody finished checking, so a human saying "not important" is
    a genuine override — the same signal any other dismissal carries. It is
    hidden from the default status=open list, so it has to be reachable by its
    own filter."""
    from app.agent.findings import persist_partial
    from app.db import repo_analysis_runs, repo_candidates
    from app.db.session import session as db_session

    with db_session() as conn:
        run = repo_analysis_runs.start_run(conn)
        repo_candidates.bulk_insert_candidates(
            conn,
            run.id,
            [
                repo_candidates.CandidateInput(
                    rank=1,
                    category="predictive_failure",
                    node="node1",
                    rationale="disk errors",
                    confidence=0.7,
                    refs=[1],
                    leads=[],
                    status="pending",
                )
            ],
        )
        candidate_id = repo_candidates.list_candidates_for_run(conn, run.id)[0].id
        record = persist_partial(
            conn,
            analysis_run_id=run.id,
            hypothesis={
                "category": "predictive_failure",
                "severity": "high",
                "title": "Disk errors escalating on node1",
                "description": "Three SMART errors in an hour on the same disk.",
                "node": "node1",
                "event_names": ["disk.smart.error"],
                "evidence_event_ids": [],
                "confidence": 0.6,
                "status": "investigating",
            },
            candidate_id=candidate_id,
            stop_reason="cost_cap",
            iterations=7,
        )

    assert client.get("/api/findings?status=open").json() == []
    listed = client.get("/api/findings?status=partial").json()
    assert [f["id"] for f in listed] == [record.id]

    r = client.post(f"/api/findings/{record.id}/dismiss", json={"scope": "node", "reason": "known"})
    assert r.status_code == 200
    assert r.json()["status"] == "dismissed"


def test_upload_ingests_immediately(client):
    """Upload used to leave the file `pending` for a separate ingest step. That
    step no longer exists in the UI, so upload has to be one action with one
    outcome or every upload would strand at zero events."""
    body = _upload_and_ingest_sample(client)
    assert body["status"] == "processed"
    assert body["event_count"] == 3
    assert client.get("/api/analysis/scope").json()["total"] == 3


def test_upload_recovers_the_cluster_from_a_fetch_filename(client):
    """Files produced by scripts/fetch_ems_events.py carry their cluster in
    their own name. Without recovering it they land in the unspecified bucket,
    where two clusters' events would sit together — the exact cross-cluster
    mixing that run scoping exists to prevent."""
    body = _upload_and_ingest_sample(client, filename="ems_fetch_netapp-prod-01_20260819T120000Z.log")
    assert body["cluster"] == "netapp-prod-01"

    options = client.get("/api/analysis/scope-options").json()
    assert {o["cluster"] for o in options} == {"netapp-prod-01"}


def test_upload_of_an_unconventional_filename_has_no_cluster(client):
    """A log file of unknown provenance genuinely has no cluster identity, and
    guessing one would be worse than leaving it unspecified."""
    body = _upload_and_ingest_sample(client, filename="some-random-export.log")
    assert body["cluster"] is None


AUTOSUPPORT_LOG = (
    "Tue Aug 25 00:08:22 -0700 [cluster2-n03: dense_ads_monitor: sis.auto.session.change:notice]: "
    "ADS: Number of auto sessions changed from 3 to 4\n"
    "Tue Aug 25 00:08:31 -0700 [cluster2-n03: ypbind: nis.server.not.available:error]: "
    "None of the NIS servers configured can be contacted.\n"
    "Tue Aug 25 00:12:08 -0700 [cluster2-n04: secd: secd.dns.srv.lookup.failed:error]: "
    "DNS server failed to look up service.\n"
)


def test_upload_accepts_an_autosupport_ems_log(client):
    """An EMS-LOG-FILE.txt from an autosupport bundle is the historical path
    into the app: no cluster to reach over REST, and a window a live fetch can
    no longer return."""
    r = client.post(
        "/api/files/upload",
        files={"file": ("EMS-LOG-FILE.txt", AUTOSUPPORT_LOG, "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "processed"
    assert body["detected_format"] == "autosupport_ems"
    assert body["event_count"] == 3

    scope = client.get("/api/analysis/scope").json()
    assert scope["total"] == 3
    assert set(scope["nodes"]) == {"cluster2-n03", "cluster2-n04"}


def test_upload_with_an_explicit_cluster_scopes_the_events(client):
    """An autosupport log carries no cluster in its lines OR its name, so
    without a way to state one every bundle joins the unspecified pool, where a
    second cluster's events would be correlated against the first."""
    r = client.post(
        "/api/files/upload",
        files={"file": ("EMS-LOG-FILE.txt", AUTOSUPPORT_LOG, "text/plain")},
        data={"cluster": "  cluster2  "},
    )
    assert r.status_code == 200
    assert r.json()["cluster"] == "cluster2"

    options = client.get("/api/analysis/scope-options").json()
    assert {o["cluster"] for o in options} == {"cluster2"}


def test_an_explicit_cluster_overrides_the_filename_convention(client):
    """The human uploading the file is a better authority on where it came from
    than a filename someone may have reused."""
    r = client.post(
        "/api/files/upload",
        files={"file": ("ems_fetch_netapp-prod-01_20260819T120000Z.log", SAMPLE_LOG, "text/plain")},
        data={"cluster": "the-real-cluster"},
    )
    assert r.json()["cluster"] == "the-real-cluster"


def test_a_blank_cluster_field_still_falls_back_to_the_filename(client):
    """An empty form field is an absent answer, not an answer of "", so the
    filename convention must still get its turn."""
    r = client.post(
        "/api/files/upload",
        files={"file": ("ems_fetch_netapp-prod-01_20260819T120000Z.log", SAMPLE_LOG, "text/plain")},
        data={"cluster": "   "},
    )
    assert r.json()["cluster"] == "netapp-prod-01"


def test_upload_cannot_write_outside_the_watch_directory(client, tmp_path):
    """The uploaded filename is attacker-controlled, and joined raw it escaped:
    `../../x.log` traverses, and an ABSOLUTE name escapes with no traversal at
    all because pathlib drops the left operand. Both wrote a real file and
    returned 200."""
    watch_dir = tmp_path / "watch"

    for i, supplied in enumerate(("../../pwned.log", "/tmp/pwned-absolute.log")):
        # Distinct content per upload: identical bytes are deduplicated by hash
        # and would 409 before reaching the path handling under test.
        r = client.post(
            "/api/files/upload",
            files={"file": (supplied, SAMPLE_LOG + f"# {i}\n", "text/plain")},
        )
        assert r.status_code == 200
        # The stored path, not just the displayed name: the old code recorded
        # dest.name, so the row looked innocent either way.
        stored = client.get(f"/api/files/{r.json()['id']}").json()["filename"]
        assert "/" not in stored and ".." not in stored

    written = sorted(p.name for p in watch_dir.iterdir())
    assert written == ["pwned-absolute.log", "pwned.log"]
    assert not (tmp_path / "pwned.log").exists()
    assert not Path("/tmp/pwned-absolute.log").exists()


def test_upload_rejects_a_cluster_that_is_not_a_hostname(client):
    """The cluster field is not a label: it becomes the REST host the Stage 2
    agent queries with the configured ONTAP credentials, so an arbitrary string
    here sends those credentials wherever it points."""
    r = client.post(
        "/api/files/upload",
        files={"file": ("EMS-LOG-FILE.txt", AUTOSUPPORT_LOG, "text/plain")},
        data={"cluster": "https://attacker.example.com/x"},
    )
    assert r.status_code == 400
    assert "valid cluster host" in r.json()["detail"]
    # Rejected before anything was written, so no orphan file is left behind.
    assert client.get("/api/files").json() == []


def test_cluster_fetch_rejects_a_cluster_that_is_not_a_hostname(client):
    r = client.post(
        "/api/clusters/fetch-events",
        json={"cluster": "admin@attacker.example.com", "username": "u", "password": "p"},
    )
    assert r.status_code == 422


# --- Severity floor -------------------------------------------------------
#
# Both ingestion paths default to notice-and-higher. What these pin is not the
# arithmetic (tests/test_severity.py does that) but the two things that make
# the default safe to have: the floor is RECORDED on the file it filtered, and
# it is enforced locally rather than trusted to whoever was asked to apply it.

NOISY_LOG = (
    "2026-08-10T00:00:12-04:00 [node1: callhome.reboot:notice]: callhome.reboot: System rebooted.\n"
    "2026-08-10T00:30:00-04:00 [node1: wafl.vol.snap.create:info]: snapshot created.\n"
    "2026-08-10T00:45:00-04:00 [node1: rastrace.dump.saved:debug]: trace dump stored.\n"
    "2026-08-10T01:00:02-04:00 [node1: disk.smart.error:error]: disk.smart.error: SMART warning.\n"
)


def test_upload_defaults_to_notice_and_higher_and_records_that_it_did(client):
    """The default drops rows, so the file has to say so: nothing downstream
    could otherwise tell a quiet cluster from a filtered one, and
    get_event_rate_baseline averages across every file a cluster ever
    contributed."""
    r = client.post("/api/files/upload", files={"file": ("noisy.log", NOISY_LOG, "text/plain")})
    assert r.status_code == 200
    body = r.json()
    assert body["event_count"] == 2
    assert body["severity_filter"] == "notice"
    assert body["severity_skipped"] == 2

    scope = client.get("/api/analysis/scope").json()
    assert set(scope["severity_counts"]) == {"notice", "error"}


def test_upload_with_all_severities_keeps_the_noise_and_records_no_floor(client):
    r = client.post(
        "/api/files/upload",
        files={"file": ("noisy.log", NOISY_LOG, "text/plain")},
        data={"min_severity": "all"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["event_count"] == 4
    assert body["severity_filter"] is None
    assert body["severity_skipped"] == 0


def test_an_uploaded_autosupport_bundles_info_lines_are_dropped_by_the_default(client):
    """A bundle writes `info`, not `informational`. The alias is what makes the
    default do anything at all on the historical path."""
    log = AUTOSUPPORT_LOG + (
        "Tue Aug 25 00:14:00 -0700 [cluster2-n03: wafl_exempt14: wafl.vol.snap_create.done:info]: "
        "params: {'vol': 'arndt1'}\n"
    )
    r = client.post("/api/files/upload", files={"file": ("EMS-LOG-FILE.txt", log, "text/plain")})
    assert r.status_code == 200
    assert r.json()["event_count"] == 3
    assert r.json()["severity_skipped"] == 1


def test_a_line_the_parser_could_not_read_survives_the_severity_floor(client):
    """An unparsed line has no severity, and dropping it would discard exactly
    what a human most needs to see — while reporting it as removed noise."""
    r = client.post(
        "/api/files/upload",
        files={"file": ("mixed.log", NOISY_LOG + "this line is not EMS at all\n", "text/plain")},
    )
    assert r.status_code == 200
    # Two of the four EMS lines clear the floor, and the unreadable line makes
    # three: it has no severity to rank, so it is kept.
    assert r.json()["event_count"] == 3
    assert r.json()["severity_skipped"] == 2

    # severity_counts only counts rows that HAVE a severity, so a total of 3
    # over 2 counted severities is the surviving unparsed row.
    scope = client.get("/api/analysis/scope").json()
    assert scope["total"] == 3
    assert sum(scope["severity_counts"].values()) == 2


def test_upload_rejects_a_severity_that_cannot_be_ranked(client):
    """A typo'd floor that fell through to "no filter" would keep every
    severity while the row recorded the filter the user asked for."""
    r = client.post(
        "/api/files/upload",
        files={"file": ("noisy.log", NOISY_LOG, "text/plain")},
        data={"min_severity": "noticee"},
    )
    assert r.status_code == 400
    assert "unknown severity" in r.json()["detail"]
    assert client.get("/api/files").json() == []


def test_cluster_fetch_asks_the_cluster_for_the_floor_as_an_or_list(client):
    """`count` caps events RETURNED, so filtering cluster-side spends the same
    event budget on a wider window rather than a shorter one of everything."""
    seen = {}

    def fake_fetch(host, user, password, count, severity=None, log_message=None, verify_tls=False, page_size=500):
        seen["severity"] = severity
        yield CLUSTER_EVENT_1

    with patch("app.api.routes_clusters.fetch_ems_events", fake_fetch):
        r = client.post(
            "/api/clusters/fetch-events",
            json={"cluster": "cluster.example.com", "username": "admin", "password": "pw", "count": 1},
        )

    assert r.status_code == 200
    assert seen["severity"] == "emergency|alert|error|notice"
    assert r.json()["severity_filter"] == "notice"


def test_cluster_fetch_with_all_severities_sends_no_severity_filter(client):
    seen = {}

    def fake_fetch(host, user, password, count, severity=None, log_message=None, verify_tls=False, page_size=500):
        seen["severity"] = severity
        yield CLUSTER_EVENT_1

    with patch("app.api.routes_clusters.fetch_ems_events", fake_fetch):
        r = client.post(
            "/api/clusters/fetch-events",
            json={
                "cluster": "cluster.example.com",
                "username": "admin",
                "password": "pw",
                "count": 1,
                "min_severity": "all",
            },
        )

    assert r.status_code == 200
    assert seen["severity"] is None
    assert r.json()["severity_filter"] is None


def test_a_cluster_that_ignores_the_severity_query_still_gets_filtered_locally(client):
    """The floor is applied again at ingestion, so `severity_filter` is a
    guarantee about the rows rather than a record of what was requested. An
    ONTAP version that mishandles the query cannot leave the row claiming a
    filter that never applied."""
    unfiltered = dict(CLUSTER_EVENT_1)
    debug_event = {
        "node": {"name": "node1"},
        "time": "2026-08-10T02:00:00-04:00",
        "message": {"name": "rastrace.dump.saved", "severity": "debug"},
        "log_message": "trace dump stored.",
    }

    def fake_fetch(host, user, password, count, severity=None, log_message=None, verify_tls=False, page_size=500):
        yield unfiltered
        yield debug_event

    with patch("app.api.routes_clusters.fetch_ems_events", fake_fetch):
        r = client.post(
            "/api/clusters/fetch-events",
            json={"cluster": "cluster.example.com", "username": "admin", "password": "pw", "count": 5},
        )

    assert r.status_code == 200
    assert r.json()["event_count"] == 1
    assert r.json()["severity_skipped"] == 1


def test_cluster_fetch_rejects_a_severity_that_cannot_be_ranked(client):
    """Rejected before the fetch: an unrankable name must not reach ONTAP,
    where an invalid enum member 400s the whole walk."""
    r = client.post(
        "/api/clusters/fetch-events",
        json={
            "cluster": "cluster.example.com",
            "username": "admin",
            "password": "pw",
            "min_severity": "critical-ish",
        },
    )
    assert r.status_code == 400


def test_scope_options_carry_a_stage1_cost_estimate(client):
    """Shown before the run, because Stage 1 is a single uncapped call whose
    cost is linear in event count and an autosupport bundle can be far larger
    than a cluster fetch."""
    _upload_and_ingest_sample(client)

    options = client.get("/api/analysis/scope-options").json()
    assert options
    for option in options:
        assert option["estimated_cost_usd"] > 0
        assert option["estimated_cost_usd"] < 0.05  # 3 events


def test_scope_options_resolves_the_file_option_server_side(client):
    """The Files page's Analyze link used to have the browser hand-build this
    option, duplicating both the cluster label and (now) the cost estimate."""
    file_id = _upload_and_ingest_sample(client)["id"]

    options = client.get(f"/api/analysis/scope-options?file_id={file_id}").json()
    assert options[0]["mode"] == "file"
    assert options[0]["file_id"] == file_id
    assert options[0]["event_count"] == 3
    assert options[0]["estimated_cost_usd"] > 0


def test_scope_options_ignores_an_unknown_file_id(client):
    """The rest of the list is still a valid set of choices, so a stale link
    must not blank the selector."""
    _upload_and_ingest_sample(client)

    r = client.get("/api/analysis/scope-options?file_id=9999")
    assert r.status_code == 200
    assert r.json()
    assert all(o["mode"] != "file" for o in r.json())
