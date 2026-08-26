import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

import pytest

from app.agent import stage1
from app.agent.findings import compute_pattern_signature, compute_signature
from app.db import repo_candidates, repo_events, repo_feedback, repo_files, repo_findings

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "app" / "db" / "schema.sql"

CATEGORY = "predictive_failure"
EVENT_NAME = "disk.predictiveFailure"
NODE = "node1"


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


class FakeMessage:
    def __init__(
        self,
        payload: Dict[str, Any],
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        stop_reason="end_turn",
    ):
        self.content = [FakeTextBlock(json.dumps(payload))]
        self.usage = FakeUsage(input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens)
        self.stop_reason = stop_reason


class FakeMessagesClient:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        return self._response


class FakeAnthropicClient:
    def __init__(self, response):
        self.messages = FakeMessagesClient(response)


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text())
    f = repo_files.insert_file(connection, "seed.log", "seed://1", "hash1", 0)
    repo_events.bulk_insert_events(
        connection,
        f.id,
        [
            {
                "raw_line": "raw",
                "event_time": "2026-08-17T00:00:00+00:00",
                "node": NODE,
                "event_name": EVENT_NAME,
                "severity": "alert",
                "message": "Disk predictively failed.",
                "sequence_num": 1,
            }
        ],
    )
    cur = connection.execute("INSERT INTO analysis_runs (status, started_at) VALUES ('running', '2026-01-01T00:00:00+00:00')")
    connection.commit()
    return connection, cur.lastrowid


def _candidate_payload(rank=1, candidate_id="C1"):
    return {
        "candidates": [
            {
                "id": candidate_id,
                "rank": rank,
                "category": CATEGORY,
                "node": NODE,
                "refs": [1],
                "rationale": "Predictive failure event flagged.",
                "confidence": 0.8,
            }
        ]
    }


def test_run_stage1_persists_candidate_and_usage(conn):
    connection, run_id = conn
    client = FakeAnthropicClient(
        FakeMessage(
            _candidate_payload(),
            input_tokens=1234,
            output_tokens=567,
            cache_creation_input_tokens=50000,
            cache_read_input_tokens=0,
        )
    )

    result = stage1.run_stage1(connection, run_id, client=client)

    assert result.candidates_generated == 1
    assert result.candidates_auto_suppressed == 0
    assert result.input_tokens == 1234
    assert result.output_tokens == 567
    assert result.cache_creation_input_tokens == 50000
    assert result.cache_read_input_tokens == 0

    candidates = repo_candidates.list_candidates_for_run(connection, run_id)
    assert len(candidates) == 1
    assert candidates[0].status == "pending"
    assert candidates[0].category == CATEGORY
    assert candidates[0].refs == [1]


def test_run_stage1_persists_candidates_in_rank_order(conn):
    connection, run_id = conn
    payload = {
        "candidates": [
            {"id": "C2", "rank": 2, "category": CATEGORY, "node": NODE, "refs": [1], "rationale": "second", "confidence": 0.5},
            {"id": "C1", "rank": 1, "category": CATEGORY, "node": NODE, "refs": [1], "rationale": "first", "confidence": 0.9},
        ]
    }
    client = FakeAnthropicClient(FakeMessage(payload))

    stage1.run_stage1(connection, run_id, client=client)

    candidates = repo_candidates.list_candidates_for_run(connection, run_id)
    assert [c.rank for c in candidates] == [1, 2]
    assert [c.rationale for c in candidates] == ["first", "second"]


def test_run_stage1_auto_suppresses_candidate_matching_dismissed_feedback(conn):
    connection, run_id = conn
    signature = compute_signature(CATEGORY, [EVENT_NAME], NODE)
    pattern_signature = compute_pattern_signature(CATEGORY, [EVENT_NAME])
    repo_feedback.insert_feedback(connection, finding_id=0, signature=signature, pattern_signature=pattern_signature)
    client = FakeAnthropicClient(FakeMessage(_candidate_payload()))

    result = stage1.run_stage1(connection, run_id, client=client)

    assert result.candidates_auto_suppressed == 1
    candidates = repo_candidates.list_candidates_for_run(connection, run_id)
    assert candidates[0].status == "auto_suppressed"


def test_run_stage1_raises_on_refusal(conn):
    connection, run_id = conn
    client = FakeAnthropicClient(FakeMessage(_candidate_payload(), stop_reason="refusal"))

    with pytest.raises(RuntimeError):
        stage1.run_stage1(connection, run_id, client=client)


def test_run_stage1_raises_on_truncated_response(conn):
    """Thinking shares the max_tokens budget, so a long think plus a long
    candidate list can cut the structured output off mid-JSON. Before this
    guard that surfaced as a bare JSONDecodeError with no hint of the cause."""
    connection, run_id = conn
    client = FakeAnthropicClient(FakeMessage(_candidate_payload(), stop_reason="max_tokens"))

    with pytest.raises(stage1.Stage1Error) as excinfo:
        stage1.run_stage1(connection, run_id, client=client)

    assert "truncated" in str(excinfo.value)


def test_stage1_failures_carry_the_usage_that_was_already_billed(conn):
    """Every failure after the model call still cost real money. Losing the
    usage there is what made a failed run display as $0.0000."""
    connection, run_id = conn
    client = FakeAnthropicClient(
        FakeMessage(
            _candidate_payload(),
            input_tokens=140_000,
            output_tokens=8_000,
            stop_reason="max_tokens",
        )
    )

    with pytest.raises(stage1.Stage1Error) as excinfo:
        stage1.run_stage1(connection, run_id, client=client)

    assert excinfo.value.usage == {
        "input_tokens": 140_000,
        "output_tokens": 8_000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


_WELL_FORMED = {
    "id": "C0",
    "rank": 1,
    "category": CATEGORY,
    "node": NODE,
    "refs": [1],
    "rationale": "x",
    "confidence": 0.8,
}


@pytest.mark.parametrize(
    "candidates",
    [
        [{k: v for k, v in _WELL_FORMED.items() if k != "rank"}],
        [{k: v for k, v in _WELL_FORMED.items() if k != "refs"}],
        [{k: v for k, v in _WELL_FORMED.items() if k != "category"}],
        # Two candidates, because a one-element sort never compares anything —
        # it takes a second element for a non-integer rank to raise at all.
        [_WELL_FORMED, dict(_WELL_FORMED, id="C2", rank="second")],
    ],
    ids=["no rank", "no refs", "no category", "unsortable rank"],
)
def test_a_malformed_candidate_list_still_reports_what_it_cost(conn, candidates):
    """The schema marks these required, so this is a response that didn't match
    the request — but it still arrived AFTER Anthropic billed the whole corpus.

    A bare KeyError here isn't a Stage1Error, so runner.py's generic handler
    recorded the run as failed with no usage at all: the $1.25 run displayed as
    $0.0000 that Stage1Error exists to prevent."""
    connection, run_id = conn
    client = FakeAnthropicClient(
        FakeMessage({"candidates": candidates}, input_tokens=140_000, output_tokens=8_000)
    )

    with pytest.raises(stage1.Stage1Error) as excinfo:
        stage1.run_stage1(connection, run_id, client=client)

    assert "malformed" in str(excinfo.value)
    assert excinfo.value.usage["input_tokens"] == 140_000
    assert excinfo.value.usage["output_tokens"] == 8_000
    # Nothing half-inserted from a list that couldn't be processed.
    assert repo_candidates.list_candidates_for_run(connection, run_id) == []


def test_run_stage1_resolves_refs_to_durable_event_ids(conn):
    """Refs are positional against the corpus that produced them, so Stage 2
    can't safely re-derive them later — Stage 1 resolves them to event ids
    once, here, and stores them with the candidate."""
    connection, run_id = conn
    client = FakeAnthropicClient(FakeMessage(_candidate_payload()))

    stage1.run_stage1(connection, run_id, client=client)

    [candidate] = repo_candidates.list_candidates_for_run(connection, run_id)
    assert len(candidate.leads) == 1
    lead = candidate.leads[0]
    assert lead["event_name"] == EVENT_NAME
    assert lead["node"] == NODE
    assert lead["first_event_id"] == 1
    assert lead["last_event_id"] == 1


def test_run_stage1_skips_a_candidate_whose_refs_resolve_to_nothing(conn):
    """A hallucinated ref list used to be silently dropped, leaving a
    candidate with no event names — whose signature then collided with every
    other empty-name finding in the same category and node."""
    connection, run_id = conn
    payload = {
        "candidates": [
            {
                "id": "C1",
                "rank": 1,
                "category": CATEGORY,
                "node": NODE,
                "refs": [9999],
                "rationale": "Cites a ref that does not exist.",
                "confidence": 0.8,
            }
        ]
    }
    client = FakeAnthropicClient(FakeMessage(payload))

    result = stage1.run_stage1(connection, run_id, client=client)

    assert result.candidates_generated == 0
    assert repo_candidates.list_candidates_for_run(connection, run_id) == []


def test_run_stage1_sends_no_cache_control_and_a_bounded_effort(conn):
    """Stage 1's re-run guard means a cached corpus block essentially never
    gets read, so the 1.25x write was a pure surcharge; and the API's default
    effort of 'high' pays top-of-range reasoning cost on a triage pass."""
    connection, run_id = conn
    captured = {}

    class CapturingMessagesClient(FakeMessagesClient):
        def create(self, **kwargs):
            captured.update(kwargs)
            return self._response

    client = FakeAnthropicClient(FakeMessage(_candidate_payload()))
    client.messages = CapturingMessagesClient(FakeMessage(_candidate_payload()))

    stage1.run_stage1(connection, run_id, client=client)

    assert all("cache_control" not in block for block in captured["system"])
    assert captured["output_config"]["effort"] in {"low", "medium", "high", "xhigh", "max"}


# --- Grounding blocks in the Stage 1 system prompt --------------------------


class CapturingMessagesClient:
    """Records the request kwargs so tests can assert on what actually went
    into the system block — the fakes never validate a request, so this is the
    only way to check the prompt was assembled as intended."""

    def __init__(self, response):
        self._response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self._response


class CapturingClient:
    def __init__(self, response):
        self.messages = CapturingMessagesClient(response)


def _system_text(client):
    return "\n".join(block["text"] for block in client.messages.kwargs["system"])


def test_stage1_injects_catalog_glossary(conn):
    """Without this block Stage 1 is judging dotted event names on how alarming
    they sound; with it, it knows what they mean."""
    connection, run_id = conn
    connection.execute(
        """
        INSERT INTO ems_catalog (name, severity, description, corrective_action, snmp_trap_type, deprecated)
        VALUES (?, 'alert', 'This message occurs when a disk reports a predictive failure.', 'Replace it.', 'standard', 0)
        """,
        (EVENT_NAME,),
    )
    connection.commit()
    client = CapturingClient(FakeMessage(_candidate_payload()))

    stage1.run_stage1(connection, run_id, client=client)

    system = _system_text(client)
    assert "Event catalog" in system
    assert "reports a predictive failure" in system


def test_stage1_glossary_is_scoped_to_ingested_events(conn):
    """The glossary must scale with corpus VARIETY, not with the catalog's
    8000 entries — an unrelated catalog row must not ride along."""
    connection, run_id = conn
    connection.executemany(
        """
        INSERT INTO ems_catalog (name, severity, description, corrective_action, snmp_trap_type, deprecated)
        VALUES (?, 'notice', ?, 'None.', 'standard', 0)
        """,
        [
            (EVENT_NAME, "Predictive failure description."),
            ("unrelated.event.name", "Should not appear anywhere near this prompt."),
        ],
    )
    connection.commit()
    client = CapturingClient(FakeMessage(_candidate_payload()))

    stage1.run_stage1(connection, run_id, client=client)

    system = _system_text(client)
    assert "Predictive failure description." in system
    assert "unrelated.event.name" not in system


def test_stage1_injects_dismissal_reasons(conn):
    """feedback.reason was written and never read by anything. Surfacing it
    lets the model generalize from what a human rejected instead of
    re-proposing near-misses that dodge an exact signature match."""
    connection, run_id = conn
    finding = repo_findings.insert_finding(
        connection,
        analysis_run_id=run_id,
        category=CATEGORY,
        severity="high",
        title="Disk predictive failure on node1",
        description="d",
        recommendation=None,
        node=NODE,
        signature=compute_signature(CATEGORY, [EVENT_NAME], NODE),
        pattern_signature=compute_pattern_signature(CATEGORY, [EVENT_NAME]),
        confidence=0.9,
        evidence_event_ids=[],
    )
    repo_feedback.insert_feedback(
        connection,
        finding_id=finding.id,
        signature=finding.signature,
        pattern_signature=finding.pattern_signature,
        scope="node",
        reason="Lab cluster, these disks are deliberately worn out.",
    )
    connection.commit()
    client = CapturingClient(FakeMessage(_candidate_payload()))

    stage1.run_stage1(connection, run_id, client=client)

    system = _system_text(client)
    assert "previously dismissed" in system
    assert "deliberately worn out" in system


def test_stage1_omits_empty_grounding_blocks(conn):
    """No catalog rows and no feedback should leave no stray headers behind."""
    connection, run_id = conn
    client = CapturingClient(FakeMessage(_candidate_payload()))

    stage1.run_stage1(connection, run_id, client=client)

    system = _system_text(client)
    assert "Event catalog" not in system
    assert "previously dismissed" not in system
