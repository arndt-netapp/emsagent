import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from app.agent import compaction
from app.agent.findings import compute_signature, is_suppressed
from app.agent.prompts import (
    STAGE1_SYSTEM_PROMPT,
    STAGE1_TASK,
    render_dismissals,
    render_glossary,
    render_scope_summary,
)
from app.config import settings
from app.db import repo_candidates, repo_catalog, repo_events, repo_feedback, repo_findings
from app.db.models import EventScope

logger = logging.getLogger(__name__)

# Room for thinking (Sonnet 5 thinks by default) plus the candidate list.
# Generous on purpose: unused headroom is free, whereas running out mid-JSON
# costs the entire corpus with nothing to show for it.
STAGE1_MAX_TOKENS = 16000

STAGE1_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Short local id, e.g. C1, C2."},
                    "rank": {"type": "integer", "description": "Priority order, 1 = investigate first."},
                    "category": {
                        "type": "string",
                        "enum": ["availability_risk", "performance_issue", "predictive_failure"],
                    },
                    "node": {"type": ["string", "null"]},
                    "refs": {
                        "type": "array",
                        "items": {"type": "integer"},
                        # NOT `maxItems`: the structured-output validator rejects it
                        # outright ("For 'array' type, property 'maxItems' is not
                        # supported", HTTP 400). The ceiling is stated in the
                        # description so the model self-limits, and enforced for real
                        # server-side by compaction.resolve_leads, which is what
                        # actually bounds Stage 2's seed message.
                        "description": (
                            f"The compact-corpus `ref` values this candidate is based on. "
                            f"Cite the rows that establish the pattern, not every row it touches; "
                            f"at most {compaction.MAX_CANDIDATE_LEADS} are carried forward."
                        ),
                    },
                    "rationale": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "rank", "category", "node", "refs", "rationale", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


@dataclass
class Stage1Result:
    candidates_generated: int
    candidates_auto_suppressed: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


class Stage1Error(RuntimeError):
    """A Stage 1 failure that happened AFTER the model call was billed.

    Carries the usage so the run can be recorded with its real cost instead
    of the $0.00 a bare exception would leave behind. `usage` is None only
    when nothing was billed (e.g. the request itself never completed)."""

    def __init__(self, message: str, usage: Optional[Dict[str, int]] = None):
        super().__init__(message)
        self.usage = usage


def _extract_usage(response: Any) -> Dict[str, int]:
    """Pull all four billed token buckets off a raw-SDK response.

    The raw `anthropic` SDK's `usage.input_tokens` EXCLUDES cached tokens —
    `cache_creation_input_tokens` / `cache_read_input_tokens` are separate
    fields carrying their own (1.25x / 0.1x) multipliers. This is the exact
    opposite of langchain_anthropic's convention, which Stage 2 has to unpack
    instead (see graph.py); don't copy one stage's handling into the other."""
    usage = response.usage
    return {
        "input_tokens": usage.input_tokens or 0,
        "output_tokens": usage.output_tokens or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def run_stage1(
    conn: sqlite3.Connection,
    analysis_run_id: int,
    model_name: Optional[str] = None,
    client: Optional[Any] = None,
    scope: Optional[EventScope] = None,
) -> Stage1Result:
    client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
    model = model_name or settings.claude_model

    groups = compaction.build_compact_corpus(conn, scope)
    groups_by_ref = {g.ref: g for g in groups}
    corpus_text = compaction.render_compact_corpus(groups)
    scope_summary = render_scope_summary(repo_events.get_scope_summary_stats(conn, scope))

    # Grounding blocks. Both are bounded by corpus *variety* and feedback
    # count, not by event count, so neither grows with the thing that actually
    # scales — see repo_catalog.glossary_for_events and repo_feedback
    # .recent_dismissals. The glossary is truncated to one short line per
    # event name: Stage 1 needs to know what an event means to judge whether a
    # pattern matters, while the full text and corrective action are a Stage 2
    # tool call away, priced under the cost cap rather than paid for here.
    glossary = render_glossary(repo_catalog.glossary_for_events(conn, scope))
    dismissals = render_dismissals(repo_feedback.recent_dismissals(conn))
    context_blocks = "\n\n".join(block for block in (glossary, dismissals) if block)

    # NOTE: no cache_control on the corpus block, deliberately. A 5-minute
    # cache write costs 1.25x base input and only pays for itself on a read,
    # but analysis_service.trigger refuses a re-run unless events changed (in
    # which case the corpus text differs and the block can't match) or new
    # feedback arrived (identical corpus, but a hit needs the human to
    # re-run inside 5 minutes). So the cache was structurally write-only:
    # a flat ~25% surcharge on the term that dominates this request, in
    # exchange for a hit that essentially never lands. Plain input is
    # strictly cheaper here. Stage 2's cache is a different mechanism over a
    # growing conversation and does pay off — see graph.py.
    response = client.messages.create(
        model=model,
        max_tokens=STAGE1_MAX_TOKENS,
        system=[
            {"type": "text", "text": STAGE1_SYSTEM_PROMPT},
            {"type": "text", "text": f"Scope: {scope_summary}\n\n{context_blocks}\n\n{corpus_text}"},
        ],
        messages=[{"role": "user", "content": STAGE1_TASK}],
        output_config={
            "format": {"type": "json_schema", "schema": STAGE1_SCHEMA},
            "effort": settings.stage1_effort,
        },
    )
    # Capture usage before any of the checks below: everything from here on
    # fails AFTER Anthropic has billed the request, so each raise carries the
    # cost with it (see Stage1Error).
    usage = _extract_usage(response)

    if response.stop_reason == "refusal":
        raise Stage1Error("Stage 1 request was refused", usage=usage)
    if response.stop_reason == "max_tokens":
        # Thinking tokens share the max_tokens budget, so a long think plus a
        # long candidate list can truncate the structured output mid-JSON.
        # Without this check that surfaced as a bare JSONDecodeError with no
        # hint of the real cause.
        raise Stage1Error(
            f"Stage 1 response was truncated at the {STAGE1_MAX_TOKENS}-token limit before the "
            "candidate list was complete — raise STAGE1_MAX_TOKENS or lower stage1_effort.",
            usage=usage,
        )

    # Sonnet 5 thinks by default, so content[0] may be a thinking block, not
    # the structured-output text block — find the text block explicitly.
    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        raise Stage1Error(
            f"Stage 1 response had no text block (stop_reason={response.stop_reason})", usage=usage
        )
    try:
        data = json.loads(text_block.text)
    except ValueError as exc:
        raise Stage1Error(f"Stage 1 returned unparseable JSON: {exc}", usage=usage)
    unsorted_candidates: List[Dict[str, Any]] = data.get("candidates", [])
    try:
        to_insert, auto_suppressed = _build_candidates(conn, unsorted_candidates, groups_by_ref)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        # The schema marks these fields required, so reaching here means the
        # response didn't match the schema the request asked for. It still has
        # to raise Stage1Error rather than escape: everything past the model
        # call has already been billed for the whole corpus, and runner.py's
        # generic handler records a failed run with no usage at all — the $1.25
        # run displayed as $0.0000 that Stage1Error exists to prevent.
        raise Stage1Error(f"Stage 1 returned a malformed candidate list: {exc!r}", usage=usage)

    repo_candidates.bulk_insert_candidates(conn, analysis_run_id, to_insert)

    return Stage1Result(
        candidates_generated=len(to_insert),
        candidates_auto_suppressed=auto_suppressed,
        **usage,
    )


def _build_candidates(
    conn: sqlite3.Connection,
    unsorted_candidates: List[Dict[str, Any]],
    groups_by_ref: Dict[int, compaction.CompactEventGroup],
) -> Tuple[List[repo_candidates.CandidateInput], int]:
    """Turn the model's raw candidate list into rows to insert, plus how many
    were auto-suppressed. Split out from run_stage1 so every way it can choke
    on a malformed response is caught in one place and re-raised carrying the
    usage — see the caller."""
    # Stable sort by rank: enumerate() first rather than calling
    # unsorted_candidates.index() from within the key function — CPython
    # empties a list's buffer while it's being sorted, so calling a method
    # on the very list being sorted raises "x not in list".
    raw_candidates = [c for _, c in sorted(enumerate(unsorted_candidates), key=lambda pair: (pair[1]["rank"], pair[0]))]

    to_insert: List[repo_candidates.CandidateInput] = []
    auto_suppressed = 0
    for c in raw_candidates:
        node = c.get("node")
        refs = c["refs"]
        # Resolve refs to durable event ids NOW, while the corpus that
        # numbered them is still the corpus on disk. Stage 2 reads these
        # rather than re-deriving positional refs against a possibly-changed
        # event set. See compaction.resolve_leads.
        leads = compaction.resolve_leads(groups_by_ref, refs)
        unresolved = [r for r in refs if r not in groups_by_ref]
        if unresolved:
            # The model can cite refs that don't exist. Dropping them
            # silently is how a candidate ends up with an empty event_names
            # list, whose signature then collides with every other
            # empty-name finding in the same category/node.
            logger.warning(
                "stage 1 candidate rank %s cited %s ref(s) not present in the corpus: %s",
                c.get("rank"),
                len(unresolved),
                unresolved,
            )
        event_names = sorted({lead["event_name"] for lead in leads})
        if not event_names:
            logger.warning(
                "stage 1 candidate rank %s resolved to no events; skipping it rather than "
                "creating a candidate with an empty signature",
                c.get("rank"),
            )
            continue
        signature = compute_signature(c["category"], event_names, node)
        status = "pending"
        if is_suppressed(conn, c["category"], event_names, node) or repo_findings.find_open_by_signature(
            conn, signature
        ):
            status = "auto_suppressed"
            auto_suppressed += 1
        to_insert.append(
            repo_candidates.CandidateInput(
                rank=c["rank"],
                category=c["category"],
                node=node,
                rationale=c["rationale"],
                confidence=c.get("confidence"),
                refs=refs,
                leads=leads,
                status=status,
            )
        )
    return to_insert, auto_suppressed
