import hashlib
import sqlite3
from typing import Any, Dict, List, Optional

from app.agent.pricing import STAGE2_COST_CAP_USD
from app.db import repo_feedback, repo_findings
from app.db.models import Finding

# The closed sets a hypothesis must use. Enforced at the tool boundary
# (tools.record_hypothesis), which is the only place model-authored values
# enter the system.
#
# This matters more than it looks: a signature is computed FROM the category
# (compute_signature below), so an off-enum category produces a signature
# Stage 1 can never reproduce — meaning is_suppressed and
# find_open_by_signature can never match it, and dismissing such a finding
# suppresses nothing at all. Severity is closed for a duller reason: the
# findings page filters on it, and an unknown value is invisible there.
VALID_CATEGORIES = {"availability_risk", "performance_issue", "predictive_failure"}
VALID_SEVERITIES = {"critical", "high", "medium", "low"}

# A hypothesis is either still being worked on or ready to become a finding.
# graph.persist_findings compares to READY_STATUS exactly, so an unnormalized
# "Ready" would silently demote a finalized hypothesis to a partial.
READY_STATUS = "ready"
INVESTIGATING_STATUS = "investigating"
VALID_HYPOTHESIS_STATUSES = {READY_STATUS, INVESTIGATING_STATUS}

# What conclude_investigation's `outcome` can say. This is a separate axis from
# a hypothesis's status and the two can disagree — an agent can announce
# 'confirmed' having never recorded a hypothesis at status='ready'. Which of the
# two graph.persist_findings trusts for which decision is the whole point of the
# distinction: `outcome` decides whether a refutation is written, `status`
# decides whether an open finding is. Defaulted to REFUTED in
# tools.conclude_investigation, so an unrecognized value lands there.
CONFIRMED_OUTCOME = "confirmed"
REFUTED_OUTCOME = "refuted"


def compute_pattern_signature(category: str, event_names: List[str]) -> str:
    key = f"{category}|{'|'.join(sorted(set(event_names)))}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def compute_signature(category: str, event_names: List[str], node: Optional[str]) -> str:
    key = f"{compute_pattern_signature(category, event_names)}|{node or ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def is_suppressed(conn: sqlite3.Connection, category: str, event_names: List[str], node: Optional[str]) -> bool:
    signature = compute_signature(category, event_names, node)
    pattern_signature = compute_pattern_signature(category, event_names)
    return repo_feedback.check_suppression(conn, signature, pattern_signature)


# A completed investigation that ruled its candidate OUT. Stored alongside
# findings so every Stage 2 result lives in one place, but as a distinct
# status carrying three deliberate consequences:
#
# 1. It is NOT an open finding, so `find_open_by_signature` (which filters
#    status='open') never matches it. This is load-bearing, not incidental:
#    a pattern ruled out today can escalate into a real problem next week,
#    and both persist_hypothesis and Stage 1's pre-filter skip anything
#    matching an open finding. If a refutation counted as one, that later
#    real finding could never be created. Pinned by test.
# 2. It is not dismissible, so it never writes suppression feedback.
#    Agreeing that something isn't a problem is not the same signal as a
#    human overriding the agent, and conflating them would poison the
#    feedback the model is shown.
# 3. Its severity is informational — it is a record of work done, not a
#    risk, and it must not appear under a severity filter for real issues.
NO_ISSUE_STATUS = "no_issue"
NO_ISSUE_SEVERITY = "info"

# An investigation that ended while it still had a working hypothesis on the
# table — normally because the budget cut it short. The agent got far enough to
# state what it suspected and which events supported it, but never reached
# status='ready' — so the money was spent and, before this existed, the result
# was discarded and the user saw nothing but a tool trace and a bill.
#
# It shares property 1 of NO_ISSUE_STATUS for the same reason: it is NOT an open
# finding, so `find_open_by_signature` never matches it and a later, properly
# concluded investigation of the same pattern can still create a real finding.
# It differs on property 2 — a partial IS dismissible. Dismissing it is a human
# overriding a claimed risk, which is exactly the signal dismissal feedback is
# for; that the agent ran out of budget before confirming it doesn't change what
# the human is saying.
PARTIAL_STATUS = "partial"

# Why the investigation left this hypothesis unfinalized, as a sentence
# completing "the investigation ...{turns}". Each entry has to read correctly with
# the turn count spliced in, which is why they are whole clauses rather than
# fragments assembled at the call site.
CONCLUDED_UNFINALIZED = "concluded_unfinalized"
_STOP_REASON_TEXT = {
    "cost_cap": (
        f"stopped at the ${STAGE2_COST_CAP_USD:.2f} cost cap{{turns}}, before it could confirm "
        "or rule out this hypothesis"
    ),
    "iteration_cap": (
        "stopped at the turn limit{turns}, before it could confirm or rule out this hypothesis"
    ),
    # Not a budget bound. The agent said it had CONFIRMED the candidate but never
    # re-recorded the hypothesis at status='ready', so nothing went through the
    # finalize step. Keeping the working hypothesis is the honest outcome: the
    # alternative used to be filing it as "no issue found" (see graph.py), which
    # stated the opposite of what the agent concluded.
    CONCLUDED_UNFINALIZED: (
        "ended{turns} with the agent stating it had confirmed this candidate, but it never "
        "finalized the hypothesis, so nothing was checked against suppression or promoted "
        "to a confirmed finding"
    ),
}
_DEFAULT_STOP_REASON_TEXT = (
    "stopped{turns} before reaching a conclusion on this hypothesis"
)


def _partial_note(stop_reason: str, iterations: Optional[int]) -> str:
    """The provenance line appended to a partial finding's description.

    Kept out of the agent's own text (below a separator) so the user can tell
    what the model actually asserted from what the system is telling them about
    how far it got."""
    template = _STOP_REASON_TEXT.get(stop_reason, _DEFAULT_STOP_REASON_TEXT)
    turns = f" after {iterations} turns" if iterations else ""
    return (
        f"\n\nUnconfirmed: the investigation {template.format(turns=turns)}. This is the "
        "working hypothesis and evidence it had reached at that point — treat it as a lead to "
        "follow up, not a confirmed finding."
    )


def persist_partial(
    conn: sqlite3.Connection,
    analysis_run_id: Optional[int],
    hypothesis: Dict[str, Any],
    candidate_id: Optional[int],
    stop_reason: str,
    iterations: Optional[int] = None,
) -> Optional[Finding]:
    """Record an in-progress hypothesis from an investigation that ended without
    finalizing it, so the partial result survives instead of being thrown away.
    `stop_reason` picks the note explaining why (see _STOP_REASON_TEXT): a budget
    bound, or an agent that announced a confirmation it never finalized.

    Suppression and the open-finding check apply exactly as they do to a
    finalized hypothesis: an unfinished investigation is not a licence to
    re-raise something a human already dismissed, or to duplicate a finding
    that is already open. Returns None in either case."""
    category = hypothesis["category"]
    event_names = hypothesis.get("event_names", [])
    node = hypothesis.get("node")
    signature = compute_signature(category, event_names, node)
    pattern_signature = compute_pattern_signature(category, event_names)

    if repo_feedback.check_suppression(conn, signature, pattern_signature):
        return None
    if repo_findings.find_open_by_signature(conn, signature) is not None:
        return None

    return repo_findings.insert_finding(
        conn,
        analysis_run_id=analysis_run_id,
        category=category,
        severity=hypothesis.get("severity", "medium"),
        title=hypothesis["title"],
        description=hypothesis["description"] + _partial_note(stop_reason, iterations),
        recommendation=hypothesis.get("recommendation"),
        node=node,
        signature=signature,
        pattern_signature=pattern_signature,
        confidence=hypothesis.get("confidence"),
        evidence_event_ids=hypothesis.get("evidence_event_ids", []),
        candidate_id=candidate_id,
        status=PARTIAL_STATUS,
    )


def persist_refutation(
    conn: sqlite3.Connection,
    analysis_run_id: Optional[int],
    candidate_id: int,
    category: str,
    node: Optional[str],
    event_names: List[str],
    summary: str,
) -> Finding:
    """Record that an investigation completed and found no real issue.

    `category` and `event_names` describe what was INVESTIGATED, not what was
    found, so the record can be filtered alongside the risks it was checked
    against."""
    return repo_findings.insert_finding(
        conn,
        analysis_run_id=analysis_run_id,
        category=category,
        severity=NO_ISSUE_SEVERITY,
        title="No issue found",
        description=summary,
        recommendation=None,
        node=node,
        signature=compute_signature(category, event_names, node),
        pattern_signature=compute_pattern_signature(category, event_names),
        confidence=None,
        evidence_event_ids=[],
        candidate_id=candidate_id,
        status=NO_ISSUE_STATUS,
    )


def persist_hypothesis(
    conn: sqlite3.Connection,
    analysis_run_id: Optional[int],
    hypothesis: Dict[str, Any],
    candidate_id: Optional[int] = None,
) -> Optional[Finding]:
    """Insert a finding for this hypothesis unless it (or its node-agnostic
    pattern) has previously been dismissed, or an open finding with the same
    signature already exists (e.g. from a prior run over the same events —
    without this check, re-analyzing an unchanged event set would create a
    fresh duplicate finding every time). Returns None if suppressed."""
    category = hypothesis["category"]
    event_names = hypothesis.get("event_names", [])
    node = hypothesis.get("node")
    signature = compute_signature(category, event_names, node)
    pattern_signature = compute_pattern_signature(category, event_names)

    if repo_feedback.check_suppression(conn, signature, pattern_signature):
        return None
    if repo_findings.find_open_by_signature(conn, signature) is not None:
        return None

    return repo_findings.insert_finding(
        conn,
        analysis_run_id=analysis_run_id,
        category=category,
        severity=hypothesis.get("severity", "medium"),
        title=hypothesis["title"],
        description=hypothesis["description"],
        recommendation=hypothesis.get("recommendation"),
        node=node,
        signature=signature,
        pattern_signature=pattern_signature,
        confidence=hypothesis.get("confidence"),
        evidence_event_ids=hypothesis.get("evidence_event_ids", []),
        candidate_id=candidate_id,
    )
