import sqlite3
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_db
from app.agent.findings import NO_ISSUE_STATUS
from app.agent.pricing import estimate_cost_usd
from app.api.schemas import AgentStepOut, DismissRequest, EventOut, FindingDetailOut, FindingOut
from app.db import repo_agent_steps, repo_candidates, repo_feedback, repo_findings

router = APIRouter(prefix="/api/findings", tags=["findings"])


@router.get("", response_model=List[FindingOut])
def list_findings(
    status: Optional[str] = None,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    conn: sqlite3.Connection = Depends(get_db),
):
    findings = repo_findings.list_findings(conn, status, category, severity, limit, offset)
    return [FindingOut.model_validate(f, from_attributes=True) for f in findings]


@router.get("/{finding_id}", response_model=FindingDetailOut)
def get_finding(finding_id: int, conn: sqlite3.Connection = Depends(get_db)):
    finding = repo_findings.get_finding(conn, finding_id)
    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    evidence = repo_findings.get_finding_evidence(conn, finding_id)

    # How this conclusion was reached, and what reaching it cost. Both hang off
    # the candidate: agent_steps is keyed by candidate_id, and the investigation
    # token columns live on the candidates row. Findings created before
    # candidates existed (or by a path that set no candidate_id) simply have no
    # trace, which is reported as an empty list rather than an error.
    steps = []
    rank = None
    iterations = None
    cost = None
    if finding.candidate_id is not None:
        steps = repo_agent_steps.list_steps_for_candidate(conn, finding.candidate_id)
        candidate = repo_candidates.get_candidate(conn, finding.candidate_id)
        if candidate is not None:
            rank = candidate.rank
            iterations = candidate.investigation_iterations
            if candidate.investigation_input_tokens is not None:
                cost = estimate_cost_usd(
                    candidate.investigation_input_tokens,
                    candidate.investigation_output_tokens or 0,
                    candidate.investigation_cache_creation_input_tokens or 0,
                    candidate.investigation_cache_read_input_tokens or 0,
                )

    return FindingDetailOut(
        **FindingOut.model_validate(finding, from_attributes=True).model_dump(),
        evidence=[EventOut.model_validate(e, from_attributes=True) for e in evidence],
        steps=[AgentStepOut(**s) for s in steps],
        analysis_run_id=finding.analysis_run_id,
        candidate_rank=rank,
        investigation_iterations=iterations,
        investigation_cost_usd=cost,
    )


@router.post("/{finding_id}/dismiss", response_model=FindingOut)
def dismiss_finding(finding_id: int, body: DismissRequest, conn: sqlite3.Connection = Depends(get_db)):
    finding = repo_findings.get_finding(conn, finding_id)
    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    if finding.status == NO_ISSUE_STATUS:
        # The UI hides the dismiss form for these, but the endpoint enforces it
        # too: dismissing writes suppression feedback, and agreeing that
        # something is not a problem is a different signal from a human
        # overriding the agent. Conflating them would poison the dismissal
        # reasons Stage 1 is shown.
        raise HTTPException(
            status_code=409,
            detail="This record is a completed investigation that found no issue — there is nothing to dismiss.",
        )

    repo_feedback.insert_feedback(
        conn,
        finding_id=finding_id,
        signature=finding.signature,
        pattern_signature=finding.pattern_signature,
        scope=body.scope,
        reason=body.reason,
    )
    repo_findings.dismiss(conn, finding_id)
    return FindingOut.model_validate(repo_findings.get_finding(conn, finding_id), from_attributes=True)
