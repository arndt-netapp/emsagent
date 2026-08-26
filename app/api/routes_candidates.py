import sqlite3

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.agent.runner import execute_investigation
from app.api.deps import get_db
from app.api.schemas import (
    AgentStepOut,
    CandidateDetailOut,
    CandidateOut,
    DiscardCandidateRequest,
    FindingOut,
    InvestigateCandidateResponse,
)
from app.db import repo_agent_steps, repo_candidates, repo_findings
from app.db.session import session as db_session
from app.services.candidate_service import InvalidCandidateStateError, discard, trigger_investigation

router = APIRouter(prefix="/api/candidates", tags=["candidates"])


@router.get("/{candidate_id}", response_model=CandidateDetailOut)
def get_candidate(candidate_id: int, conn: sqlite3.Connection = Depends(get_db)):
    candidate = repo_candidates.get_candidate(conn, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    findings = repo_findings.list_findings_for_candidate(conn, candidate_id)
    steps = repo_agent_steps.list_steps_for_candidate(conn, candidate_id)
    return CandidateDetailOut(
        **CandidateOut.model_validate(candidate, from_attributes=True).model_dump(),
        findings=[FindingOut.model_validate(f, from_attributes=True) for f in findings],
        steps=[AgentStepOut(**s) for s in steps],
    )


@router.post("/{candidate_id}/investigate", response_model=InvestigateCandidateResponse)
def investigate(candidate_id: int, background_tasks: BackgroundTasks):
    # Own short-lived session, same rationale as routes_analysis.trigger_run:
    # commit before the background task's connection touches the same row.
    try:
        with db_session() as conn:
            trigger_investigation(conn, candidate_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidCandidateStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(execute_investigation, candidate_id)
    return InvestigateCandidateResponse(candidate_id=candidate_id, status="investigating")


@router.post("/{candidate_id}/discard", response_model=CandidateOut)
def discard_candidate(candidate_id: int, body: DiscardCandidateRequest, conn: sqlite3.Connection = Depends(get_db)):
    try:
        candidate = discard(conn, candidate_id, reason=body.reason)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidCandidateStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return CandidateOut.model_validate(candidate, from_attributes=True)
