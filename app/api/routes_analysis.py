import sqlite3
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.deps import get_db
from app.api.schemas import (
    AnalysisRunOut,
    CandidateOut,
    ScopeOptionOut,
    ScopeOut,
    TriggerAnalysisRequest,
    TriggerAnalysisResponse,
)
from app.db import repo_analysis_runs, repo_candidates, repo_events
from app.db.session import session as db_session
from app.services.analysis_service import (
    InvalidScopeError,
    NoNewDataError,
    available_scopes,
    execute_run,
    scope_option,
    trigger,
)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.post("/runs", response_model=TriggerAnalysisResponse)
def trigger_run(background_tasks: BackgroundTasks, body: Optional[TriggerAnalysisRequest] = None):
    # Uses its own short-lived session (committed and closed before this
    # function returns) instead of the shared get_db dependency, so the
    # analysis_runs insert is guaranteed durable before execute_run's
    # background-task connection tries to write to the same SQLite file —
    # otherwise the two connections can race and SQLite raises "database is
    # locked".
    request = body or TriggerAnalysisRequest()
    try:
        with db_session() as conn:
            run_id = trigger(conn, mode=request.mode, cluster=request.cluster, file_id=request.file_id)
    except NoNewDataError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidScopeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(execute_run, run_id)
    return TriggerAnalysisResponse(run_id=run_id, status="running")


@router.get("/runs", response_model=List[AnalysisRunOut])
def list_runs(limit: int = 50, offset: int = 0, conn: sqlite3.Connection = Depends(get_db)):
    return [
        AnalysisRunOut.model_validate(r, from_attributes=True)
        for r in repo_analysis_runs.list_runs(conn, limit, offset)
    ]


@router.get("/runs/{run_id}", response_model=AnalysisRunOut)
def get_run(run_id: int, conn: sqlite3.Connection = Depends(get_db)):
    run = repo_analysis_runs.get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    return AnalysisRunOut.model_validate(run, from_attributes=True)


@router.get("/runs/{run_id}/candidates", response_model=List[CandidateOut])
def list_run_candidates(run_id: int, conn: sqlite3.Connection = Depends(get_db)):
    if repo_analysis_runs.get_run(conn, run_id) is None:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    return [
        CandidateOut.model_validate(c, from_attributes=True)
        for c in repo_candidates.list_candidates_for_run(conn, run_id)
    ]


@router.get("/scope", response_model=ScopeOut)
def get_scope(conn: sqlite3.Connection = Depends(get_db)):
    """The full set of currently-ingested events — what exists, across all
    clusters. Distinct from /scope-options, which is what a single run can
    actually be pointed at."""
    return ScopeOut.model_validate(repo_events.get_scope_summary_stats(conn))


@router.get("/scope-options", response_model=List[ScopeOptionOut])
def get_scope_options(
    file_id: Optional[int] = None, conn: sqlite3.Connection = Depends(get_db)
):
    """The concrete choices for a new run: two per cluster present in the
    data. Enumerated server-side so the user picks one thing rather than
    combining a cluster and a mode themselves.

    `file_id` prepends the one extra option for a specific file, as arrived at
    via the Files page's Analyze link. Resolved here rather than assembled in
    the browser so that a scope's cluster label and cost estimate have exactly
    one source; an unknown id is simply omitted, since the rest of the list is
    still a valid set of choices."""
    options = available_scopes(conn)
    if file_id is not None:
        try:
            options.insert(0, scope_option(conn, "file", None, file_id))
        except InvalidScopeError:
            pass
    return [ScopeOptionOut(**option) for option in options]
