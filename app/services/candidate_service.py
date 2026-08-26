import sqlite3
from typing import Optional

from app.db import repo_candidates
from app.db.models import Candidate

__all__ = ["trigger_investigation", "discard", "InvalidCandidateStateError"]


class InvalidCandidateStateError(Exception):
    """Raised when an action is attempted on a candidate that isn't in the
    state it requires (e.g. investigating an already-investigated candidate)."""


def trigger_investigation(conn: sqlite3.Connection, candidate_id: int) -> Candidate:
    """Synchronously flip the candidate to 'investigating'; the caller is
    responsible for scheduling execute_investigation(candidate_id) (e.g. via
    FastAPI BackgroundTasks)."""
    candidate = repo_candidates.get_candidate(conn, candidate_id)
    if candidate is None:
        raise LookupError(f"candidate {candidate_id} not found")
    if candidate.status != "pending":
        raise InvalidCandidateStateError(f"candidate {candidate_id} is '{candidate.status}', not pending")
    # The check above is not the guard — the repo's conditional UPDATE is, and
    # it raises ValueError if the row moved in between (two clicks, two tabs).
    # Translated here so a lost race reaches the caller as the same 409 the
    # non-race path returns, instead of escaping the route's handlers as a 500
    # with a traceback. The repo keeps raising ValueError: it must not import
    # this module, and its own tests assert that contract.
    try:
        repo_candidates.start_investigation(conn, candidate_id)
    except ValueError as exc:
        raise InvalidCandidateStateError(str(exc)) from exc
    return repo_candidates.get_candidate(conn, candidate_id)


def discard(conn: sqlite3.Connection, candidate_id: int, reason: Optional[str] = None) -> Candidate:
    candidate = repo_candidates.get_candidate(conn, candidate_id)
    if candidate is None:
        raise LookupError(f"candidate {candidate_id} not found")
    if candidate.status != "pending":
        raise InvalidCandidateStateError(f"candidate {candidate_id} is '{candidate.status}', not pending")
    # Same race translation as trigger_investigation above.
    try:
        repo_candidates.discard(conn, candidate_id, reason=reason)
    except ValueError as exc:
        raise InvalidCandidateStateError(str(exc)) from exc
    return repo_candidates.get_candidate(conn, candidate_id)
