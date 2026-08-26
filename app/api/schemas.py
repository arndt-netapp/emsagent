import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field

from app.agent.pricing import estimate_cost_usd
from app.ontap.client import CLUSTER_HOST_PATTERN
from app.severity import DEFAULT_MIN_SEVERITY


class FileOut(BaseModel):
    id: int
    filename: str
    # Recovered from the fetch-output filename convention, or set directly by
    # the cluster fetch endpoint. None means an uploaded file of unknown
    # provenance, which analysis treats as the "unspecified" pseudo-cluster.
    cluster: Optional[str] = None
    status: str
    detected_format: Optional[str]
    event_count: int
    duplicates_skipped: int = 0
    # The severity floor these events were ingested under (None = all
    # severities), and how many parsed events it dropped.
    severity_filter: Optional[str] = None
    severity_skipped: int = 0
    discovered_at: str
    ingested_at: Optional[str]
    error_message: Optional[str]


class IngestRequest(BaseModel):
    file_ids: Optional[List[int]] = None


class IngestResult(BaseModel):
    processed: int
    failed: int
    event_count: int
    discovered: Optional[int] = None


class TriggerAnalysisResponse(BaseModel):
    run_id: int
    status: str


class ScopeOut(BaseModel):
    """Snapshot of the ingested EMS event set: what's currently available to
    analyze, or (via AnalysisRunOut.scope) what a completed run covered."""

    total: int
    min_time: Optional[str]
    max_time: Optional[str]
    nodes: List[str]
    severity_counts: Dict[str, int]
    files: List[str]
    cluster: Optional[str] = None
    scope_label: Optional[str] = None


class AnalysisRunOut(BaseModel):
    id: int
    status: str
    started_at: str
    completed_at: Optional[str]
    events_considered: int
    iterations: int
    candidates_generated: int
    candidates_auto_suppressed: int
    error_message: Optional[str]
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    # What this run was pointed at, shown in run history so a list of runs is
    # readable when they cover different clusters and windows.
    scope_label: Optional[str] = None
    scope_cluster: Optional[str] = None
    scope_mode: Optional[str] = None
    scope_json: Optional[str] = Field(default=None, exclude=True)

    @computed_field  # type: ignore[misc]
    @property
    def cost_usd(self) -> float:
        """Approximate cost of this run's Stage 1 model call, at Sonnet 5 list
        pricing — includes prompt-cache write/read cost, which dominates once
        the compact corpus is large (see app/agent/pricing.py)."""
        return estimate_cost_usd(
            self.input_tokens, self.output_tokens, self.cache_creation_input_tokens, self.cache_read_input_tokens
        )

    @computed_field  # type: ignore[misc]
    @property
    def scope(self) -> Optional[ScopeOut]:
        """Which events/files/nodes this run covered, snapshotted at
        completion time — None for runs that never completed."""
        if not self.scope_json:
            return None
        try:
            return ScopeOut.model_validate(json.loads(self.scope_json))
        except (TypeError, ValueError):
            return None


class EventOut(BaseModel):
    id: int
    event_time: Optional[str]
    node: Optional[str]
    event_name: str
    severity: Optional[str]
    message: Optional[str]


class FindingOut(BaseModel):
    id: int
    candidate_id: Optional[int]
    category: str
    severity: str
    title: str
    description: str
    recommendation: Optional[str]
    node: Optional[str]
    status: str
    confidence: Optional[float]
    created_at: str
    dismissed_at: Optional[str]


class AgentStepOut(BaseModel):
    """One tool call from a Stage 2 investigation — the audit trail that makes
    an agent's conclusion reviewable instead of asserted."""

    id: int
    iteration: int
    step_index: int
    tool_name: str
    tool_args: Optional[dict] = None
    result_summary: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    created_at: str


class FindingDetailOut(FindingOut):
    """A finding plus everything needed to judge it: the events it cites, the
    tool calls that produced it, and what that investigation cost.

    The trace lives here rather than only on the Analysis page because this is
    where someone decides whether to act on or dismiss the conclusion — the
    audit trail belongs next to the consequential action, not on a page you
    stop visiting once the run finishes."""

    evidence: List[EventOut]
    steps: List[AgentStepOut] = []
    analysis_run_id: Optional[int] = None
    candidate_rank: Optional[int] = None
    investigation_iterations: Optional[int] = None
    investigation_cost_usd: Optional[float] = None


class DismissRequest(BaseModel):
    reason: Optional[str] = None
    scope: str = Field(default="node", pattern="^(node|global)$")


class ClusterFetchRequest(BaseModel):
    # Constrained to a hostname/IPv4: this value is used directly as the REST
    # host here AND stored as the events' cluster, from where the Stage 2 agent
    # later re-queries it with the configured credentials. See
    # ontap.client.CLUSTER_HOST_PATTERN.
    cluster: str = Field(pattern=CLUSTER_HOST_PATTERN)
    username: str
    password: str
    count: int = 500
    # A severity FLOOR, not a single severity to match: the route expands it to
    # ONTAP's OR-list before it becomes a query parameter. Defaults to
    # notice-and-higher, so an omitted field filters — deliberately, since the
    # events this drops are informational/debug noise that would otherwise
    # consume both the `count` ceiling and Stage 1 budget. Send "all" (or null)
    # for every severity. Validated in the route against app/severity.py, which
    # is also what keeps an invalid enum member from reaching the cluster and
    # 400ing the whole fetch.
    min_severity: Optional[str] = DEFAULT_MIN_SEVERITY
    log_message: Optional[str] = None
    verify_tls: bool = False


class CandidateOut(BaseModel):
    id: int
    analysis_run_id: int
    rank: int
    category: str
    node: Optional[str]
    rationale: str
    confidence: Optional[float]
    refs: List[int]
    status: str
    discard_reason: Optional[str]
    investigation_input_tokens: Optional[int]
    investigation_output_tokens: Optional[int]
    investigation_cache_creation_input_tokens: Optional[int]
    investigation_cache_read_input_tokens: Optional[int]
    investigation_iterations: Optional[int]
    investigation_started_at: Optional[str]
    investigation_completed_at: Optional[str]
    investigation_error: Optional[str]
    created_at: str

    @computed_field  # type: ignore[misc]
    @property
    def investigation_cost_usd(self) -> Optional[float]:
        if self.investigation_input_tokens is None:
            return None
        return estimate_cost_usd(
            self.investigation_input_tokens,
            self.investigation_output_tokens or 0,
            self.investigation_cache_creation_input_tokens or 0,
            self.investigation_cache_read_input_tokens or 0,
        )


class CandidateDetailOut(CandidateOut):
    findings: List[FindingOut]
    steps: List[AgentStepOut] = []


class DiscardCandidateRequest(BaseModel):
    reason: Optional[str] = None


class InvestigateCandidateResponse(BaseModel):
    candidate_id: int
    status: str


class TriggerAnalysisRequest(BaseModel):
    """Which events a new run should cover. Defaults to the last 24 hours of
    the unspecified (uploaded-logs) cluster, which is the right default for the
    drop-a-file-in workflow and harmless otherwise — the UI always sends an
    explicit choice."""

    mode: str = Field(default="last_24h", pattern="^(recent_pull|last_24h|file)$")
    cluster: Optional[str] = None
    # Required for mode='file' (the Analyze button on the Files page). The
    # cluster is derived from the file itself, not from `cluster` above.
    file_id: Optional[int] = None


class ScopeOptionOut(BaseModel):
    """One selectable run scope. Always a single cluster: see EventScope."""

    mode: str
    cluster: Optional[str]
    cluster_label: str
    label: str
    event_count: int
    # Only set for mode='file', where the file IS the scope; the other modes
    # resolve their own file (or window) at run time.
    file_id: Optional[int] = None
    # What Stage 1 is likely to cost over this scope, shown before the run
    # rather than only recorded after it. An extrapolation from a measured
    # tokens-per-event rate, not a quote — see pricing.estimate_stage1_cost_usd.
    estimated_cost_usd: float = 0.0
