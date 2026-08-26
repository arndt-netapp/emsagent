from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FileRecord:
    id: int
    filename: str
    cluster: Optional[str]
    filepath: str
    file_hash: str
    file_size_bytes: Optional[int]
    detected_format: Optional[str]
    status: str
    error_message: Optional[str]
    discovered_at: str
    ingested_at: Optional[str]
    event_count: int
    duplicates_skipped: int = 0
    # Severity floor applied at ingestion; None means every severity was kept.
    severity_filter: Optional[str] = None
    severity_skipped: int = 0


@dataclass
class EventRecord:
    id: int
    file_id: int
    cluster: Optional[str]
    raw_line: str
    event_time: Optional[str]
    node: Optional[str]
    event_name: str
    severity: Optional[str]
    message: Optional[str]
    sequence_num: int
    parse_confidence: str
    created_at: str


@dataclass
class Finding:
    id: int
    analysis_run_id: Optional[int]
    candidate_id: Optional[int]
    category: str
    severity: str
    title: str
    description: str
    recommendation: Optional[str]
    node: Optional[str]
    signature: str
    pattern_signature: str
    status: str
    confidence: Optional[float]
    created_at: str
    dismissed_at: Optional[str]


@dataclass
class Candidate:
    id: int
    analysis_run_id: int
    rank: int
    category: str
    node: Optional[str]
    rationale: str
    confidence: Optional[float]
    refs: List[int]
    # Stage 1's refs resolved to durable event ids; empty for candidates
    # created before the column existed (see repo_candidates._row_to_candidate).
    leads: List[Dict[str, Any]]
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


@dataclass
class FeedbackRecord:
    id: int
    finding_id: int
    signature: str
    pattern_signature: str
    scope: str
    reason: Optional[str]
    created_at: str


@dataclass
class EventScope:
    """Which events one analysis run covers.

    Every run is scoped to exactly ONE cluster. There is deliberately no
    "all clusters" mode: correlating events across unrelated clusters is
    meaningless, and because ONTAP names nodes <cluster>-01/-02 (the simulator
    defaults to "cluster1"), two clusters can present identical node names that
    compaction would silently merge into single runs.

    `cluster is None` is not "any cluster" — it is the *unspecified* pseudo-
    cluster holding events from log files dropped in the watch directory, which
    carry no cluster identity in their text format.
    """

    mode: str  # "recent_pull" | "last_24h"
    cluster: Optional[str]
    label: str
    file_id: Optional[int] = None
    since: Optional[str] = None
