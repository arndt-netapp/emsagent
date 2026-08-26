CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Which cluster this file's events came from. Set from the fetch form /
    -- CLI for cluster pulls; NULL for a log file someone dropped in the watch
    -- directory, since the text log format carries no cluster identity at all.
    cluster TEXT,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL UNIQUE,
    file_hash TEXT NOT NULL,
    file_size_bytes INTEGER,
    detected_format TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    discovered_at TEXT NOT NULL,
    ingested_at TEXT,
    event_count INTEGER DEFAULT 0,
    -- Events in this file that a previous ingestion had already stored. Kept
    -- visible rather than silently discarded: after a repeated cluster fetch
    -- this is usually most of the file, and a user who fetched 500 events and
    -- sees 12 ingested needs to know why.
    duplicates_skipped INTEGER DEFAULT 0,
    -- The severity floor this file's events were ingested under (a canonical
    -- name from app/severity.py), or NULL for "every severity". Recorded
    -- because it changes which rows exist, and nothing downstream could
    -- otherwise tell a quiet cluster from a filtered one: get_event_rate_
    -- baseline computes rates across every pull ever ingested for a cluster,
    -- so a database mixing filtered and unfiltered files reports rates over
    -- two different populations with no way to see it. Pre-existing rows keep
    -- NULL, which is accurate — they were ingested before any filter existed.
    severity_filter TEXT,
    -- Events parsed out of this file and then dropped by that floor. Visible
    -- for the same reason as duplicates_skipped: it is the only way to tell
    -- "this file held 40 events" from "this file held 4,000 and you asked for
    -- the loud ones".
    severity_skipped INTEGER DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id),
    -- Denormalized from files.cluster at ingestion, matching how `node` is
    -- already carried on the event row. Analysis is ALWAYS scoped to one
    -- cluster: without this column a run mixes clusters, and since ONTAP names
    -- nodes <cluster>-01/-02 (and the simulator defaults to "cluster1"), two
    -- clusters can present identical node names — which compaction would
    -- silently interleave into single runs with fabricated time spans.
    cluster TEXT,
    raw_line TEXT NOT NULL,
    event_time TEXT,
    node TEXT,
    event_name TEXT NOT NULL,
    severity TEXT,
    message TEXT,
    sequence_num INTEGER NOT NULL,
    parse_confidence TEXT NOT NULL DEFAULT 'high',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_file ON events(file_id);
CREATE INDEX IF NOT EXISTS idx_events_cluster ON events(cluster);
CREATE INDEX IF NOT EXISTS idx_events_cluster_time ON events(cluster, event_time);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time);
CREATE INDEX IF NOT EXISTS idx_events_node ON events(node);
CREATE INDEX IF NOT EXISTS idx_events_name ON events(event_name);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    events_considered INTEGER DEFAULT 0,
    iterations INTEGER DEFAULT 0,
    candidates_generated INTEGER DEFAULT 0,
    candidates_auto_suppressed INTEGER DEFAULT 0,
    error_message TEXT,
    -- The requested scope, recorded at run START so the background task can
    -- reconstruct exactly which events to analyze, and so re-run refusal can
    -- compare a run against the last run over the SAME scope. scope_json below
    -- is different: it's the resulting summary, written at completion.
    scope_mode TEXT,
    scope_cluster TEXT,
    scope_file_id INTEGER,
    scope_since TEXT,
    scope_label TEXT,
    scope_json TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_creation_input_tokens INTEGER DEFAULT 0,
    cache_read_input_tokens INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(id),
    rank INTEGER NOT NULL,
    category TEXT NOT NULL,
    node TEXT,
    rationale TEXT NOT NULL,
    confidence REAL,
    refs TEXT NOT NULL,
    -- Stage 1's refs resolved to concrete event ids at generation time. refs
    -- are positional against the corpus that produced them, so re-deriving
    -- them at investigation time resolves to the wrong events once anything
    -- is ingested out of order; these are durable. NULL for rows created
    -- before this column existed.
    leads TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    discard_reason TEXT,
    investigation_input_tokens INTEGER,
    investigation_output_tokens INTEGER,
    investigation_cache_creation_input_tokens INTEGER,
    investigation_cache_read_input_tokens INTEGER,
    investigation_iterations INTEGER,
    investigation_started_at TEXT,
    investigation_completed_at TEXT,
    investigation_error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidates_run_rank ON candidates(analysis_run_id, rank);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);

CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_run_id INTEGER REFERENCES analysis_runs(id),
    candidate_id INTEGER REFERENCES candidates(id),
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    recommendation TEXT,
    node TEXT,
    signature TEXT NOT NULL,
    pattern_signature TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    confidence REAL,
    created_at TEXT NOT NULL,
    dismissed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_findings_signature ON findings(signature);
CREATE INDEX IF NOT EXISTS idx_findings_pattern ON findings(pattern_signature);
CREATE INDEX IF NOT EXISTS idx_findings_status ON findings(status);
CREATE INDEX IF NOT EXISTS idx_findings_candidate ON findings(candidate_id);

CREATE TABLE IF NOT EXISTS finding_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_id INTEGER NOT NULL REFERENCES findings(id),
    event_id INTEGER NOT NULL REFERENCES events(id),
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_finding ON finding_evidence(finding_id);

-- The ONTAP EMS message catalog: the static, product-shipped definition of
-- every event ONTAP can emit. Not customer data — loaded from the committed
-- data/ems_catalog.json.gz fixture, so the agent has grounded event
-- definitions even with no cluster attached. Keyed by the same dotted name
-- that appears in events.event_name.
CREATE TABLE IF NOT EXISTS ems_catalog (
    name TEXT PRIMARY KEY,
    severity TEXT,
    description TEXT,
    corrective_action TEXT,
    snmp_trap_type TEXT,
    deprecated INTEGER DEFAULT 0
);
-- Catalog names are matched case-insensitively against event_name: ONTAP's
-- catalog casing ("AccessCache.NearLimits") does not always match what shows
-- up in a log line, so lookups normalize both sides via this index.
CREATE INDEX IF NOT EXISTS idx_catalog_name_nocase ON ems_catalog(name COLLATE NOCASE);

-- One row per tool call the Stage 2 agent makes, so an investigation's
-- reasoning is auditable after the fact instead of vanishing with the
-- in-memory message list. Written by graph.py's tool-tracing wrapper.
CREATE TABLE IF NOT EXISTS agent_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id),
    iteration INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    tool_args TEXT,
    result_summary TEXT,
    error TEXT,
    duration_ms INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_steps_candidate ON agent_steps(candidate_id, id);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_id INTEGER NOT NULL REFERENCES findings(id),
    signature TEXT NOT NULL,
    pattern_signature TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'node',
    reason TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_signature ON feedback(signature);
CREATE INDEX IF NOT EXISTS idx_feedback_pattern ON feedback(pattern_signature);
