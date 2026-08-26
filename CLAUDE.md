# CLAUDE.md

Guidance for working in this repo. See `README.md` for user-facing setup/run instructions.

The brief: a web-based agentic tool that ingests NetApp ONTAP EMS log files, uses an LLM
agent to investigate and surface storage-risk findings, and lets a user dismiss findings so
similar ones are suppressed later.

**This file states rules and decisions. The reasoning behind most of them lives in a comment
or docstring next to the code** — `app/` carries ~2,100 lines of it, and that is deliberately
where the archaeology sits rather than here, so this file stays cheap to load. When a bullet
below points at a file, that file explains itself; read it before changing the behaviour.

## Read these first

Seven things that break something silently — no exception, no failing test, just a wrong result:

- **Severity has two vocabularies and `app/severity.py` spans both.** An autosupport bundle
  writes `info` where ONTAP's catalog writes `informational`, and text logs carry syslog's
  `warning`/`critical`, which ONTAP's REST enum has no member for. A name missing from `_RANKS`
  is *kept* (the filter fails open on purpose), so the floor silently does less than the file
  row claims it did. In the other direction, a name that isn't in `ONTAP_SEVERITIES` must never
  reach a REST query — it's an invalid enum member that 400s the whole fetch, not a filter that
  matches nothing.

- **Target Python is 3.9.** Never use `X | None` union syntax in type hints; use
  `typing.Optional[X]`. It crashes pydantic/dataclasses at import time on 3.9, but works on a
  3.11 box — so a green local run proves nothing.
- **A cluster value is a host, not a label** (`CLUSTER_HOST_PATTERN` / `validate_host` in
  `app/ontap/client.py`, enforced in `ClusterFetchRequest`, the upload form field, and
  `cluster_state._require_host`). The same string becomes `f"https://{host}"` with the shared
  ONTAP credentials attached. You therefore cannot label an uploaded bundle
  `Building 4 lab cluster`, and IPv6 literals are rejected.
- **`pricing.py` is hardcoded to Sonnet 5** and not keyed off `CLAUDE_MODEL`. Change models
  and every cost in the UI is wrong *and* `STAGE2_COST_CAP_USD` mis-scales, because it is
  enforced against those same constants. Change the constants too.
- **Never attach a listener with a bare `getElementById(x).addEventListener(...)`** — use
  `api.js`'s `onSubmit(id, handler)`. A bare call throws when `x` is absent, aborting the whole
  script, so every listener registered *after* it silently never attaches and its form falls
  back to native browser submission. That once put a cluster password in the URL and access log.
- **A test connection must match `session.py`'s**: `check_same_thread=False` *and*
  `PRAGMA foreign_keys = ON` (SQLite defaults it OFF). A fixture missing either tests a
  database the app never creates — and the hallucinated-`event_id` crash passed cleanly against
  such a fixture while failing in production.
- **The fakes cannot catch request-shape errors.** They never validate a schema or reject a
  parameter; `maxItems` in `STAGE1_SCHEMA` passed every test and 400'd on the first real call.
  If you change the *shape* of a request (schema properties, `output_config`, new params),
  verify against the real API with a one-off call before trusting green tests.

## Environment

- Venv at `.venv/` (gitignored): `source .venv/bin/activate`, `pip install -e ".[dev]"`.
- `ANTHROPIC_API_KEY` in `.env` (copy `.env.example`). Only needed to trigger analysis
  (`POST /api/analysis/runs`, `POST /api/candidates/{id}/investigate`) — ingestion, parsing and
  the rest of the UI work without it.
- `CLAUDE_MODEL` (default `claude-sonnet-5`), `MAX_AGENT_ITERATIONS` (15), `STAGE1_EFFORT` /
  `STAGE2_EFFORT` (`medium` / `low`) are env-configurable. `STAGE2_COST_CAP_USD` (0.50) is
  **not** a Settings field — it's a constant in `app/agent/pricing.py`.
- **Live cluster access is optional and uses ONE credential pair for every cluster**
  (`ONTAP_USER`, `ONTAP_PASSWORD`, `ONTAP_VERIFY_TLS`, `ONTAP_TIMEOUT_SECONDS`) — a deliberate
  PoC simplification, rationale in `config.py`. Leave them unset and the app is unchanged; the
  `get_cluster_state` tool just reports itself unavailable and the agent works from logs alone.
- **There is deliberately no host setting.** Which cluster to query is *data*: every run is
  scoped to one cluster and that cluster value **is** the host. `graph.load_context` resolves
  it from the candidate's run; `cluster_state`'s functions take it as a required argument with
  no fallback, because a configured default would be used for every investigation regardless of
  where its evidence came from. See `config.py` and `cluster_state._require_host`.

## Architecture

Layered, roughly bottom-up. Each bullet is the rule; the file named explains why.

- `app/db/` — SQLite schema (`schema.sql`) + hand-written repo layer (`repo_*.py`), no ORM.
  `session.py` is the only place connections are created. Two ordering/config constraints, both
  documented there: `_migrate()` runs **before** `schema.sql`'s `executescript`, and connections
  open with `check_same_thread=False` (load-bearing for FastAPI's yield dependencies and for
  LangGraph's `ToolNode` worker thread). `database_path` / `ems_watch_dir` are read from
  settings **lazily at call time**, not captured as function defaults — `test_api.py`'s
  per-test isolation depends on it.
- `app/ingestion/watcher.py` — file discovery, `cluster_from_filename` (recovers the cluster
  from this app's own `ems_fetch_<cluster>_<timestamp>.log` naming, now a *fallback* behind
  `POST /api/files/upload`'s optional `cluster` form field), and `safe_upload_name`, which every
  upload must go through — an uploaded filename escapes the watch directory two ways, and the
  absolute-path one reads as safe. `routes_files` also asserts the resolved parent, so a
  regression in the helper is an error rather than a write. **Upload ingests immediately** and
  registers only the file it wrote, under the severity floor from its `min_severity` form
  field (see "Severity filtering" below).
- `app/parsing/` — pluggable format parsers (`formats/registry.py` tries each in turn), falling
  back to a low-confidence `unparsed.line` row rather than failing ingestion. Two formats, for
  the two ways EMS data reaches this tool:
  - `ems_text_format.py` — `<ISO time> [<node>: <event.name>:<severity>]: <msg>`, i.e. what
    `app/ontap/converter.py` writes from a live REST fetch.
  - `autosupport_format.py` — `EMS-LOG-FILE.txt` from an autosupport bundle. The *historical*
    path: a bundle exists for clusters this tool can never reach, over windows a live fetch no
    longer returns.

  Three constraints, all explained in `autosupport_format.py`: **the two regexes are
  deliberately disjoint** (which is what makes `PARSERS` order irrelevant — loosen either and
  an autosupport line parses *almost* correctly, with the process name in `event_name`);
  **autosupport timestamps carry no year**, so it is reconstructed by walking backwards from
  the file's mtime, with a documented failure mode when the file is over a year older than that
  reference; and **the emitting process is parsed and discarded on purpose**.

  **Timestamps are normalized to UTC at ingestion** (`ingestion_service._to_utc_iso`) and that
  is not cosmetic: SQLite has no date type, so every `event_time` comparison downstream is a
  *text* comparison, and lexicographic order over ISO strings is chronological only if they
  share one offset — which the two parsers don't. **Naive datetimes are left exactly as
  parsed.** Consequence: `event_time` is part of the ingestion dedup key, so rows ingested
  before this change won't match a re-ingest. Rebuild the database rather than reasoning about
  the mix.
- `app/ontap/` — the REST client (`client.py`) and its inverse, the JSON→text-line converter
  (`converter.py`) that `ems_text_format.py` parses. **These two stay in lockstep** — change
  one, check the other. (`autosupport_format.py` is read-only and outside that round trip.)
  **A fetch is a walk, not a request**: `count / page_size` sequential GETs over one keep-alive
  connection, so a 10,000-event fetch runs for minutes and ONTAP *will* close the socket
  underneath it. Three consequences, invisible at small counts and documented in place: a
  `Retry` policy restricted to GET; every `RequestException` wrapped in `OntapClientError`; and
  `routes_clusters` **keeps the events that already arrived** when a walk dies part-way, only
  erroring if it got nothing.
  `cluster_state.py` is the third piece — live "what is true right now" reads for Stage 2.
  **Two invariants any new area must obey**: every function returns an **already-summarized**
  structure, never a raw REST payload; and **nothing raises** — failures come back as
  `{"available": False, "reason": ...}`. Note summarizing is not the same as fetching one page:
  `_get_all` walks every page so counts are true, keeps partial results on a mid-walk failure,
  still raises on a first-page one, and guards against a repeating pagination link. Trimmed
  lists always ship their true count beside them (`matching_count`/`unhealthy_count` vs
  `listed`).
- `app/api/` + `app/services/` — FastAPI routes and thin service wrappers.
- `web/static/` — plain HTML/JS/CSS, no build step, no framework. `main.NoCacheStaticFiles`
  serves the frontend `no-cache` so HTML and JS can't drift apart; asset tags carry `?v=N`, bump
  only if stale assets are seen again. The palette is a **light, single-theme token block** at
  the top of `style.css` in NetApp's visual language — no dark mode, so a dark-tuned color below
  the token block is a bug rather than a variant. Read that block's header before touching the
  semantic colors; they are contrast-paired and the pairing is not free.

### Severity filtering

Every ingestion path applies a severity floor, defaulting to **notice and higher**
(`app/severity.py`: `DEFAULT_MIN_SEVERITY`) — the fetch form, the upload form,
`scripts/fetch_ems_events.py`, and `watcher.discover_new_files`, which carries the same default
although nobody chose it, since a discovery path that quietly kept everything would be the one
way to get unfiltered rows into an otherwise-filtered database.

- **It is a minimum, not a set.** Severity is an ordered scale, so a multi-select would permit
  a corpus of emergency-plus-debug with the middle missing, which no analysis can reason about.
  The UI dropdown offers four floors plus "all"; the API accepts any rankable name.
- **The floor is recorded on `files.severity_filter`, and that record is the whole reason the
  default is safe to have.** `repo_events.get_event_rate_baseline` computes an event's normal
  rate across every file a cluster ever contributed, not per run, so a database mixing filtered
  and unfiltered pulls averages two different populations. Pre-existing rows keep NULL, which
  correctly reads as unfiltered — **do not backfill them with today's default.**
- **Enforced twice on the fetch path, deliberately.** `routes_clusters` expands the floor into
  ONTAP's `a|b|c` OR-list so the cluster does the filtering (which is what makes it pay: `count`
  caps events *returned*, so a floor buys a wider window for the same budget), and `ingest_file`
  applies it again locally from the file row. The duplication makes `severity_filter` a
  guarantee about the rows rather than a record of what was requested.
- **Unrankable severities are kept** (`meets_minimum` fails open): an `unparsed.line` row has no
  severity, and dropping it would discard exactly what a human most needs to see, under the
  heading of removing noise.
- Filtering happens on `ParsedEvent`s *before* `_to_event_dicts`, so `sequence_num` is dense
  over the survivors — compaction orders by `(event_time, sequence_num)`.
- **The OR-list syntax is unverified against a live cluster.** Per the fakes rule above, the
  ontap fakes assert only that the string reaches `params`. A rejected value surfaces as an
  `OntapClientError` carrying ONTAP's body — but note `fetch_ems_events` logs a first-request
  400 as "cluster rejected order_by", which would misattribute it.

### Run scoping (`EventScope`)

Every analysis run covers **one cluster over one window**, resolved by
`analysis_service.resolve_scope` into an `EventScope` (`app/db/models.py`), stored on the
`analysis_runs` row at start and rebuilt by the background task via
`repo_analysis_runs.get_scope_for_run` — the task receives only a `run_id`, so the row is how
the scope reaches it. Three modes and deliberately only three — `recent_pull`, `file`, and
`last_24h`, the last **anchored to the newest event in the data, not wall-clock now** (this
tool is normally pointed at logs collected earlier). There is deliberately **no all-clusters
mode**: node names are not unique across clusters, so correlating them fabricates patterns.
Re-run refusal keys on `(scope_mode, scope_cluster, scope_file_id)`, not a global event count.

## Two-stage analysis pipeline (`app/agent/`)

Split into an autonomous broad scan and a human-gated deep investigation, not one continuous
loop. This was driven by a concrete scale target — ~10,000 events/day, under 10 minutes, under
$1 — that a single open-ended loop over the whole backlog couldn't meet, on cost *or* on
correctness (it was structurally blind to most events at that volume). Measured on real cluster
data, Stage 1 lands at **~$1.25 per 10,000 events**: slightly over the line, accepted for a PoC.

- **`compaction.py`** — deduplicates the event set into one row per `(node, event_name,
  severity)` run, tracking adjacency **per `(cluster, node)` independently**. Never merges
  different event types or nodes; that judgment is left to the model. **How much it saves is
  entirely data-dependent** — measured rates and the two slimmings that were considered and
  declined are in `render_compact_corpus`'s docstring.
- **`stage1.py`** — one Sonnet 5 structured-output call (raw `anthropic` SDK, not LangChain)
  sees the *entire* compact corpus and produces a ranked candidate list. Autonomous, triggered
  by "Run analysis". A candidate matching dismissed feedback or an open finding is inserted as
  `auto_suppressed` rather than `pending`, spending no investigation budget on it. Every failure
  after the model call raises `Stage1Error`, which carries the usage so the run records what it
  actually cost.
  **Cost is linear in corpus size and the corpus is the whole bill** — output is capped at
  `STAGE1_MAX_TOKENS = 16000` (~$0.16 at most, and it doesn't grow with event count) and the
  fixed terms are bounded by event-name *variety*, so past a small floor input is everything. Measured: **~$0.12 per 1k events, ~$1.25 per 10k, ~$2.50
  per 20k**. Re-derive per-event cost from `analysis_runs.input_tokens / events_considered`
  rather than trusting those on a corpus that dedups differently. Three constraints found only
  against the real API, each documented at its line: no `cache_control` (the cache is
  structurally write-only here); `maxItems` is rejected in structured-output schemas; and Sonnet
  5 thinks by default, so find the `type == "text"` block explicitly and check for
  `stop_reason == "max_tokens"`.
- **`candidates` / `repo_candidates.py`** — persists Stage 1's ranked output (`pending` /
  `investigating` / `investigated` / `discarded` / `auto_suppressed`). A human reviews it and
  either discards (no LLM call) or triggers an investigation; **no auto-escalation**. Alongside
  the model's `refs` it stores `leads` — those refs **resolved to concrete event ids at Stage 1
  time**, because refs are positional and re-deriving them later silently points at the wrong
  events (see `compaction.resolve_leads`).
- **`graph.py`** — the LangGraph investigate ⇄ tools loop, running against **one candidate at a
  time**, seeded with that candidate's category/rationale/stored leads. Terminates on an
  explicit `conclude_investigation()`, `max_agent_iterations` (backup bound), or
  `STAGE2_COST_CAP_USD` (primary bound). The cap is checked in `route_after_tools` only — the
  edge back into a *model* call — so the cap-crossing turn's own tool calls still execute and
  the final cost is `cap + that last call`, not a ceiling on the nose.
- **Deliberately NOT given to Stage 2: the full compact corpus.** Its size scales with total
  event count, not one candidate's scope, and it would be baked into the very first model call
  before any cap check runs — at 10k+ events that alone could exceed `STAGE2_COST_CAP_USD`
  (a real bug during development). Stage 2 gets a fixed-size scope summary plus the candidate's
  resolved leads, and uses its tools for the rest.
- **Every Stage 2 read is scoped to the investigation's own cluster** — the evidence tools via
  `InvestigationContext.event_cluster()`, and the scope summary in the system prompt via the
  run's own `EventScope`. Node names are not unique across clusters, so unscoped, `get_event_
  context` answers "what else happened on this node" with another cluster's events and the
  prompt names nodes the tools can never return a row for. `cluster=None` is the *unspecified*
  pseudo-cluster and must match itself via `IS NULL`, which is why `repo_events.ANY_CLUSTER` is
  a sentinel rather than `None`. `InvestigationContext.scoped` is False only for a run predating
  scoping, which analyzed everything.
- **Tool results are clamped server-side** (`tools.py`: `MAX_TOOL_RESULT_EVENTS = 100`,
  `MAX_CONTEXT_WINDOW_MINUTES = 240`). The cap is only evaluated *after* a model call, so
  whatever a tool returns reaches the model once at full price and then rides along in
  `messages` for every remaining turn — an unbounded tool result is a hole straight through
  `STAGE2_COST_CAP_USD`. The model may ask for more; it does not get more. Same reasoning bounds
  `repo_catalog`'s lookups and `repo_findings.MAX_EVIDENCE_IDS`.
- **Grounding: the evidence surface is wider than the event table.** `lookup_event_definition`
  reads the **ONTAP EMS message catalog** — 8065 product-shipped definitions with NetApp's own
  corrective actions, committed as `data/ems_catalog.json.gz`, which is what makes the tool
  useful to someone who clones it with no cluster (lookups are case-insensitive on purpose).
  `get_event_rate_baseline` turns "this fired 40 times" into "and the normal rate is 0.5/day";
  its denominator is the **corpus-wide** day span deliberately. `get_cluster_state` is one tool
  with an `area` enum rather than four tools, because four more schemas would ride in every
  turn's request — which also makes a *new* area cheap.
- **Live performance telemetry is deliberately out of scope, and the prompts must not promise
  it.** No latency, IOPS or throughput anywhere; `get_aggregate_capacity` is space consumed, not
  utilization. `performance_issue` survives as a Stage 1 *triage* category because EMS carries
  perf-shaped events and labelling them is honest — but both prompts now say plainly that the
  category is judged from events alone. Removing the category would be worse: those events would
  land in `availability_risk`, and `is_suppressed` keys on category. Closing the gap for real
  means a time-series backend, and NetApp Harvest's MCP server already does it — don't build a
  thin imitation here. Unimplemented `get_cluster_state` areas are listed in
  `cluster_state.py`'s module docstring as scope, not oversight.
- **Stage 1 gets two grounding blocks** (`prompts.render_glossary` / `render_dismissals`), both
  bounded by something that *isn't* event count: a glossary of only the event names in the
  corpus, and recent dismissals **with the human's stated reason**. The signature tables already
  block an exact repeat, so the reason is not deduplication — it lets the model generalize from
  what a human rejected instead of re-proposing near-misses that dodge a signature match. Full
  descriptions and corrective actions are a Stage 2 tool call away, priced under the cost cap
  rather than paid for in the most expensive call in the system.
- **`agent_steps` / `repo_agent_steps.py`** — one row per tool call, written by the `traced`
  wrapper, so an investigation's reasoning survives the graph returning. **Every Stage 2 result
  lives on the Findings page** under one heading, with its cost and a back-link to the run; the
  Analysis page shows only decisions and spend and links out. An audit trail belongs next to the
  consequential action (dismissal), not on a page nobody revisits. `record_step` **never
  raises**: a lost trace row is cheap, a lost investigation is money.
- **`build_tools` serializes every tool call behind a `threading.Lock`, and that lock is not
  optional.** `ToolNode` dispatches a turn's tool calls to a threadpool, and
  `check_same_thread=False` silences Python's check without making the connection thread-safe —
  concurrent use takes the process down with `SIGSEGV`, no traceback, nothing catchable.
  Reproduced 10/10 without it (`test_concurrent_tool_calls_do_not_corrupt_the_connection`).

### What Stage 2 leaves behind

Four outcomes, and which row each writes is load-bearing — full rules in `findings.py`'s status
constants and `graph.persist_findings`:

- **A finalized hypothesis** → `status='open'`.
- **A refutation** → `status='no_issue'`, `severity='info'`, carrying the agent's own summary,
  because ruling a candidate out is a real result rather than an absence of one. **Not** an open
  finding (a pattern ruled out today can escalate next week, and open findings block creation),
  and **not dismissible** — agreeing something isn't a problem is a different signal from a human
  overriding the agent. Gated on the agent's stated `outcome`, **not** on "concluded and
  finalized nothing": that distinction was a real bug in which a `outcome='confirmed'`
  conclusion got filed under the title "No issue found".
- **An unfinished hypothesis** → `status='partial'`, on a budget bound *or* a
  confirmed-but-unfinalized conclusion. The run was billed in full, so what it worked out
  survives. Not an open finding, for the same escalation reason — but **dismissible, unlike
  `no_issue`**, because a partial is a claimed risk nobody finished checking.
- **Nothing**, when the model stopped talking without concluding or recorded nothing at all. The
  UI says "ended without recording a result", deliberately unattributed: the same empty state
  covers a budget bound, a silent model, and an outright failure.

Two invariants across all four: **suppression is checked in three independent places** — Stage
1's pre-filter, the agent's own `check_suppression` tool call, and defensively again in
`persist_hypothesis` — so a model that skips the check can't bypass it. And **model-authored
values are validated at the tool boundary**: off-enum categories get a corrective `ToolMessage`
rather than an exception (a signature is computed *from* the category, so an off-enum one can
never be suppressed or matched), and hallucinated or over-long evidence id lists are dropped
rather than fatal — see `repo_findings.insert_evidence` for the three compounding costs that
one `IntegrityError` used to cause.

### `pricing.py`

Sonnet 5 constants and `estimate_cost_usd(...)`, used by both stages' cost display and Stage 2's
cap check. **Cache tokens are priced separately from plain input** (1.25x writes, 0.1x reads).
Three things to know:

- **The two SDKs report cache tokens in opposite conventions.** The raw `anthropic` SDK
  (Stage 1) *excludes* cached tokens from `usage.input_tokens`; `langchain_anthropic` (Stage 2)
  blends them in and `graph.py`'s `investigate` node unpacks them back out. Get it wrong and
  cost is undercounted 10-40x, or the cap fires early. Know which convention you are in.
- **Wrong constants are invisible and cost you twice.** These shipped at $3/$15 for a while — a
  scheduled increase that was cancelled — so every cost read 1.5x high *and* every investigation
  was cut off at two thirds of its nominal budget. Now pinned by `tests/test_agent_pricing.py`.
  **Verify against platform.claude.com/docs/en/about-claude/pricing, not from memory.**
- `estimate_stage1_cost_usd(event_count)` is the *forward* estimate shown beside the scope
  selector, since Stage 1 is one uncapped call and an autosupport bundle can be an order of
  magnitude bigger than a fetch. **It is a warning, not a quote** — up to 4x high on repetitive
  data, which is the safe direction, and the UI renders two decimals for that reason.

## Testing

- `pytest` — 265 tests, all fully offline via scripted fakes; no API key or cluster needed. Test
  names are written as sentences, so `pytest --collect-only -q` is the index. Note
  `test_agent_stage1.py` and `test_agent_graph.py` use **different** fake shapes, matching each
  stage's real SDK.
- `test_parsing.py`'s autosupport half is mostly about the reconstructed year. The reference
  time is **always passed explicitly** — a wall-clock default makes a test that passes all year
  fail on New Year's Eve — and the cases that matter are the ones a naive implementation gets
  wrong: a December file read in January, a file spanning Dec 31→Jan 1, out-of-order jitter that
  must *not* roll the year, Feb 29, and a space-padded day (`Aug  5`).
- UI changes were verified with Playwright driving system-installed Chrome (no
  `chromium-cli`/pre-installed browser in this environment). A manual step during development,
  not wired into the suite.

## Validated against real systems

- **Stage 1 against the real API**: on the 9-event `data/sample.log` it identified a genuine
  cross-node HA-failover correlation (two event types on both HA-partner nodes within 3
  seconds). Cost tracking checked against real usage objects, not fakes.
- **Stage 1 at production volume** (2026-08-19): 1,000 events ~$0.12, 10,000 ~$1.25 — confirming
  cost is linear in event count and that the per-event rate is ~4x the best-case compaction
  figure on data that doesn't dedup. Both completed; neither hit `STAGE1_MAX_TOKENS`.
- **Both stages' request shapes confirmed against the real API**, including Stage 2's
  `output_config` reaching it through `model_kwargs` and the `cache_control` breakpoint landing.
  That check is also the evidence behind the pricing rules above: it re-confirmed langchain's
  blended-`input_tokens` convention against a real usage object and that only
  `ephemeral_5m_input_tokens` is ever populated.
- **The EMS catalog resolves against real data**: pulled from a live cluster; 6 of 6 genuine
  event names in `data/sample.log` match. The two misses in `samples/sample_ems_events.log` are
  invented names in that synthetic file.
- **The autosupport format, end-to-end against a sanitized real bundle**
  (`samples/sample_autosupport.log`, 2026-08-25): uploaded as `EMS-LOG-FILE.txt` with an
  explicit cluster, 9/9 events ingest at high confidence and **7 of 7** distinct event names
  resolve against the catalog — which says the regex produces real ONTAP event names rather than
  plausible-looking substrings. It is only 9 lines, so it does not exercise a header banner,
  wrapped lines, or the multi-megabyte sizes a real bundle reaches.
- `app/ontap/` and the full Stage 2 loop have been exercised against a real ONTAP cluster and
  the real API at proof-of-concept depth. Enough to trust the happy paths; not exhaustive
  coverage of every ONTAP version, response shape, or failure mode — which is why the code
  degrades rather than raises.
