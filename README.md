# ONTAP EMS Log Agent

A web-based agentic AI tool for analyzing NetApp ONTAP EMS (Event Management System) log
events. Events are pulled from a cluster's REST API (or supplied as log files), parsed into
structured records, and investigated by a Claude-powered agent that surfaces findings
(availability risks, performance issues, predictive failures). A web UI lets you trigger
ingestion/analysis, review findings with supporting evidence, and dismiss findings you don't
care about — dismissals are remembered and suppress similar findings in future runs.

Analysis runs in two stages, so large event volumes don't mean unbounded token cost:

1. **Broad scan** (autonomous) — one Claude Sonnet 5 call sees every event in scope, deduplicated
   into a token-efficient form, and returns a ranked list of candidate patterns — including
   correlations across nodes or event types that a hand-written rule wouldn't catch.
2. **Investigation** (human-gated) — you pick which candidates are worth a real iterative
   LangGraph investigation; discarding one costs nothing. Each investigation is capped at $0.50,
   and every tool call it makes is shown in the UI as an expandable trace.

## How it works

The whole pipeline, end to end. Everything above the HUMAN GATE runs on its own and is
cheap; everything below it costs real money and therefore only runs when a person asks
for it, on one candidate at a time.

```
┌─ INPUT ────────────────────────────────────────────────────────────────────┐
│ cluster fetch (REST)   ·   log-file upload   ·   seeded events             │
│ parsed into `events`, de-duplicated against what is already stored,        │
│ and tagged with the cluster it came from                                   │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │  EventScope: ONE cluster, ONE window
                                     ↓
┌─ COMPACTION   (no model — pure Python) ────────────────────────────────────┐
│ one row per run of identical (node, event_name, severity) events:          │
│     ref|first_time|last_time|node|event_name|severity|count                │
│ ~14 tokens/event instead of ~48 — this is what lets the whole corpus       │
│ fit in a single call at 10k events/day                                     │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     ↓
┌─ STAGE 1   ·   broad scan   ·   autonomous   ·   ONE model call ───────────┐
│ sees the ENTIRE compact corpus, plus two grounding blocks bounded by       │
│ something other than event count:                                          │
│   · glossary — catalog definitions for the event names in THIS corpus      │
│   · dismissals — what a human rejected before, and the reason given        │
│ returns a ranked candidate list (structured output). No tools, no loop.    │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ↓  candidates: rank, category, node, rationale, leads
┌─ PRE-FILTER   (no model) ──────────────────────────────────────────────────┐
│ a candidate matching a dismissal or an open finding is stored as           │
│ `auto_suppressed` — no human review, no investigation spend                │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     ↓
┌─ HUMAN GATE   (Analysis page)  —  nothing escalates on its own ────────────┐
│ Discard  →  0 tokens, no model call         Investigate  →  Stage 2        │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     ↓  ONE candidate + its leads + a fixed-size scope summary
┌─ STAGE 2   ·   deep investigation   ·   human-gated   ·   LangGraph loop ──┐
│                                                                            │
│    ┌──────────────┐   tool calls    ┌────────────────────────────────────┐ │
│    │              │ ──────────────→ │ EVIDENCE                           │ │
│    │  investigate │                 │   query_events                     │ │
│    │   (model)    │                 │   get_event_context                │ │
│    │              │                 │   search_events_by_name_pattern    │ │
│    │              │ ←────────────── │   lookup_event_definition (catalog)│ │
│    └──────────────┘    results      │   get_event_rate_baseline          │ │
│                                     │   get_cluster_state (live cluster) │ │
│                                     │ DECIDE                             │ │
│                                     │   check_suppression                │ │
│                                     │   record_hypothesis                │ │
│                                     │   conclude_investigation           │ │
│                                     └────────────────┬───────────────────┘ │
│                                        │  every call is traced to          │
│                                        ↓  agent_steps → the UI             │
│                                                                            │
│    back to the model ONLY if all three hold:                               │
│      · conclude_investigation not called                                   │
│      · cost so far < $0.50            (primary bound)                      │
│      · turns < MAX_AGENT_ITERATIONS   (backup bound)                       │
│                                                                            │
│    the cap-crossing turn's own tool calls still run — already paid for,    │
│    and running them is free local DB work                                  │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     ↓  whatever the loop reached — every outcome is recorded
┌─ OUTCOMES   (a findings row, always with its trace and its cost) ──────────┐
│ status=open       confirmed risk: severity, evidence, recommendation       │
│ status=no_issue   explicitly ruled out, with the reason that ruled it      │
│                   out — a result, not an absence of one                    │
│ status=partial    the budget ran out first: the agent's working            │
│                   hypothesis, kept and labelled UNCONFIRMED                │
│                                                                            │
│ only the first is an *open* finding, so a pattern ruled out or             │
│ half-checked today can still be raised properly tomorrow                   │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     ↓  human dismisses an open or partial finding
┌─ FEEDBACK   —   suppression, checked in three independent places ──────────┐
│ 1. Stage 1's pre-filter                 (before any spend)                 │
│ 2. the agent's check_suppression tool   (before it finalizes)              │
│ 3. findings.persist_hypothesis          (defensively, at write time)       │
│                                                                            │
│ the human's stated REASON also goes into Stage 1's prompt, so the next     │
│ scan can generalize instead of re-proposing near-misses                    │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     └──→  folded into the NEXT Stage 1 scan
```

## Status

**This is a proof of concept, not a production tool.** It exists to demonstrate an agentic
workflow over real NetApp ONTAP EMS data, and it is deliberately simple in places a production
system would not be: one credential pair shared across all clusters, cost estimates hardcoded to
a single model, and **no authentication on the web UI or the API**. Run it locally or on a
trusted network — anyone who can reach the port can trigger analysis runs that spend money, and
can read every ingested event.

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` and set `ANTHROPIC_API_KEY` before running an analysis (everything else —
ingestion, parsing, the file/findings UI — works without it).

### Optional: let the agent read live cluster state

Set `ONTAP_USER` and `ONTAP_PASSWORD` in `.env` and the investigation agent can check what's true
on the cluster *right now* — aggregate capacity, disk health, HA takeover status, volume state —
to corroborate or refute what the logs suggest. Leave them unset and it works from logs alone.
There's no host setting: every run is scoped to one cluster, and that cluster *is* the host its
events came from. One credential pair is used for every cluster (a PoC simplification), and it
has to live in config because investigations run as background tasks with nobody to prompt.

Those four areas are the whole of it — there is **no performance telemetry**. No latency, IOPS or
throughput, and "aggregate capacity" means space consumed, not how busy an aggregate is. A
`performance_issue` finding therefore describes what the events reported; it is not a
measurement, and the agent is told to say so. Answering "is this actually slow right now?" needs
historical trending over a metrics backend — a separate system, not a bigger version of this one.
[NetApp Harvest's MCP server](https://netapp.github.io/harvest/nightly/mcp/overview/) does it
against Prometheus/VictoriaMetrics.

## Running the app

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/ for the Files page (`/analysis.html`, `/findings.html` for the
other two views). The API is also browsable at http://127.0.0.1:8000/docs.

## Getting EMS events into the app

Three ways in. All of them apply the severity floor described below.

![The Files page: the upload form, the cluster fetch form, and the table of ingested files
showing cluster, detected format, severity floor and event count](docs/screenshots/emsagent1.png)

1. **Pull from a live cluster** via the "Fetch recent logs from a cluster" form on the Files
   page, or the standalone script:
   ```bash
   python -m scripts.fetch_ems_events --cluster 10.0.0.5 --user admin --count 500
   ```
   Prompts for a password (or set `EMS_CLUSTER_PASSWORD`) — credentials are never persisted.
   Uses `GET /api/support/ems/events`; TLS verification is off by default since most clusters use
   self-signed certs (`--verify-tls` to enable it).

   Big fetches are paginated at 500 records per request, so 10,000 events is a walk of ~20
   sequential requests and takes a while. Transient drops are retried; if the walk still dies
   part-way, the events already retrieved are kept and the shortfall is shown on the file's row
   rather than passed off as "the cluster only had that many".
2. **Upload a log file** on the Files page. Two formats are detected automatically, and either is
   parsed and ingested immediately:

   ```
   2026-08-10T00:00:12-04:00 [node1: callhome.reboot:notice]: System rebooted.
   Tue Aug 25 00:08:22 -0700 [cluster2-n03: dense_ads_monitor: sis.auto.session.change:notice]: ADS: ...
   ```

   The first is what a fetch produces (`samples/sample_ems_events.log`) — mainly for logs
   collected elsewhere, e.g. by running the script on a machine that can reach the cluster.

   The second is the `EMS-LOG-FILE.txt` inside an **autosupport bundle**, downloadable per-cluster
   from activeiq.netapp.com (`samples/sample_autosupport.log`). This is the way in for a cluster
   you can't reach over REST, or a window that has already aged out of one. Two things to know:

   - **Fill in the Cluster field.** An autosupport log names no cluster anywhere, so without it
     the events join the "unspecified" pool where a second bundle can't be told apart from the
     first. (A file still named `ems_fetch_<cluster>_<timestamp>.log` keeps its cluster
     automatically.)
   - **Its lines carry no year** — only `Tue Aug 25 00:08:22` — so it's inferred from the file's
     own timestamp, handling a rollover past Jan 1 correctly. A bundle left on disk more than a
     year before upload is dated a year late: order is preserved, absolute dates are shifted.

   Files dropped straight into `data/logs_incoming/` can be ingested with
   `python -m app.ingestion.cli ingest`.
3. **Seed synthetic events** to try the agent with no log source at all:
   ```bash
   python -m scripts.seed_events
   ```

### The severity floor

Both forms carry a "Minimum severity" dropdown defaulting to **notice and higher**, so
`informational` and `debug` events are dropped and never stored. Choose *All severities* to keep
everything; `--min-severity` does the same for the script. It's a minimum rather than a set of
checkboxes because severity is an ordered scale — "emergency plus debug, nothing in between"
isn't a corpus anything can reason about.

Filtering pays twice over: Stage 1's cost is linear in events seen (~$0.12 per 1,000), and on a
fetch `count` caps events *returned*, so a floor spends the same budget on a wider time window.
The floor is `notice` rather than `error` because a lot of real operational signal —
takeover/giveback, callhome — is notice-severity. Lines the parser *couldn't* read are always
kept, whatever the floor: they have no severity to rank, and they're what a human most needs
to see.

The floor is recorded on the file's row and shown in the Severity column (`notice+`, plus how
many events it dropped). That record matters: rate baselines are computed across every file a
cluster ever contributed, so a database mixing filtered and unfiltered pulls averages two
different populations, and the column is how you see that happened. Files ingested before this
existed show `all`, which is accurate. Widening the floor later re-ingests only what's missing,
though a byte-identical file is refused by hash — fetch again, or upload a copy.

## Running an analysis

1. **Pick a scope and click "Run analysis"** on the Analysis page (or `POST /api/analysis/runs`).
   Requires `ANTHROPIC_API_KEY`. This is the broad scan only, so it's fast however many
   candidates it finds.

   Every run covers **one cluster**, either its **most recent pull** or its **last 24 hours** —
   there is no all-clusters option, because ONTAP names nodes `<cluster>-01/-02` and correlating
   two clusters that share node names fabricates patterns. Files dropped in the watch directory
   have no cluster identity and group as "unspecified". The 24-hour window is measured from the
   newest event in the data, not the current time, so it works on logs collected weeks ago. The
   **Analyze** button beside any file jumps here with that file pre-selected.

   ![The Analysis page: the scope selector with its cost estimate, the run history row with
   token counts, and the ranked candidate list below it](docs/screenshots/emsagent2.png)

2. **Review the ranked candidates** and, for each, either **Discard** (no model call, no cost) or
   **Investigate** — a real iterative investigation of that one candidate, capped at $0.50.

The two pages divide cleanly. **Analysis** is about decisions and spend: what Stage 1 flagged,
what you chose to investigate, what it cost. **Findings** holds every Stage 2 result with its
supporting events, its cost, and an expandable trace of every tool the agent called.

![The Findings page: one open finding with its evidence summary and recommendation, the
expandable supporting-events and tool-call trace, and the dismissal
controls](docs/screenshots/emsagent3.png)

Three things about cost and outcomes are worth knowing before you spend anything:

- **The estimate under the scope selector is an upper bound, not a quote.** Stage 1 is one
  uncapped call priced by events in scope — roughly $0.12 per 1,000, $1.25 per 10,000 — and
  repetitive logs compact much further and cost several times less.
- **All costs shown are estimates at Claude Sonnet 5 list prices, hardcoded.** Change
  `CLAUDE_MODEL` without updating `app/agent/pricing.py` and both the displayed cost and the
  $0.50 cap (enforced against those same numbers) are wrong.
- **An investigation that runs out of budget still leaves something behind**: its working
  hypothesis, filed as `partial` and labelled UNCONFIRMED rather than as an open finding. Treat
  those as leads. If they're frequent, raise `STAGE2_COST_CAP_USD` in `app/agent/pricing.py`.

Ruling a candidate out is a result too, recorded as "no issue found" with the reason. Those
can't be dismissed — dismissal teaches the agent to suppress a pattern, and agreeing with it
isn't the same as overriding it — and neither they nor `partial` findings block the same pattern
from being raised properly later.

Re-running the scan on an unchanged scope is refused (409) unless a dismissal has been recorded
since, so repeat clicks don't burn tokens re-deriving the same list. The check is per-scope, so
one cluster never blocks a first run on another.

## Tests

```bash
pytest
```

All tests run offline — the analysis pipeline and ONTAP REST client are exercised against
mocked responses, not live services.

## Project layout

- `app/db/` — SQLite schema and data-access layer, including `repo_candidates.py` for the
  ranked-candidate list Stage 1 produces.
- `app/ingestion/`, `app/parsing/` — watch-directory ingestion and log parsing.
- `app/ontap/` — ONTAP REST API client and JSON→text-log converter.
- `app/agent/` — the two-stage analysis pipeline: `compaction.py` (event deduplication),
  `stage1.py` (the autonomous broad scan), `graph.py` (the human-gated, per-candidate
  LangGraph investigation loop), `tools.py`, `pricing.py` (cost tracking/cap), and suppression
  logic in `findings.py`.
- `data/ems_catalog.json.gz` — the ONTAP EMS message catalog: 8065 event definitions with
  descriptions and NetApp's own corrective actions (product data shipped with ONTAP, not customer
  data). Loaded into SQLite on first run, it's what lets the agent explain what an event means
  rather than guessing from its name — no cluster required.
- `app/api/` — FastAPI routes, including `routes_candidates.py` for investigate/discard.
- `web/static/` — plain HTML/JS/CSS frontend.
- `scripts/` — standalone CLIs (`fetch_ems_events.py`, `seed_events.py`).
- `samples/`, `data/sample.log` — sample logs for manual testing without a real cluster, one
  per supported format: `samples/sample_ems_events.log` (fetch output) and
  `samples/sample_autosupport.log` (a sanitized autosupport `EMS-LOG-FILE.txt`).

## License

MIT — see [LICENSE](LICENSE).
