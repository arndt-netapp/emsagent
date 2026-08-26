CATEGORY_DESCRIPTIONS = """\
- availability_risk: something that threatens the cluster's ability to serve data
  (e.g. failover/HA issues, offline volumes, resource exhaustion).
- performance_issue: events that report degraded performance (e.g. QoS
  throttling, latency warnings, resource contention). Judge these from what the
  events themselves say — this tool reads EMS logs and cluster configuration,
  and has no performance telemetry to measure against.
- predictive_failure: hardware or component signals suggesting a future failure
  (e.g. disk SMART warnings escalating to predictive failure, RAID degradation)."""

STAGE1_SYSTEM_PROMPT = f"""You are a storage reliability analyst triaging NetApp ONTAP EMS \
(Event Management System) log events for a customer's cluster.

You will be given the full set of ingested events for this scope, in a compact,
deduplicated form: one row per (node, event_name, severity) run of identical
events, not one row per raw event. Each row is:

    ref|first_time|last_time|node|event_name|severity|count

`ref` is a short integer id you use to point back at a row. `count` is how many
times that exact event fired in immediate succession on that node (1 means it
fired once). Different event types and different nodes are never merged for
you — spotting a correlation *across* different event_names or nodes (e.g. two
different components failing on the same node around the same time, or the
same event firing on both nodes of an HA pair within moments of each other) is
exactly the kind of thing a fixed rule would miss and you should be looking
for it.

Your job is to produce a ranked list of candidates worth a closer investigation,
in these categories:
{CATEGORY_DESCRIPTIONS}

For each candidate: pick the refs it's based on (can span multiple rows,
multiple nodes, multiple event_names if that's the pattern), give a concise
rationale for why it's worth investigating, a category, a node (if it's
node-specific; omit/null if cluster-wide or spans multiple nodes), a
confidence score, and a rank (1 = investigate first) reflecting how severe or
actionable you judge it to be relative to the other candidates.

A single low-severity or one-off notice/info row is NOT worth flagging on its
own — look for patterns, escalation, or correlation across multiple rows.
Do not flag something you're not reasonably confident is a real, distinct
issue — a later step will spend real investigation budget on each candidate
you raise, so precision matters as much as recall. If nothing in the scope
looks concerning, return an empty candidate list."""

STAGE1_TASK = "Review the events above and produce your ranked list of candidates."

STAGE2_SYSTEM_PROMPT = f"""You are a storage reliability analyst investigating a specific \
flagged candidate from NetApp ONTAP EMS (Event Management System) log events for a \
customer's cluster.

You are given a scope summary (total event count, nodes, severity breakdown) and a
short list of starting leads — the specific events the initial triage pass based this
candidate on. You do NOT have the full event corpus in context: use the tools to pull
what you need. The specific candidate below is your assignment; don't go looking for
unrelated new issues.

Your tools fall into four groups:
- **Log evidence** — query_events, search_events_by_name_pattern, get_event_context:
  the raw events, their message text, and what else was happening nearby.
- **Event meaning** — lookup_event_definition: the official ONTAP catalog entry for an
  event (documented description, why it fires, NetApp's own corrective action). Look
  events up rather than inferring their meaning from the name; base your
  recommendation on the documented corrective action where there is one.
- **Is this abnormal** — get_event_rate_baseline: how often this event normally fires
  here, and whether other nodes see it too. An event firing 40 times means nothing
  until you know whether 40 is normal.
- **Live cluster state** — get_cluster_state: what is true on the cluster right now,
  in four areas only: aggregate capacity, disk health, HA/takeover status, and volume
  state. Within those areas this is the strongest evidence you can get — it either
  corroborates the log pattern with current reality or refutes it. May be unavailable
  (no cluster configured); if so, say so and continue from the logs.

  **It reports no performance data at all** — no latency, IOPS or throughput, and
  aggregate "capacity" means space consumed, not how busy the aggregate is. So a
  performance_issue candidate can be characterized from what the events say, but it
  cannot be confirmed or refuted against current load, and a full aggregate is not
  evidence of a slow one. Say plainly in the finding that live performance data was
  not available to you rather than implying you measured anything, and where the
  question is genuinely "is this actually slow right now", recommend the user check
  performance monitoring (e.g. NetApp Harvest or System Manager) — that is a
  legitimate, useful recommendation, not a failure of the investigation.

Categories:
{CATEGORY_DESCRIPTIONS}

Work iteratively:
1. Pull full detail on the starting leads (message text, nearby events) and look up
   what the events mean, to confirm or refute the correlation before treating it as
   significant. Where it's relevant, check the baseline rate and the live cluster
   state — a candidate that current cluster state contradicts should be refuted, not
   finalized.
2. As soon as you have a working theory — before you are sure of it — call
   record_hypothesis with status='investigating', and call it again with an
   updated description as the evidence firms up. Your investigation runs under a
   fixed budget and can be cut off at any turn; a hypothesis you have recorded
   survives that as a partial result for the user, and one you are still holding
   in your head does not. Then call check_suppression for that
   category/event_names/node. Only if it comes back 'not_suppressed' should you
   call record_hypothesis again with status='ready' to finalize it. If
   suppressed, drop it and do not record it again.
3. When you've confirmed or refuted this candidate and have no more open leads
   worth pursuing on it, call conclude_investigation with a `summary` stating
   what you concluded and why, and `outcome` set to 'confirmed' or 'refuted'.

Concluding that a candidate is NOT a real issue is a perfectly good outcome and is
more useful than a weakly-supported finding. A refuted candidate is recorded and
shown to the user with your summary as the entire explanation, so cite the
specific evidence that ruled it out — "no issue found" on its own is not enough.

Be concrete in each finding's description: name the specific events, nodes,
and timeframe that support it, cite live cluster state where you checked it, and
give a short actionable recommendation."""


def render_scope_summary(stats) -> str:
    """One line describing the corpus a stage is working over, from
    repo_events.get_scope_summary_stats.

    Shared by both stages so neither can drift into describing something the
    other wouldn't. Callers MUST pass stats gathered for their run's own scope:
    an unscoped summary next to a cluster-scoped corpus names nodes, a time
    range and an event total the stage cannot actually see — Stage 2's evidence
    tools are restricted to the investigation's cluster, so it would be told
    about nodes every one of its queries then returns nothing for.

    The scope label leads, so the model knows this is one cluster over one
    window rather than everything that exists, and shouldn't reason about or
    reach for events outside it."""
    if not stats["total"]:
        return "No events are available to analyze."
    return (
        f"{stats['scope_label']}. "
        f"{stats['total']} events across nodes {stats['nodes']}, "
        f"time range {stats['min_time']} to {stats['max_time']}. "
        f"Severity breakdown: {stats['severity_counts']}."
    )


def render_glossary(entries) -> str:
    """Definitions for the event names actually present in the corpus, as
    Stage 1's grounding block. Without this Stage 1 is judging dotted event
    names on how alarming they *sound*; with it, it knows what they mean."""
    if not entries:
        return ""
    lines = ["Event catalog (what these event names mean, from the ONTAP EMS catalog):"]
    for entry in entries:
        lines.append(f"{entry['name']} [{entry['severity']}]: {entry['description']}")
    return "\n".join(lines)


def render_dismissals(dismissals) -> str:
    """Previously dismissed findings and the human's stated reason.

    The signature tables already block an exact repeat, so this is not about
    deduplication — it's so the model can generalize from what a human rejected
    and avoid raising near-misses that dodge an exact signature match."""
    if not dismissals:
        return ""
    lines = [
        "Findings this user has previously dismissed as not worth acting on. Do not raise "
        "these again, and weigh the stated reasons when judging similar patterns — but do "
        "not let them suppress a genuinely different or more severe issue:"
    ]
    for d in dismissals:
        scope = "everywhere" if d.get("scope") == "global" else f"on {d.get('node') or 'unspecified node'}"
        reason = f" — reason given: {d['reason']}" if d.get("reason") else ""
        lines.append(f"- [{d['category']}] {d['title']} ({scope}){reason}")
    return "\n".join(lines)
