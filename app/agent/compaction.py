import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.db import repo_events
from app.db.models import EventScope

# Ceiling on how many resolved leads a single candidate carries into Stage 2.
# Stage 2's seed message renders one line per lead and is built into the very
# first model call, before the cost cap's routing checks can run — so an
# "event storm" candidate citing thousands of refs would blow the cap before
# the loop starts.
#
# This is enforced HERE and nowhere else. It is quoted in STAGE1_SCHEMA's `refs`
# description so the model self-limits, but it deliberately is NOT a `maxItems`
# on that array: the structured-output validator rejects `maxItems` outright
# with HTTP 400, and the scripted fakes cannot catch a request-shape error, so
# "tidying" it into the schema breaks every real Stage 1 call while the suite
# stays green. See stage1.STAGE1_SCHEMA.
MAX_CANDIDATE_LEADS = 40


@dataclass
class CompactEventGroup:
    ref: int
    node: Optional[str]
    event_name: str
    severity: Optional[str]
    count: int
    first_event_id: int
    last_event_id: int
    first_time: Optional[str]
    last_time: Optional[str]


def build_compact_corpus(
    conn: sqlite3.Connection, scope: Optional[EventScope] = None
) -> List[CompactEventGroup]:
    """Deduplicate exact, immediately-repeated (node, event_name, severity)
    firings into single rows carrying a count and time span. Adjacency is
    tracked per node independently while walking all events in chronological
    order, so a node's run of identical events collapses even when other
    nodes' unrelated events are interleaved in time — but a different event
    on the same node still correctly breaks the run, rather than being
    silently folded into a misleadingly wide time span. Different event
    types and different nodes are never merged: that judgment is left
    entirely to the LLM, which sees every distinct group side-by-side in
    chronological order.

    Adjacency is keyed on (cluster, node), not node alone. A run is normally
    scoped to one cluster so the distinction can't bite — but ONTAP names nodes
    <cluster>-01/-02 and the simulator defaults to "cluster1", so two clusters
    genuinely can present identical node names. Keyed on node alone, an
    unscoped call would interleave two clusters' events into single groups with
    fabricated counts and time spans, and hand the model a pattern that never
    happened."""
    events = repo_events.get_all_events_ordered(conn, scope)
    open_group_by_node: Dict[Tuple[Optional[str], Optional[str]], CompactEventGroup] = {}
    groups: List[CompactEventGroup] = []
    next_ref = 1
    for e in events:
        key = (e.cluster, e.node)
        open_group = open_group_by_node.get(key)
        if open_group is not None and open_group.event_name == e.event_name and open_group.severity == e.severity:
            open_group.count += 1
            open_group.last_event_id = e.id
            open_group.last_time = e.event_time
            continue
        new_group = CompactEventGroup(
            ref=next_ref,
            node=e.node,
            event_name=e.event_name,
            severity=e.severity,
            count=1,
            first_event_id=e.id,
            last_event_id=e.id,
            first_time=e.event_time,
            last_time=e.event_time,
        )
        next_ref += 1
        groups.append(new_group)
        open_group_by_node[key] = new_group
    return groups


def resolve_leads(
    groups_by_ref: Dict[int, CompactEventGroup],
    refs: List[int],
    limit: int = MAX_CANDIDATE_LEADS,
) -> List[Dict[str, Any]]:
    """Turn a candidate's `ref` list into concrete, self-contained lead
    records, resolved at Stage 1 time and persisted with the candidate.

    Refs are POSITIONAL — `build_compact_corpus` numbers groups by walk order
    over every ingested event — so they are only meaningful against the exact
    event set they were generated from. Re-deriving them later (which Stage 2
    used to do) silently resolves to the wrong events as soon as anything is
    ingested out of chronological order, an older log file is uploaded, or an
    event is deleted. Resolving once, here, and storing event ids makes the
    lead durable; ids are stable for the life of the row.

    Refs with no matching group are dropped (the model can hallucinate them);
    callers should compare the returned length against `len(refs)` if they
    care. Returns at most `limit` leads — see MAX_CANDIDATE_LEADS."""
    leads: List[Dict[str, Any]] = []
    for ref in refs:
        group = groups_by_ref.get(ref)
        if group is None:
            continue
        leads.append(
            {
                "ref": group.ref,
                "node": group.node,
                "event_name": group.event_name,
                "severity": group.severity,
                "count": group.count,
                "first_event_id": group.first_event_id,
                "last_event_id": group.last_event_id,
                "first_time": group.first_time,
                "last_time": group.last_time,
            }
        )
        if len(leads) >= limit:
            break
    return leads


def render_leads(leads: List[Dict[str, Any]], total_refs: Optional[int] = None) -> str:
    """Format resolved leads as the starting-evidence block of Stage 2's seed
    message. `total_refs` is the candidate's original ref count, used only to
    tell the model when its lead list was truncated so it doesn't mistake a
    capped list for the whole picture."""
    if not leads:
        return "(no specific events resolved — use the tools to search for this pattern)"
    lines = []
    for lead in leads:
        span = (
            f"event_id {lead['first_event_id']}"
            if lead["count"] == 1
            else f"event_ids {lead['first_event_id']}-{lead['last_event_id']}"
        )
        lines.append(
            f"- {lead['event_name']} ({lead['severity']}) on {lead['node']}: "
            f"fired {lead['count']}x, {lead['first_time']} to {lead['last_time']}, {span}"
        )
    if total_refs is not None and total_refs > len(leads):
        lines.append(
            f"- (...{total_refs - len(leads)} further leads omitted — use the search tools "
            "if you need to see the rest of this pattern)"
        )
    return "\n".join(lines)


def render_compact_corpus(groups: List[CompactEventGroup]) -> str:
    """The corpus text Stage 1 pays for, one line per group.

    How much compaction actually saves is entirely data-dependent, and the
    optimistic number is the wrong one to plan with. Only immediately adjacent,
    identical firings merge, so the ratio is set by how repetitive the corpus
    is, not by its size:

      * ~14 tokens/event (vs ~48 for naive per-event JSON) where several events
        collapse into each row — the best case;
      * ~55-60 tokens/event on real cluster data (measured 2026-08-19), where
        events rarely repeat back-to-back and this produces roughly one row per
        event. A row is ~93 characters, dominated by two full ISO-8601
        timestamps — which are IDENTICAL when count=1, since last_time is
        emitted regardless.

    Two slimmings follow from that and are deliberately NOT implemented: drop
    last_time when count=1, and hoist the repeated date out of the per-row
    timestamps (~25% of the corpus text between them). Judged not worth the
    format churn at PoC scale, and both would need a real-API check that the
    model still reads the format as well — the fakes cannot tell you that."""
    lines = ["ref|first_time|last_time|node|event_name|severity|count"]
    for g in groups:
        lines.append(
            f"{g.ref}|{g.first_time or ''}|{g.last_time or ''}|{g.node or ''}|{g.event_name}|{g.severity or ''}|{g.count}"
        )
    return "\n".join(lines)
