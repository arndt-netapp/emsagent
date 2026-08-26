import sqlite3
from typing import Dict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent import compaction
from app.agent.findings import (
    CONCLUDED_UNFINALIZED,
    CONFIRMED_OUTCOME,
    READY_STATUS,
    persist_hypothesis,
    persist_partial,
    persist_refutation,
)
from app.agent.pricing import STAGE2_COST_CAP_USD, estimate_cost_usd
from app.agent.prompts import STAGE2_SYSTEM_PROMPT, render_scope_summary
from app.agent.state import AgentState
from app.agent.tools import InvestigationContext, build_tools
from app.config import settings
from app.db import repo_analysis_runs, repo_candidates, repo_events


def _over_cost_cap(state: AgentState) -> bool:
    return (
        estimate_cost_usd(
            state.get("input_tokens", 0),
            state.get("output_tokens", 0),
            state.get("cache_creation_input_tokens", 0),
            state.get("cache_read_input_tokens", 0),
        )
        >= STAGE2_COST_CAP_USD
    )


def _stop_reason(state: AgentState) -> str:
    """Why the loop ended, evaluated in persist_findings.

    LangGraph's conditional-edge functions can only return a route, not a state
    update, so the routers below can't record this — it is re-derived here from
    the same predicates in the same order. Keep the two in sync: this is what
    tells a partial result apart from a completed one."""
    if state.get("ready_to_conclude"):
        return "concluded"
    if _over_cost_cap(state):
        return "cost_cap"
    if state["iteration"] >= state["max_iterations"]:
        return "iteration_cap"
    # The model answered with prose and no tool calls, so route_after_investigate
    # sent us straight here. Not a budget bound, so no partial is written: the
    # agent stopped of its own accord without concluding.
    return "no_tool_calls"


def _latest_working_hypotheses(state: AgentState) -> list:
    """The still-open hypotheses to keep as partial results, most recent
    version of each.

    `hypotheses` is append-only (state.py's operator.add reducer), and
    record_hypothesis is documented as "record OR UPDATE" — an agent refining
    its hypothesis over several turns appends a fuller version of the same
    thing each time. Keying on (category, node, event_names) and keeping the
    last collapses those revisions into one finding instead of one per turn,
    while genuinely distinct hypotheses still come through separately."""
    latest = {}
    for hyp in state.get("hypotheses", []):
        if hyp.get("status") == READY_STATUS:
            continue
        key = (
            hyp.get("category"),
            hyp.get("node"),
            tuple(sorted(set(hyp.get("event_names") or []))),
        )
        latest[key] = hyp
    return list(latest.values())


def build_graph(
    conn: sqlite3.Connection,
    model_name: Optional[str] = None,
    usage_sink: Optional[Dict[str, int]] = None,
):
    """Build the investigate <-> tools loop for a single candidate,
    terminating via an explicit conclude_investigation() tool call, the
    max_iterations cap, or the STAGE2_COST_CAP_USD hard cost ceiling, then
    persisting any finalized (non-suppressed) hypotheses as findings linked
    back to the candidate.

    `usage_sink`, if given, is a plain dict the investigate node accumulates
    token counts into as it goes. The graph's own state is only readable
    once invoke() returns, so an exception mid-loop would otherwise discard
    every token it had already spent — see runner.execute_investigation."""
    # Shared with the tools so each call lands in agent_steps tagged with the
    # candidate and turn it belongs to. Populated in load_context (candidate)
    # and investigate (iteration) as the graph runs.
    trace = InvestigationContext()
    tools = build_tools(conn, trace=trace)
    model = ChatAnthropic(
        model=model_name or settings.claude_model,
        api_key=settings.anthropic_api_key,
        # Shared with thinking, which Sonnet 5 does by default; 4096 left a
        # deep-thinking turn able to truncate the tool call it was about to
        # emit. Headroom is only billed if used.
        max_tokens=8192,
        # langchain_anthropic has no first-class output_config field, but
        # model_kwargs is spread straight into the Messages API payload
        # (see its _get_request_payload). Without this the loop runs at the
        # API default effort of "high" on turns that are mostly routing.
        model_kwargs={"output_config": {"effort": settings.stage2_effort}},
    ).bind_tools(tools)

    def load_context(state: AgentState) -> dict:
        candidate = repo_candidates.get_candidate(conn, state["candidate_id"])
        trace.candidate_id = state["candidate_id"]
        # The cluster this candidate's run was scoped to, which is also the
        # host get_cluster_state queries. Resolved here rather than passed in
        # so no caller can build a graph that queries the wrong cluster.
        run = repo_analysis_runs.get_run(conn, candidate.analysis_run_id)
        trace.cluster = run.scope_cluster if run else None
        # The same scope the run's Stage 1 pass analyzed, rebuilt from the row.
        # None for a run predating scoping, which analyzed everything — matching
        # `trace.scoped` below, so the summary and the evidence tools always
        # describe the same set of events.
        scope = repo_analysis_runs.get_scope_for_run(conn, candidate.analysis_run_id)
        # It is also the cluster every event query is restricted to, so the
        # agent can't correlate against a different cluster's identically-named
        # nodes. `scoped` is False only for a run that predates scoping, whose
        # NULL scope_cluster means "unrecorded", not "the unspecified cluster"
        # — those runs analyzed everything and their investigations still
        # should.
        trace.scoped = bool(run and run.scope_mode)
        # Deliberately NOT the full compact corpus here (unlike Stage 1): its
        # size scales with total event count, not with this one candidate's
        # scope, and it would be baked into the very first model call — before
        # the cost-cap routing below ever gets a chance to run. At 10k+ events
        # that alone could cost far more than STAGE2_COST_CAP_USD, silently
        # defeating the cap. The scope summary is a fixed-size aggregate
        # (counts/node list/severity breakdown), not a per-event dump, so it's
        # safe regardless of scale; anything beyond it is available through
        # the tools below, at cost the cap already correctly bounds turn by turn.
        #
        # Scoped to this run, not to the whole database. Unscoped (which it was)
        # it opens the investigation by naming every node, the combined event
        # total and the full time range of every cluster ever ingested — while
        # every evidence tool below is restricted to this run's cluster, so the
        # agent is handed nodes its own queries can never return a row for.
        scope_summary = render_scope_summary(repo_events.get_scope_summary_stats(conn, scope))

        system_content = f"{STAGE2_SYSTEM_PROMPT}\n\nScope: {scope_summary}"

        # This candidate's starting evidence, resolved to concrete event ids
        # back at Stage 1 time and stored with the row — a small,
        # per-candidate list (capped at compaction.MAX_CANDIDATE_LEADS), not
        # the full corpus, so its size doesn't scale with total event count.
        #
        # It used to be re-derived here by rebuilding the whole compact
        # corpus and looking up candidate.refs. That was both a full table
        # scan per investigation and outright wrong: refs are positional, so
        # ingesting an older log file (or any out-of-order event) between the
        # scan and the investigation shifted every ref and pointed the whole
        # budget at unrelated events. Rows created before `leads` existed
        # still have to take that legacy path.
        leads_list = candidate.leads
        if not leads_list and candidate.refs:
            groups_by_ref = {g.ref: g for g in compaction.build_compact_corpus(conn)}
            leads_list = compaction.resolve_leads(groups_by_ref, candidate.refs)
        leads = compaction.render_leads(leads_list, total_refs=len(candidate.refs))

        seed = (
            "Investigate this candidate flagged by the initial triage pass:\n"
            f"Category: {candidate.category}\n"
            f"Node: {candidate.node or 'multiple/unspecified'}\n"
            f"Rationale: {candidate.rationale}\n"
            f"Starting leads:\n{leads}\n\n"
            "Use the tools available (e.g. get_event_context on the event_ids above) to gather "
            "further evidence, confirm or refute this candidate, and either finalize it as a "
            "finding (after checking suppression) or conclude it isn't a real issue."
        )
        return {
            "messages": [
                SystemMessage(content=system_content),
                HumanMessage(content=seed),
            ],
            "iteration": 0,
            "hypotheses": [],
            "finalized_findings": [],
            "findings_suppressed": 0,
            "ready_to_conclude": False,
            "conclusion_summary": "",
            "conclusion_outcome": "",
            "stop_reason": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    def investigate(state: AgentState) -> dict:
        # cache_control auto-caches the last block of the outgoing request.
        # Since messages only ever grow turn to turn, each turn's prefix is a
        # byte-identical extension of the previous one, so this is the
        # standard "growing conversation" caching pattern: turn N's marker
        # writes a cache entry for everything up to N; turn N+1 reads that
        # back cheaply and writes only its own new increment. Verified live
        # against the real API before wiring this in.
        # Bump before the call so the tool calls this turn produces — which run
        # in the *next* node — are traced against the turn that generated them.
        trace.iteration = state["iteration"] + 1
        response = model.invoke(state["messages"], cache_control={"type": "ephemeral"})
        usage = getattr(response, "usage_metadata", None) or {}
        details = usage.get("input_token_details") or {}
        cache_read = details.get("cache_read") or 0
        cache_creation = details.get("cache_creation") or 0
        # langchain_anthropic's usage_metadata["input_tokens"] is base input
        # PLUS cache_read PLUS cache_creation, summed into one "total
        # processed" figure (see its _create_usage_metadata) — subtract the
        # cache portions back out so estimate_cost_usd prices each tier
        # correctly instead of billing cheap cache reads at full input price.
        total_input = usage.get("input_tokens") or 0
        base_input = max(total_input - cache_read - cache_creation, 0)
        output = usage.get("output_tokens") or 0
        if usage_sink is not None:
            # Mirror into the caller's dict so a failure later in the loop
            # still reports what this turn cost.
            usage_sink["input_tokens"] = usage_sink.get("input_tokens", 0) + base_input
            usage_sink["output_tokens"] = usage_sink.get("output_tokens", 0) + output
            usage_sink["cache_creation_input_tokens"] = (
                usage_sink.get("cache_creation_input_tokens", 0) + cache_creation
            )
            usage_sink["cache_read_input_tokens"] = usage_sink.get("cache_read_input_tokens", 0) + cache_read
            usage_sink["iterations"] = state["iteration"] + 1
        return {
            "messages": [response],
            "iteration": state["iteration"] + 1,
            "input_tokens": state.get("input_tokens", 0) + base_input,
            "output_tokens": state.get("output_tokens", 0) + output,
            "cache_creation_input_tokens": state.get("cache_creation_input_tokens", 0) + cache_creation,
            "cache_read_input_tokens": state.get("cache_read_input_tokens", 0) + cache_read,
        }

    def persist_findings(state: AgentState) -> dict:
        finalized = []
        suppressed_count = 0
        stop_reason = _stop_reason(state)
        ready = [h for h in state.get("hypotheses", []) if h.get("status") == READY_STATUS]
        for hyp in ready:
            finding = persist_hypothesis(
                conn, state.get("analysis_run_id"), hyp, candidate_id=state.get("candidate_id")
            )
            if finding is None:
                suppressed_count += 1
            else:
                finalized.append(
                    {"finding_id": finding.id, "title": finding.title, "category": finding.category}
                )

        # Everything below is about what to leave behind when NOTHING reached
        # status='ready'. `ready` rather than `finalized`: hypotheses that were
        # finalized but suppressed mean the agent DID find something, already
        # known — that is neither a refutation nor an unfinished investigation.
        if not ready:
            # An investigation that RULED THE CANDIDATE OUT is a real result and
            # is recorded as one, so every Stage 2 outcome is visible in the same
            # place instead of a refutation leaving nothing behind but a tool
            # trace.
            #
            # Gated on the agent's own `outcome`, not merely on "it concluded and
            # finalized nothing". An agent that concludes outcome='confirmed'
            # without re-recording its hypothesis at status='ready' HAS NOT ruled
            # anything out — filing that as "No issue found" (which is what
            # happened before this check existed) states the exact opposite of
            # what it concluded, at severity=info, on a record the user is not
            # even allowed to dismiss. `!= CONFIRMED_OUTCOME` rather than
            # `== "refuted"`: conclude_investigation already normalizes and
            # defaults to 'refuted', so an unrecognized value keeps the
            # refutation behaviour rather than silently recording nothing.
            #
            # Hitting the cost cap or the iteration cap is not a refutation
            # either — the budget ran out before the agent reached a view — so
            # `ready_to_conclude` is still required.
            concluded_confirmed = state.get("conclusion_outcome") == CONFIRMED_OUTCOME
            if state.get("ready_to_conclude") and state.get("conclusion_summary") and not concluded_confirmed:
                candidate = repo_candidates.get_candidate(conn, state["candidate_id"])
                if candidate is not None:
                    event_names = sorted(
                        {lead["event_name"] for lead in (candidate.leads or []) if lead.get("event_name")}
                    )
                    if event_names:
                        refutation = persist_refutation(
                            conn,
                            state.get("analysis_run_id"),
                            candidate_id=candidate.id,
                            category=candidate.category,
                            node=candidate.node,
                            event_names=event_names,
                            summary=state["conclusion_summary"],
                        )
                        finalized.append(
                            {
                                "finding_id": refutation.id,
                                "title": refutation.title,
                                "category": refutation.category,
                            }
                        )

            # An investigation that ended without finalizing still has whatever
            # it had worked out, in the hypotheses the agent recorded with
            # status='investigating' as it went (the Stage 2 prompt tells it to
            # record early for exactly this reason). Those used to be dropped on
            # the floor: the run was billed in full and the user was shown
            # nothing but a tool trace. They are kept as `partial` findings
            # instead — explicitly unconfirmed, and never as *open* findings, so
            # a later investigation that actually concludes can still raise the
            # real one.
            #
            # Two ways to get here, and the note on the finding says which:
            #   * a budget bound (cost or turns), the original case;
            #   * the agent concluding 'confirmed' while leaving its hypothesis
            #     at status='investigating' — the claim is unverified by the
            #     finalize path (no suppression check, no promotion), so it is a
            #     lead, not a finding, but it is certainly not nothing.
            #
            # NOT written when the model simply stopped talking without
            # concluding: that didn't run out of anything and asserted nothing.
            partial_reason = None
            if stop_reason in {"cost_cap", "iteration_cap"}:
                partial_reason = stop_reason
            elif stop_reason == "concluded" and concluded_confirmed:
                partial_reason = CONCLUDED_UNFINALIZED
            if partial_reason is not None:
                for hyp in _latest_working_hypotheses(state):
                    partial = persist_partial(
                        conn,
                        state.get("analysis_run_id"),
                        hyp,
                        candidate_id=state.get("candidate_id"),
                        stop_reason=partial_reason,
                        iterations=state.get("iteration"),
                    )
                    if partial is None:
                        suppressed_count += 1
                    else:
                        finalized.append(
                            {"finding_id": partial.id, "title": partial.title, "category": partial.category}
                        )

        return {
            "finalized_findings": finalized,
            "findings_suppressed": suppressed_count,
            "stop_reason": stop_reason,
        }

    def route_after_investigate(state: AgentState) -> str:
        # Always run the tool calls this turn produced, cap or no cap. The
        # model already generated them and you were already billed for them;
        # executing them is pure local DB work costing zero tokens, and it
        # is the only way a record_hypothesis(status='ready') from the
        # cap-crossing turn ever reaches persist_findings. The cap is
        # enforced one step later, in route_after_tools, which is what
        # actually matters — that's the edge back into a *model* call.
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "persist_findings"

    def route_after_tools(state: AgentState) -> str:
        # The single gate on spending more money: everything below stops the
        # loop before another model call, having already banked this turn's
        # tool results.
        if state.get("ready_to_conclude"):
            return "persist_findings"
        if _over_cost_cap(state):
            return "persist_findings"
        if state["iteration"] >= state["max_iterations"]:
            return "persist_findings"
        return "investigate"

    graph = StateGraph(AgentState)
    graph.add_node("load_context", load_context)
    graph.add_node("investigate", investigate)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("persist_findings", persist_findings)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "investigate")
    graph.add_conditional_edges("investigate", route_after_investigate, ["tools", "persist_findings"])
    graph.add_conditional_edges("tools", route_after_tools, ["investigate", "persist_findings"])
    graph.add_edge("persist_findings", END)

    return graph.compile()
