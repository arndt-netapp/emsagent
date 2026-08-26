import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Hypothesis(TypedDict, total=False):
    category: str
    severity: str
    title: str
    description: str
    recommendation: str
    node: str
    event_names: List[str]
    evidence_event_ids: List[int]
    confidence: float
    status: str  # "investigating" | "ready"


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    analysis_run_id: int
    candidate_id: int
    iteration: int
    max_iterations: int
    # operator.add reducer: parallel tool calls in the same step (e.g. two
    # record_hypothesis calls from one model turn) each contribute only their
    # own new hypothesis (see tools.py), and LangGraph concatenates them
    # instead of rejecting the step for multiple writes to the same key.
    hypotheses: Annotated[List[Hypothesis], operator.add]
    finalized_findings: List[Dict[str, Any]]
    findings_suppressed: int
    ready_to_conclude: bool
    # Why the agent stopped, from conclude_investigation. A candidate it ruled
    # out is shown to the user with this summary as the explanation, so the
    # reasoning has to be stated rather than left implicit in the tool trace.
    conclusion_summary: str
    conclusion_outcome: str  # "confirmed" | "refuted"
    # Why the loop ended, derived in persist_findings: "concluded" |
    # "cost_cap" | "iteration_cap" | "no_tool_calls". Distinguishes an
    # investigation that finished from one the budget cut short, which is what
    # decides whether a still-open hypothesis is kept as a partial result.
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
