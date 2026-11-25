"""The state graph: nodes, conditional edges, checkpointer and run/resume helpers.

    START → agent_node ─(tool calls?)─► circuit_breaker_node ─(tripped)──────► error_node → END
                │ no                           │ sensitive                          ▲
                ▼                              ▼                                    │
               END                    human_approval_gate ── interrupt() ── (pause, checkpoint)
                                         │ APPROVED      │ REJECTED
                                         ▼               ▼
              agent_node ◄──── tool_executor_node    agent_node (with rejection ToolMessages)
              (safe calls go straight from circuit_breaker_node to tool_executor_node)

Rules every node follows:
- Return a partial update dict; never mutate `state` (LangGraph merges the update and checkpoints it).
- Every tool_call_id in an AIMessage gets exactly one ToolMessage, on every path (run, rejected,
  blocked, halted). Otherwise the next OpenAI request on this thread is rejected.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src import circuit_breaker as cb
from src.state import AgentState
from src.tools import ALL_TOOLS, is_sensitive, risk_label

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GRAPH_RECURSION_LIMIT = 40  # LangGraph's own backstop; our breaker stops much earlier

SYSTEM_PROMPT = """You are the Internal Cloud & Financial Operations agent.
- Use tools to get facts. Never invent metrics, balances or confirmations.
- Read-only tools (query_service_metrics, check_billing_balance) run immediately.
- Sensitive tools (scale_kubernetes_cluster, execute_wire_transfer) are paused automatically for human
  approval. Just call them when the request needs them; do not ask the user for permission in text.
- For conditional requests ("if the error rate is over 5%..."), first read the data, then decide.
- If a tool result says the operator REJECTED an action, do not call that action again. Tell the user
  it was cancelled and offer a read-only alternative.
- If a tool fails, you may retry once with corrected arguments; otherwise explain the failure.
- Keep answers short and factual."""

NODE_LABELS = {
    "agent_node": "AGENT", "circuit_breaker_node": "CIRCUIT_BREAKER", "human_approval_gate": "HUMAN_GATE",
    "tool_executor_node": "TOOL_EXECUTOR", "error_node": "ERROR", "__interrupt__": "PAUSED: HUMAN_GATE",
}


def _last_ai(state: AgentState) -> AIMessage:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            return m
    raise ValueError("no AIMessage in state")


def _json(obj: Any) -> str:
    return json.dumps(obj, default=str)


def build_graph(llm: BaseChatModel, tools: Sequence[BaseTool] = ALL_TOOLS, checkpointer=None,
                max_iterations: int = cb.MAX_ITERATIONS):
    by_name = {t.name: t for t in tools}
    model = llm.bind_tools(list(tools))

    def sensitive(name: str) -> bool:
        return name in by_name and is_sensitive(by_name[name])

    # ------------------------------------------------------------------ nodes
    def agent_node(state: AgentState) -> dict:
        reply = model.invoke([SystemMessage(SYSTEM_PROMPT), *state.get("messages", [])])
        return {"messages": [reply], "iteration_count": state.get("iteration_count", 0) + 1}

    def circuit_breaker_node(state: AgentState) -> dict:
        calls = _last_ai(state).tool_calls
        verdict = cb.check(calls, state.get("iteration_count", 0), state.get("executed_tool_hashes", []),
                           state.get("rejected_hashes", []), max_iterations)
        update: dict = {"executed_tool_hashes": verdict.hashes, "halt_reason": verdict.reason,
                        "pending_approval": None}
        if not verdict.tripped and any(sensitive(c["name"]) for c in calls):
            update["pending_approval"] = {
                "actions": [{"tool_call_id": c["id"], "tool": c["name"], "args": c.get("args", {}),
                             "risk": risk_label(by_name[c["name"]])} for c in calls if sensitive(c["name"])],
                "also_in_batch": [c["name"] for c in calls if not sensitive(c["name"])],
                "iteration": state.get("iteration_count", 0),
            }
        return update

    def human_approval_gate(state: AgentState) -> dict:
        # interrupt() saves the checkpoint and suspends the run here. On resume the node runs again
        # from the top and interrupt() returns the human's answer, so nothing before it has side effects.
        answer = interrupt(state["pending_approval"])
        decision = str(answer.get("decision") if isinstance(answer, dict) else answer).strip().upper()
        status = "APPROVED" if decision in {"APPROVED", "APPROVE", "YES"} else "REJECTED"
        update: dict = {"approval_status": status, "pending_approval": None}
        if status == "REJECTED":
            msgs, denied = [], []
            for c in _last_ai(state).tool_calls:
                if sensitive(c["name"]):
                    denied.append(cb.tool_call_hash(c["name"], c.get("args", {})))
                    text = (f"REJECTED BY HUMAN OPERATOR: authorization for {c['name']}"
                            f"({cb.normalize_args(c.get('args'))}) was denied. Do NOT call this action again "
                            "and do not ask for approval again. Tell the user it was cancelled and offer "
                            "read-only alternatives.")
                else:
                    text = "NOT RUN: skipped because the operator rejected this batch of actions."
                msgs.append(ToolMessage(text, tool_call_id=c["id"], name=c["name"], status="error"))
            update["messages"] = msgs
            update["rejected_hashes"] = [*state.get("rejected_hashes", []), *denied]
        return update

    def tool_executor_node(state: AgentState) -> dict:
        approved = state.get("approval_status") == "APPROVED"
        out = []
        for c in _last_ai(state).tool_calls:
            tool_obj = by_name.get(c["name"])
            if tool_obj is None:
                content, status = _json({"error": f"Unknown tool '{c['name']}'. Available: {sorted(by_name)}"}), "error"
            elif is_sensitive(tool_obj) and not approved:  # defense in depth: never run unapproved
                content, status = _json({"error": f"{c['name']} requires human approval."}), "error"
            else:
                try:
                    content, status = _json(tool_obj.invoke(c.get("args", {}))), "success"
                except Exception as exc:  # tool failures go back to the model, not up the stack
                    content, status = _json({"error": f"{type(exc).__name__}: {exc}"}), "error"
            out.append(ToolMessage(content, tool_call_id=c["id"], name=c["name"], status=status))
        return {"messages": out, "approval_status": None}

    def error_node(state: AgentState) -> dict:
        reason = state.get("halt_reason") or "Unknown error."
        last = _last_ai(state)
        closing = [ToolMessage(f"NOT RUN: halted by circuit breaker. {reason}", tool_call_id=c["id"],
                               name=c["name"], status="error") for c in last.tool_calls]
        final = AIMessage(f"⛔ I stopped this request safely. {reason} No further actions were taken. "
                          "Please check the inputs or try again later.")
        return {"messages": [*closing, final], "pending_approval": None}

    # ------------------------------------------------------------------ edges
    def after_agent(state: AgentState) -> Literal["circuit_breaker_node", "__end__"]:
        return "circuit_breaker_node" if _last_ai(state).tool_calls else END

    def after_breaker(state: AgentState) -> Literal["error_node", "human_approval_gate", "tool_executor_node"]:
        if state.get("halt_reason"):
            return "error_node"
        return "human_approval_gate" if state.get("pending_approval") else "tool_executor_node"

    def after_gate(state: AgentState) -> Literal["tool_executor_node", "agent_node"]:
        return "tool_executor_node" if state.get("approval_status") == "APPROVED" else "agent_node"

    g = StateGraph(AgentState)
    g.add_node("agent_node", agent_node)
    g.add_node("circuit_breaker_node", circuit_breaker_node)
    g.add_node("human_approval_gate", human_approval_gate)
    g.add_node("tool_executor_node", tool_executor_node)
    g.add_node("error_node", error_node)
    g.add_edge(START, "agent_node")
    g.add_conditional_edges("agent_node", after_agent)
    g.add_conditional_edges("circuit_breaker_node", after_breaker)
    g.add_conditional_edges("human_approval_gate", after_gate)
    g.add_edge("tool_executor_node", "agent_node")
    g.add_edge("error_node", END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def openai_llm(model: str = DEFAULT_MODEL, api_key: Optional[str] = None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=0, timeout=30, max_retries=1,
                      api_key=(api_key or os.getenv("OPENAI_API_KEY", "")).strip() or None)


# ---------------------------------------------------------------------- run / resume helpers
@dataclass
class RunResult:
    path: list[str]                      # nodes visited in this run, in order
    paused: bool                         # True if waiting at the human gate
    ticket: Optional[dict]               # the approval ticket when paused
    values: dict = field(default_factory=dict)
    next: tuple = ()

    @property
    def answer(self) -> str:
        for m in reversed(self.values.get("messages", [])):
            if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                return m.content
        return ""


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": GRAPH_RECURSION_LIMIT}


def new_turn(text: str) -> dict:
    """Input for a new user message. Per-turn counters reset; the message history is kept by the checkpointer."""
    return {"messages": [HumanMessage(text)], "iteration_count": 0, "executed_tool_hashes": [],
            "rejected_hashes": [], "halt_reason": None, "approval_status": None, "pending_approval": None}


def _drive(graph, payload, thread_id: str) -> RunResult:
    cfg = config_for(thread_id)
    path: list[str] = []
    for chunk in graph.stream(payload, cfg, stream_mode="updates"):
        path.extend(chunk.keys())
    snap = graph.get_state(cfg)
    paused = "human_approval_gate" in snap.next
    ticket = None
    if paused:
        ticket = next((i.value for t in snap.tasks for i in (t.interrupts or ())), None)
    return RunResult([p for p in path if p != "__interrupt__"], paused, ticket, dict(snap.values), snap.next)


def send(graph, text: str, thread_id: str) -> RunResult:
    if not text or not text.strip():
        raise ValueError("Empty message.")
    if is_paused(graph, thread_id):
        raise RuntimeError("This thread is waiting for an approval decision. Approve or reject first.")
    return _drive(graph, new_turn(text.strip()), thread_id)


def resume(graph, decision: Literal["APPROVED", "REJECTED"], thread_id: str) -> RunResult:
    if not is_paused(graph, thread_id):
        raise RuntimeError("Nothing to resume on this thread.")
    return _drive(graph, Command(resume={"decision": decision}), thread_id)


def is_paused(graph, thread_id: str) -> bool:
    return "human_approval_gate" in graph.get_state(config_for(thread_id)).next
