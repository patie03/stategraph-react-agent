"""Human-in-the-loop: pause before sensitive tools, resume on approval, recover on rejection."""

import json
import os

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from src import tools as T
from src.graph import build_graph, is_paused, openai_llm, resume, send
from tests.conftest import call

SCENARIO = ("Check the current error rate on the auth-service. If it's over 5%, scale its replicas to 10 and "
            "transfer $1,500 to the contracted on-call engineering firm (Vendor ID: V-8812) for emergency mitigation.")


def ops_agent(msgs):
    """Scripted version of what gpt-4o-mini does for the scenario."""
    last = msgs[-1]
    if not isinstance(last, ToolMessage):
        return AIMessage("", tool_calls=[call("query_service_metrics", service_name="auth-service")])
    if last.name == "query_service_metrics":
        rate = json.loads(last.content)["error_rate_pct"]
        assert rate > 5
        return AIMessage("", tool_calls=[call("scale_kubernetes_cluster", service_name="auth-service", replica_count=10),
                                         call("execute_wire_transfer", recipient_id="V-8812", amount_usd=1500)])
    if "REJECTED BY HUMAN OPERATOR" in last.content or "NOT RUN" in last.content:
        return AIMessage("Understood: the scaling and the $1,500 transfer were cancelled by the operator. "
                         "auth-service is still at 7.2% errors; I can keep monitoring it.")
    return AIMessage("Done: auth-service scaled to 10 replicas and $1,500 sent to V-8812.")


# ---------------------------------------------------------------- Test 5: interruption
def test_sensitive_action_suspends_at_human_gate(scripted, thread_id):
    graph = build_graph(scripted(ops_agent))
    res = send(graph, SCENARIO, thread_id)

    assert res.paused and res.next == ("human_approval_gate",)
    assert res.path == ["agent_node", "circuit_breaker_node", "tool_executor_node", "agent_node", "circuit_breaker_node"]
    assert [a["tool"] for a in res.ticket["actions"]] == ["scale_kubernetes_cluster", "execute_wire_transfer"]
    wire = res.ticket["actions"][1]
    assert wire["args"] == {"recipient_id": "V-8812", "amount_usd": 1500}
    assert wire["risk"].startswith("HIGH")
    assert res.values["pending_approval"] == res.ticket  # visible in the checkpointed state
    assert T.EXECUTION_LOG == [] and T.SERVICES["auth-service"]["replicas"] == 4  # nothing ran yet


def test_safe_only_requests_never_pause(scripted, thread_id):
    graph = build_graph(scripted(lambda m: AIMessage("", tool_calls=[call("check_billing_balance", account_id="ACCT-3003")])
                                 if not isinstance(m[-1], ToolMessage) else AIMessage("Balance read.")))
    assert not send(graph, "balance?", thread_id).paused


def test_pause_survives_a_new_graph_instance(scripted, thread_id):
    """The checkpoint, not the Python object, holds the paused run (e.g. a Streamlit rerun)."""
    saver = MemorySaver()
    send(build_graph(scripted(ops_agent), checkpointer=saver), SCENARIO, thread_id)
    fresh = build_graph(scripted(ops_agent), checkpointer=saver)
    assert is_paused(fresh, thread_id)
    assert resume(fresh, "APPROVED", thread_id).next == ()


# ---------------------------------------------------------------- Test 6: approval resumes
def test_approval_resumes_and_executes(scripted, thread_id):
    graph = build_graph(scripted(ops_agent))
    send(graph, SCENARIO, thread_id)
    res = resume(graph, "APPROVED", thread_id)

    assert res.path == ["human_approval_gate", "tool_executor_node", "agent_node"]
    assert res.next == () and not res.paused  # END
    assert [e["action"] for e in T.EXECUTION_LOG] == ["scale_kubernetes_cluster", "execute_wire_transfer"]
    assert T.SERVICES["auth-service"]["replicas"] == 10
    assert T.ACCOUNTS["ACCT-3003"]["balance_usd"] == 5900.00
    results = [m for m in res.values["messages"] if isinstance(m, ToolMessage)][-2:]
    assert all(m.status == "success" for m in results)
    assert "Done" in res.answer
    with pytest.raises(RuntimeError):
        resume(graph, "APPROVED", thread_id)  # can't approve twice


# ---------------------------------------------------------------- Test 7: rejection recovery
def test_rejection_bypasses_tool_and_agent_explains(scripted, thread_id):
    llm = scripted(ops_agent)
    graph = build_graph(llm)
    send(graph, SCENARIO, thread_id)
    calls_before = len(llm.seen)
    res = resume(graph, "REJECTED", thread_id)

    assert res.path == ["human_approval_gate", "agent_node"]
    assert res.next == () and not res.paused
    assert T.EXECUTION_LOG == [] and T.SERVICES["auth-service"]["replicas"] == 4
    assert T.ACCOUNTS["ACCT-3003"]["balance_usd"] == 7400.00
    rejected = [m for m in res.values["messages"] if isinstance(m, ToolMessage) and "REJECTED BY HUMAN" in m.content]
    assert {m.name for m in rejected} == {"scale_kubernetes_cluster", "execute_wire_transfer"}
    assert "cancelled" in res.answer
    assert len(llm.seen) == calls_before + 1  # one explanation, no new approval request
    assert res.values["approval_status"] == "REJECTED"


def test_reproposing_a_rejected_action_is_blocked_not_reasked(scripted, thread_id):
    """A stubborn model that asks for the same transfer again must not create an approval loop."""
    def stubborn(msgs):
        if not isinstance(msgs[-1], ToolMessage):
            return AIMessage("", tool_calls=[call("execute_wire_transfer", recipient_id="V-8812", amount_usd=1500)])
        return AIMessage("", tool_calls=[call("execute_wire_transfer", recipient_id="V-8812", amount_usd=1500.0)])

    graph = build_graph(scripted(stubborn))
    send(graph, "pay V-8812 $1500", thread_id)
    res = resume(graph, "REJECTED", thread_id)
    assert not res.paused
    assert res.path == ["human_approval_gate", "agent_node", "circuit_breaker_node", "error_node"]
    assert "already rejected" in res.answer
    assert T.EXECUTION_LOG == []


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY", "").strip(), reason="needs OPENAI_API_KEY")
def test_live_llm_full_scenario_reject(thread_id):
    graph = build_graph(openai_llm())
    res = send(graph, SCENARIO, thread_id)
    assert res.paused and {a["tool"] for a in res.ticket["actions"]} & {"execute_wire_transfer", "scale_kubernetes_cluster"}
    while res.paused:  # the model may ask for the two actions in one or two steps
        res = resume(graph, "REJECTED", thread_id)
    assert T.EXECUTION_LOG == []
    assert any(w in res.answer.lower() for w in ("cancel", "reject", "denied", "not"))
