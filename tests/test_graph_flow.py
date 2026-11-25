"""Autonomous traversal with safe tools, and short-term memory across turns."""

import json
import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph import build_graph, openai_llm, send
from tests.conftest import call


def last_tool_results(messages):
    out = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            out.append(m)
        elif out:
            break
    return list(reversed(out))


# ---------------------------------------------------------------- Test 1: safe tools run autonomously
def test_safe_tool_runs_without_interruption(scripted, thread_id):
    def respond(msgs):
        if not isinstance(msgs[-1], ToolMessage):
            return AIMessage("", tool_calls=[call("query_service_metrics", service_name="auth-service")])
        data = json.loads(msgs[-1].content)
        return AIMessage(f"auth-service error rate is {data['error_rate_pct']}%.")

    graph = build_graph(scripted(respond))
    res = send(graph, "What's the error rate on auth-service?", thread_id)

    assert res.path == ["agent_node", "circuit_breaker_node", "tool_executor_node", "agent_node"]
    assert "human_approval_gate" not in res.path and not res.paused
    assert res.next == ()  # reached END
    assert "7.2%" in res.answer
    assert res.values["iteration_count"] == 2


def test_tool_error_is_returned_to_the_model_not_raised(scripted, thread_id):
    def respond(msgs):
        if not isinstance(msgs[-1], ToolMessage):
            return AIMessage("", tool_calls=[call("query_service_metrics", service_name="billing-db")])
        return AIMessage("That service does not exist.")

    res = send(build_graph(scripted(respond)), "metrics for billing-db", thread_id)
    tool_msg = last_tool_results(res.values["messages"])[0]
    assert tool_msg.status == "error" and "Unknown service" in tool_msg.content
    assert res.answer == "That service does not exist."


# ---------------------------------------------------------------- Test 2: multi-turn memory
def test_memory_across_turns_on_same_thread(scripted, thread_id):
    def respond(msgs):
        humans = [m.content for m in msgs if isinstance(m, HumanMessage)]
        if humans[-1] == "What is my name?":
            name = next((h.split("is ")[-1] for h in humans if h.startswith("My name is")), None)
            return AIMessage(f"Your name is {name}." if name else "I don't know your name.")
        return AIMessage("Nice to meet you!")

    llm = scripted(respond)
    graph = build_graph(llm)
    send(graph, "My name is Alice", thread_id)
    res = send(graph, "What is my name?", thread_id)

    assert res.answer == "Your name is Alice."
    assert len(res.values["messages"]) == 4  # 2 human + 2 AI, restored from the checkpointer
    other = send(graph, "What is my name?", thread_id + "-other")
    assert other.answer == "I don't know your name."  # threads are isolated


def test_per_turn_counters_reset_but_history_is_kept(scripted, thread_id):
    graph = build_graph(scripted(lambda m: AIMessage("ok")))
    send(graph, "one", thread_id)
    res = send(graph, "two", thread_id)
    assert res.values["iteration_count"] == 1
    assert [m.content for m in res.values["messages"] if isinstance(m, HumanMessage)] == ["one", "two"]


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY", "").strip(), reason="needs OPENAI_API_KEY")
def test_live_llm_memory_and_safe_tool(thread_id):
    graph = build_graph(openai_llm())
    send(graph, "Hi, my name is Alice.", thread_id)
    assert "alice" in send(graph, "What is my name?", thread_id).answer.lower()
    res = send(graph, "What is the current error rate on auth-service?", thread_id)
    assert not res.paused and "7.2" in res.answer
