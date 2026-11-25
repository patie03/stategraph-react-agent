"""Iteration cap and SHA-256 repeat detection."""

import itertools

from langchain_core.messages import AIMessage, ToolMessage

from src import circuit_breaker as cb
from src.graph import build_graph, send
from src.tools import ALL_TOOLS, FAILING_TOOL
from tests.conftest import call

TOOLS = [*ALL_TOOLS, FAILING_TOOL]


def assert_every_tool_call_answered(messages):
    asked = [tc["id"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
    answered = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    assert sorted(asked) == sorted(answered)


# ---------------------------------------------------------------- Test 3: hard iteration bound
def test_iteration_bound_halts_at_six(scripted, thread_id):
    n = itertools.count(1)
    # A failing tool called with a DIFFERENT argument every time, so only the iteration cap can stop it.
    llm = scripted(lambda m: AIMessage("", tool_calls=[call("fetch_incident_report", incident_id=f"INC-{next(n)}")]))
    res = send(build_graph(llm, TOOLS), "Get the incident report", thread_id)

    assert res.values["iteration_count"] == cb.MAX_ITERATIONS == 6
    assert res.path[-2:] == ["circuit_breaker_node", "error_node"]
    assert len(llm.seen) == 6
    assert "Iteration limit reached" in res.answer
    assert_every_tool_call_answered(res.values["messages"])  # thread stays valid for the next turn


# ---------------------------------------------------------------- Test 4: SHA-256 repeat trip
def test_identical_consecutive_calls_trip_the_hash_breaker(scripted, thread_id):
    llm = scripted(lambda m: AIMessage("", tool_calls=[call("fetch_incident_report", incident_id="INC-42")]))
    res = send(build_graph(llm, TOOLS), "Get INC-42", thread_id)

    assert res.path == ["agent_node", "circuit_breaker_node", "tool_executor_node",
                        "agent_node", "circuit_breaker_node", "error_node"]
    assert res.values["iteration_count"] == 2
    assert "Repeated identical call" in res.answer
    assert_every_tool_call_answered(res.values["messages"])


def test_hash_is_normalized():
    a = cb.tool_call_hash("execute_wire_transfer", {"recipient_id": "V-8812", "amount_usd": 1500})
    b = cb.tool_call_hash("execute_wire_transfer", {"amount_usd": 1500.0, "recipient_id": " V-8812 "})
    c = cb.tool_call_hash("execute_wire_transfer", {"amount_usd": 1500.01, "recipient_id": "V-8812"})
    assert a == b != c
    assert len(a) == 64 and int(a, 16) >= 0  # SHA-256 hex digest
    assert str({"x": 1, "y": 2}) != str({"y": 2, "x": 1})  # why str(kwargs) was a bug


def test_check_unit():
    calls = [{"name": "query_service_metrics", "args": {"service_name": "auth-service"}}]
    ok = cb.check(calls, 1, [])
    assert not ok.tripped and len(ok.hashes) == 1
    assert cb.check(calls, 2, ok.hashes).tripped
    assert cb.check(calls, 6, []).tripped
    assert not cb.check(calls, 5, []).tripped


def test_breaker_history_resets_for_the_next_turn(scripted, thread_id):
    """The same call in a NEW user turn is not a loop: per-turn hash history starts empty."""
    respond = lambda m: (AIMessage("Still down.") if isinstance(m[-1], ToolMessage)
                         else AIMessage("", tool_calls=[call("fetch_incident_report", incident_id="INC-42")]))
    graph = build_graph(scripted(respond), TOOLS)
    send(graph, "Get INC-42", thread_id)
    res = send(graph, "Try INC-42 once more", thread_id)
    assert "tool_executor_node" in res.path and "error_node" not in res.path
    assert res.answer == "Still down."
