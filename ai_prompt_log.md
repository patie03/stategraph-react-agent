# AI Prompt Log & Code Audit

**Student:** Patience Kalanga
**AI assistant used:** Claude (Anthropic), in a chat-based coding session
**Date:** November 2026

## 1. Prompt used

Adapted from the course's sample prompt:

```text
You are a Principal AI Agent Architect specializing in deterministic state machines and LangGraph.
Build a production-grade, state-graph-governed ReAct agent in Python 3.11+ featuring memory persistence,
cycle breaking, and human-in-the-loop (HITL) approval gates.

Agent State (src/state.py): AgentState(TypedDict) with messages: Annotated[list[AnyMessage], add_messages],
iteration_count: int, executed_tool_hashes: list[str], pending_approval: Optional[dict],
approval_status: Optional[Literal["APPROVED", "REJECTED"]].

Tools (src/tools.py): safe query_service_metrics, check_billing_balance; sensitive scale_kubernetes_cluster,
execute_wire_transfer tagged with metadata is_sensitive=True.

Graph (src/graph.py): agent_node, circuit_breaker_node (iteration_count <= 6, no consecutive repeated
SHA-256 hash of (tool_name, args)), human_approval_gate (dynamic interrupt() with MemorySaver),
tool_executor_node, error_node. Conditional edges as in the assignment. Compile with MemorySaver and
thread_id configuration.

My additions:
 - Simulated systems only (fake services, accounts and vendors). Nothing real is touched.
 - Tests must run without an API key: use a scripted chat model, but the real graph, checkpointer and interrupt().
 - A Streamlit UI with a state inspector, an approval card and a circuit-breaker test toggle.
```

## 2. Audit checklist from the assignment

| Risk | What I checked | Result |
|---|---|---|
| **State mutation vs. pure returns** | Any `state["messages"].append(...)` or other in-place edits? | ✅ None. Every node returns a partial dict, e.g. `return {"messages": [reply], "iteration_count": state.get("iteration_count", 0) + 1}`. The `add_messages` reducer appends, and the checkpointer saves each update. |
| **Infinite loop on human rejection** | After "Reject", does the model ask for the same action again, creating an endless approval cycle? | ✅ Two layers. (1) The rejection `ToolMessage` says "REJECTED BY HUMAN OPERATOR … Do NOT call this action again and do not ask for approval again". (2) The rejected call's fingerprint is stored in `rejected_hashes`; if a stubborn model proposes it again, `circuit_breaker_node` routes to `error_node` instead of the gate. A test uses a scripted "stubborn" model to prove it. |
| **Payload hashing normalization** | `str(kwargs)` hashing? | ✅ `json.dumps(args, sort_keys=True, separators=(",", ":"))`, plus whole-number floats folded (`1500` and `1500.0` hash the same) and strings trimmed. A test shows `str({"x": 1, "y": 2}) != str({"y": 2, "x": 1})` but the normalized hashes match. |
| **Thread ID & session isolation** | A hardcoded global `thread_id`? | ✅ Each browser session gets `ops-<random>`; there's a "New thread" button and an editable field. A test checks that a second thread can't see the first thread's memory. |
| `interrupt()` re-execution | On resume, LangGraph re-runs the gate node from the top. Anything with side effects before `interrupt()` would run twice. | ✅ The gate does nothing before `interrupt()`; the ticket is built earlier, in `circuit_breaker_node`. |

## 3. Bugs found (before → after)

### Bug 1: The tools module crashed on import
The first draft tagged tools like this:

**Before**
```python
@tool(metadata={"is_sensitive": True, "risk": "HIGH (Financial Transaction)"})
def execute_wire_transfer(recipient_id: str, amount_usd: float) -> dict:
```
```text
TypeError: tool() got an unexpected keyword argument 'metadata'
```
LangChain's `@tool` decorator doesn't take `metadata`, even though `BaseTool` has a `metadata` field. The draft assumed the two worked the same.

**After**
```python
def tagged(is_sensitive: bool, risk: str):
    """@tool plus metadata: {"is_sensitive": ..., "risk": ...} on the resulting BaseTool."""
    def wrap(fn):
        t = tool(fn)
        t.metadata = {"is_sensitive": is_sensitive, "risk": risk}
        return t
    return wrap

@tagged(True, "HIGH (Financial Transaction)")
def execute_wire_transfer(recipient_id: str, amount_usd: float) -> dict:
```

### Bug 2: A circuit-breaker halt corrupted the conversation thread
The design in the prompt says `error_node` "emits graceful termination messages". Implemented literally, it only adds a final `AIMessage`:

**Before**
```python
def error_node(state):
    return {"messages": [AIMessage(f"Stopped: {state['halt_reason']}")]}
```
But the previous `AIMessage` still contains `tool_calls` that never got a `ToolMessage`. Because the checkpointer saves that history, the **next** user message on the same thread sends an invalid conversation to OpenAI ("an assistant message with tool_calls must be followed by tool messages…"), and the thread is stuck.

**After**
```python
def error_node(state):
    reason = state.get("halt_reason") or "Unknown error."
    last = _last_ai(state)
    closing = [ToolMessage(f"NOT RUN: halted by circuit breaker. {reason}", tool_call_id=c["id"],
                           name=c["name"], status="error") for c in last.tool_calls]
    final = AIMessage(f"⛔ I stopped this request safely. {reason} No further actions were taken. ...")
    return {"messages": [*closing, final], "pending_approval": None}
```
The same rule is applied on the rejection path (every call in the batch gets a `ToolMessage`). The tests `assert_every_tool_call_answered` after both breaker trips.

### Bug 3: `iteration_count` never reset between user turns
With a checkpointer, every state field persists across turns, not just `messages`. The draft only sent the new message:

**Before**
```python
graph.invoke({"messages": [HumanMessage(text)]}, config)
```
So `iteration_count` and `executed_tool_hashes` kept growing. After about 6 total tool steps in a conversation, every new question was halted immediately, and asking about the same service again in a later turn looked like a "repeated call".

**After**
```python
def new_turn(text: str) -> dict:
    """Input for a new user message. Per-turn counters reset; the message history is kept by the checkpointer."""
    return {"messages": [HumanMessage(text)], "iteration_count": 0, "executed_tool_hashes": [],
            "rejected_hashes": [], "halt_reason": None, "approval_status": None, "pending_approval": None}
```
`messages` uses the `add_messages` reducer, so it still appends; the other fields have no reducer, so they are replaced. Tests: `test_per_turn_counters_reset_but_history_is_kept`, `test_breaker_history_resets_for_the_next_turn`.

### Bug 4 (minor): Wrong wire-transfer confirmation number
Found while testing the approval flow in the UI: the first transfer got confirmation `WT-00002`, because the number was based on the length of the whole execution log, which already contained the scaling action.

**Before**
```python
confirmation = f"WT-{len(EXECUTION_LOG) + 1:05d}"
```
**After**
```python
confirmation = f"WT-{sum(e['action'] == 'execute_wire_transfer' for e in EXECUTION_LOG) + 1:05d}"
```

## 4. Known limitations (honest notes)
- `MemorySaver` keeps threads in memory. If the Streamlit app restarts, paused approvals and memory are lost. `SqliteSaver` (or Postgres) would make them durable.
- The approval decision covers the whole batch: if the model asks to scale *and* transfer in one step, the operator approves or rejects both together.
- The tests use a scripted model for determinism; the live gpt-4o-mini tests run only when a key is present.

## 5. My verification (tick after doing it myself)
- [ ] `uv run pytest -v`: all tests pass (live tests run with my key)
- [ ] A safe question runs to END without the approval card
- [ ] "My name is Alice" → "What is my name?" remembers on the same thread
- [ ] The full scenario pauses; Approve runs both actions; Reject cancels them without asking again
- [ ] The loop test toggle trips the circuit breaker
