"""Streamlit console for the state-graph ops agent.

Run locally:  uv run streamlit run src/app.py
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
try:  # Streamlit Cloud secrets win over .env
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"]).strip()
except Exception:
    pass

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from src import tools as T  # noqa: E402
from src.circuit_breaker import MAX_ITERATIONS  # noqa: E402
from src.graph import NODE_LABELS, build_graph, config_for, is_paused, openai_llm, resume, send  # noqa: E402

st.set_page_config(page_title="State-Graph Ops Agent", page_icon="🧭", layout="wide")

SCENARIO = ("Check the current error rate on the auth-service. If it's over 5%, scale its replicas to 10 and "
            "transfer $1,500 to the contracted on-call engineering firm (Vendor ID: V-8812) for emergency mitigation.")
EXAMPLES = {
    "📊 Safe: auth-service metrics": "What is the current error rate on the auth-service?",
    "🧠 Memory 1: my name": "Hi, my name is Alice and I'm the on-call operator today.",
    "🧠 Memory 2: recall": "What is my name, and which service did I ask about?",
    "🚨 HITL: full ops scenario": SCENARIO,
    "🔁 Loop test (turn on the toggle)": "Fetch the incident report for INC-42. If it fails, keep retrying until it works.",
}


@st.cache_resource
def checkpointer() -> MemorySaver:
    """One in-memory checkpointer for the whole app, so paused threads survive Streamlit reruns."""
    return MemorySaver()


def get_graph(model: str, with_failing_tool: bool):
    tools = [*T.ALL_TOOLS, T.FAILING_TOOL] if with_failing_tool else T.ALL_TOOLS
    return build_graph(openai_llm(model), tools, checkpointer())


def new_thread_id() -> str:
    return f"ops-{uuid.uuid4().hex[:6]}"


def msg_to_dict(m) -> dict:
    d = {"type": m.type, "content": m.content}
    if isinstance(m, AIMessage) and m.tool_calls:
        d["tool_calls"] = [{"name": c["name"], "args": c["args"], "id": c["id"]} for c in m.tool_calls]
    if isinstance(m, ToolMessage):
        d.update(tool_call_id=m.tool_call_id, name=m.name, status=getattr(m, "status", None))
    return d


def fmt_params(tool: str, args: dict) -> str:
    if tool == "execute_wire_transfer":
        return f"Recipient: **{args.get('recipient_id')}** · Amount: **${float(args.get('amount_usd', 0)):,.2f}**"
    if tool == "scale_kubernetes_cluster":
        return f"Service: **{args.get('service_name')}** · Replicas: **{args.get('replica_count')}**"
    return ", ".join(f"{k}: **{v}**" for k, v in args.items())


has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
st.session_state.setdefault("thread_id", new_thread_id())
st.session_state.setdefault("path", [])
st.session_state.setdefault("error", None)

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Session")
    st.caption(("✅" if has_key else "❌") + " OpenAI key · checkpointer: **MemorySaver**")
    tid = st.text_input("thread_id", value=st.session_state.thread_id,
                        help="Each thread has its own memory. Reuse an id to continue a conversation.")
    st.session_state.thread_id = tid.strip() or st.session_state.thread_id
    c1, c2 = st.columns(2)
    if c1.button("➕ New thread", width="stretch"):
        st.session_state.thread_id, st.session_state.path = new_thread_id(), []
        st.rerun()
    if c2.button("♻️ Reset systems", width="stretch", help="Restore replicas and balances, clear the execution log"):
        T.reset_systems()
        st.rerun()
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    loop_test = st.toggle("🔁 Circuit breaker test: add failing tool",
                          help="Adds fetch_incident_report, which always times out, to prove the breaker stops the loop.")

graph = get_graph(model, loop_test) if has_key else None
cfg = config_for(st.session_state.thread_id)
snap = graph.get_state(cfg) if graph else None
values = dict(snap.values) if snap else {}
paused = bool(graph) and is_paused(graph, st.session_state.thread_id)

with st.sidebar:
    st.divider()
    st.header("🔎 Graph state")
    path = st.session_state.path
    st.markdown("**Active path (last run)**")
    st.code(" → ".join(f"[{NODE_LABELS.get(p, p)}]" for p in path) or "(nothing run yet)", language=None, wrap_lines=True)
    it = values.get("iteration_count", 0)
    m1, m2 = st.columns(2)
    m1.metric("Iterations", f"{it} / {MAX_ITERATIONS}")
    m2.metric("Next node", NODE_LABELS.get(snap.next[0], snap.next[0]).replace("_", " ") if snap and snap.next else "END")
    st.progress(min(it / MAX_ITERATIONS, 1.0))
    if values.get("halt_reason"):
        st.error(f"Circuit breaker: {values['halt_reason']}")
    with st.expander("🧾 Checkpointed state (JSON)"):
        st.json({**{k: v for k, v in values.items() if k != "messages"},
                 "messages": [msg_to_dict(m) for m in values.get("messages", [])]}, expanded=False)

# --------------------------------------------------------------------------- main
st.title("🧭 State-Graph Ops Agent")
st.caption("ReAct agent on a LangGraph state machine · short-term memory per thread · "
           f"max {MAX_ITERATIONS} iterations · SHA-256 repeat breaker · human approval for sensitive tools")

if not has_key:
    st.warning("Add OPENAI_API_KEY to .env (local) or Streamlit Secrets (cloud) to run the agent.")


def run(fn, *args):
    try:
        res = fn(graph, *args, st.session_state.thread_id)
        before = st.session_state.path if fn is resume else []  # show the whole turn across the pause
        st.session_state.path = before + res.path + (["__interrupt__"] if res.paused else [])
        st.session_state.error = None
    except Exception as exc:  # show a clean message, never a stack trace
        st.session_state.error = f"{type(exc).__name__}: {str(exc)[:300]}"
    st.rerun()


st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES.items()):
    if col.button(label, width="stretch", disabled=paused or not has_key):
        run(send, text)

if st.session_state.error:
    st.error(st.session_state.error)

# transcript
for m in values.get("messages", []):
    if isinstance(m, HumanMessage):
        st.chat_message("user").markdown(m.content)
    elif isinstance(m, AIMessage):
        with st.chat_message("assistant"):
            if m.content:
                st.markdown(m.content)
            for c in m.tool_calls:
                st.caption(f"🛠️ wants to call `{c['name']}` with `{json.dumps(c['args'])}`")
    elif isinstance(m, ToolMessage):
        icon = "⛔" if getattr(m, "status", "") == "error" else "📦"
        with st.expander(f"{icon} {m.name} result", expanded=False):
            try:
                st.json(json.loads(m.content))
            except ValueError:
                st.write(m.content)

# approval card
if paused:
    ticket = next((i.value for t in snap.tasks for i in (t.interrupts or ())), None) or values.get("pending_approval") or {}
    with st.container(border=True):
        st.subheader("✋ Human approval required")
        st.caption("The graph is paused at `human_approval_gate`. Its state is saved in the checkpointer "
                   f"under thread `{st.session_state.thread_id}`.")
        for a in ticket.get("actions", []):
            st.markdown(f"**Requested action:** `{a['tool']}`  \n**Parameters:** {fmt_params(a['tool'], a['args'])}")
            (st.error if a["risk"].startswith("HIGH") else st.warning)(f"Risk classification: {a['risk']}")
        if ticket.get("also_in_batch"):
            st.caption("Also in this step (read-only): " + ", ".join(ticket["also_in_batch"]))
        b1, b2 = st.columns(2)
        if b1.button("✅ Approve action", type="primary", width="stretch"):
            run(resume, "APPROVED")
        if b2.button("❌ Reject action", width="stretch"):
            run(resume, "REJECTED")

prompt = st.chat_input("Ask the ops agent…" if not paused else "Approve or reject the pending action first",
                       disabled=paused or not has_key)
if prompt:
    run(send, prompt)

# simulated systems
with st.expander("🏗️ Simulated systems & execution log", expanded=bool(T.EXECUTION_LOG)):
    c1, c2 = st.columns(2)
    c1.markdown("**Services**")
    c1.dataframe([{"service": k, **v} for k, v in T.SERVICES.items()], hide_index=True, width="stretch")
    c2.markdown("**Accounts**")
    c2.dataframe([{"account": k, **v} for k, v in T.ACCOUNTS.items()], hide_index=True, width="stretch")
    st.markdown("**Sensitive actions that actually ran**")
    st.dataframe(T.EXECUTION_LOG or [{"action": "(none yet)"}], hide_index=True, width="stretch")
