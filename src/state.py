"""Typed state schema for the graph.

Every node returns a *partial update* (a dict with only the keys it changes).
LangGraph merges it into the checkpointed state; nodes never mutate `state` in place.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

ApprovalStatus = Literal["APPROVED", "REJECTED"]


class AgentState(TypedDict, total=False):
    # Conversation history. `add_messages` appends new messages (and updates by id),
    # so the checkpointer keeps the full thread across turns and across approval pauses.
    messages: Annotated[list[AnyMessage], add_messages]
    # Number of agent (LLM) steps in the current user turn. Reset to 0 at the start of each turn.
    iteration_count: int
    # SHA-256 fingerprints of the tool calls made in the current turn, in order.
    executed_tool_hashes: list[str]
    # Fingerprints of sensitive calls the human rejected in this thread. A re-proposal is blocked.
    rejected_hashes: list[str]
    # The approval ticket while the graph is paused at the human gate.
    pending_approval: Optional[dict]
    # The human decision for the most recent ticket.
    approval_status: Optional[ApprovalStatus]
    # Why the circuit breaker stopped the run (for the UI and tests).
    halt_reason: Optional[str]
