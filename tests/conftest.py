"""Shared fixtures. Tests use a scripted chat model so they need no API key and are deterministic;
the graph, checkpointer, interrupt/resume and tools are all real."""

from __future__ import annotations

import itertools
import uuid
from pathlib import Path
from typing import Callable

import pytest
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_ids = itertools.count(1)


def call(name: str, **args) -> dict:
    return {"name": name, "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}


class ScriptedLLM(BaseChatModel):
    """A fake chat model: `respond(messages) -> AIMessage`. Records every prompt it receives."""

    respond: Callable[[list[BaseMessage]], AIMessage]
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.respond(list(messages)))])


@pytest.fixture
def scripted():
    def make(respond):
        return ScriptedLLM(respond=respond, seen=[])
    return make


@pytest.fixture
def thread_id():
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def _fresh_systems():
    from src.tools import reset_systems
    reset_systems()
    yield
    reset_systems()
