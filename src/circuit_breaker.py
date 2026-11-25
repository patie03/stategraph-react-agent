"""Loop protection: an iteration cap and a SHA-256 repeat detector for tool calls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

MAX_ITERATIONS = 6


def normalize_args(args: Any) -> str:
    """Canonical JSON for tool arguments.

    `str(dict)` depends on key order and Python's repr ("1500" vs 1500.0, spaces), so two
    identical calls can hash differently. Sorted keys + compact separators fix that, and
    whole-number floats are folded to ints so {"amount": 1500} == {"amount": 1500.0}.
    """
    def fold(v: Any) -> Any:
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, dict):
            return {str(k): fold(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [fold(x) for x in v]
        if isinstance(v, str):
            return v.strip()
        return v
    return json.dumps(fold(args or {}), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tool_call_hash(name: str, args: Any) -> str:
    payload = f"{name.strip()}|{normalize_args(args)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class BreakerVerdict:
    tripped: bool
    reason: Optional[str]
    hashes: list[str]  # updated hash history to store in state


def check(tool_calls: Iterable[dict], iteration_count: int, history: list[str],
          rejected: Iterable[str] = (), max_iterations: int = MAX_ITERATIONS) -> BreakerVerdict:
    """Decide whether the proposed tool calls may proceed."""
    hashes = list(history)
    if iteration_count >= max_iterations:
        return BreakerVerdict(True, f"Iteration limit reached ({iteration_count}/{max_iterations}). "
                                    "Stopping to avoid a runaway loop.", hashes)
    rejected = set(rejected)
    for call in tool_calls:
        h = tool_call_hash(call["name"], call.get("args", {}))
        if h in rejected:
            return BreakerVerdict(True, f"The operator already rejected {call['name']} with these "
                                        "exact parameters. Not asking again.", hashes)
        if hashes and hashes[-1] == h:
            return BreakerVerdict(True, f"Repeated identical call to {call['name']} detected "
                                        f"(SHA-256 {h[:12]}…). Stopping the loop.", hashes)
        hashes.append(h)
    return BreakerVerdict(False, None, hashes[-20:])
