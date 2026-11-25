"""Tools for the Cloud & Financial Operations agent.

All data is simulated (no real cloud or bank is touched). Each tool carries
`metadata={"is_sensitive": bool, "risk": ...}`; the graph uses that flag to decide
whether a call can run on its own or must stop at the human approval gate.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime, timezone

from langchain_core.tools import BaseTool, tool

# --------------------------------------------------------------------------- simulated systems
SERVICES = {
    "auth-service": {"error_rate_pct": 7.2, "p95_latency_ms": 840, "replicas": 4, "status": "degraded"},
    "payments-api": {"error_rate_pct": 1.1, "p95_latency_ms": 210, "replicas": 6, "status": "healthy"},
    "search-service": {"error_rate_pct": 0.4, "p95_latency_ms": 120, "replicas": 3, "status": "healthy"},
    "notification-service": {"error_rate_pct": 5.8, "p95_latency_ms": 460, "replicas": 2, "status": "degraded"},
}
ACCOUNTS = {
    "ACCT-1001": {"name": "Operations budget", "balance_usd": 48250.00},
    "ACCT-2002": {"name": "Cloud spend", "balance_usd": 12980.50},
    "ACCT-3003": {"name": "Vendor payments", "balance_usd": 7400.00},
}
VENDORS = {"V-8812": "Northwind On-Call Engineering", "V-4410": "Contoso Security Partners"}
PAYING_ACCOUNT = "ACCT-3003"
MAX_REPLICAS = 50
MAX_TRANSFER_USD = 25_000

_lock = threading.Lock()
EXECUTION_LOG: list[dict] = []  # every sensitive action that actually ran


def _log(action: str, details: str) -> None:
    with _lock:
        EXECUTION_LOG.append({"time": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                              "action": action, "details": details})


def reset_systems() -> None:
    """Restore the simulated systems (used by tests and the UI's reset button)."""
    with _lock:
        SERVICES["auth-service"].update(replicas=4)
        SERVICES["payments-api"].update(replicas=6)
        SERVICES["search-service"].update(replicas=3)
        SERVICES["notification-service"].update(replicas=2)
        ACCOUNTS["ACCT-3003"]["balance_usd"] = 7400.00
        EXECUTION_LOG.clear()


def tagged(is_sensitive: bool, risk: str):
    """@tool plus metadata: {"is_sensitive": ..., "risk": ...} on the resulting BaseTool."""
    def wrap(fn):
        t = tool(fn)
        t.metadata = {"is_sensitive": is_sensitive, "risk": risk}
        return t
    return wrap


def _service(name: str) -> dict:
    key = name.strip().lower()
    if key not in SERVICES:
        raise ValueError(f"Unknown service '{name}'. Known services: {', '.join(SERVICES)}")
    return SERVICES[key]


# --------------------------------------------------------------------------- safe (read-only)
@tagged(False, "LOW (Read-only)")
def query_service_metrics(service_name: str) -> dict:
    """Read live health metrics for a service: error rate (%), p95 latency (ms), replica count and status.
    Read-only. Known services: auth-service, payments-api, search-service, notification-service."""
    s = _service(service_name)
    return {"service": service_name.strip().lower(), **s}


@tagged(False, "LOW (Read-only)")
def check_billing_balance(account_id: str) -> dict:
    """Read the current balance of a billing account (e.g. ACCT-1001, ACCT-2002, ACCT-3003). Read-only.
    Vendor wire transfers are paid from ACCT-3003."""
    key = account_id.strip().upper()
    if key not in ACCOUNTS:
        raise ValueError(f"Unknown account '{account_id}'. Known accounts: {', '.join(ACCOUNTS)}")
    return {"account_id": key, **ACCOUNTS[key]}


# --------------------------------------------------------------------------- sensitive (mutations)
@tagged(True, "MEDIUM (Infrastructure Mutation)")
def scale_kubernetes_cluster(service_name: str, replica_count: int) -> dict:
    """SENSITIVE: change the number of Kubernetes replicas for a service (1-50).
    Requires human approval before it runs."""
    if not 1 <= int(replica_count) <= MAX_REPLICAS:
        raise ValueError(f"replica_count must be between 1 and {MAX_REPLICAS}")
    s = _service(service_name)
    with _lock:
        before = s["replicas"]
        s["replicas"] = int(replica_count)
    _log("scale_kubernetes_cluster", f"{service_name}: {before} → {int(replica_count)} replicas")
    return {"service": service_name.strip().lower(), "replicas_before": before,
            "replicas_after": int(replica_count), "status": "scaled"}


@tagged(True, "HIGH (Financial Transaction)")
def execute_wire_transfer(recipient_id: str, amount_usd: float) -> dict:
    """SENSITIVE: send a wire transfer in USD to an approved vendor ID (format V-1234), paid from ACCT-3003.
    Requires human approval before it runs."""
    rid = recipient_id.strip().upper()
    if not re.fullmatch(r"V-\d{4}", rid) or rid not in VENDORS:
        raise ValueError(f"Unknown or invalid vendor ID '{recipient_id}'")
    amount = round(float(amount_usd), 2)
    if not 0 < amount <= MAX_TRANSFER_USD:
        raise ValueError(f"amount_usd must be between 0 and {MAX_TRANSFER_USD:,}")
    with _lock:
        acct = ACCOUNTS[PAYING_ACCOUNT]
        if acct["balance_usd"] < amount:
            raise ValueError("Insufficient funds in ACCT-3003")
        acct["balance_usd"] = round(acct["balance_usd"] - amount, 2)
    confirmation = f"WT-{sum(e['action'] == 'execute_wire_transfer' for e in EXECUTION_LOG) + 1:05d}"
    _log("execute_wire_transfer", f"${amount:,.2f} to {rid} ({VENDORS[rid]}) · {confirmation}")
    return {"recipient_id": rid, "vendor": VENDORS[rid], "amount_usd": amount,
            "confirmation": confirmation, "status": "sent", "remaining_balance_usd": acct["balance_usd"]}


# --------------------------------------------------------------------------- circuit-breaker demo
@tagged(False, "LOW (Read-only)")
def fetch_incident_report(incident_id: str) -> dict:
    """Fetch the incident report for an incident ID like INC-42."""
    raise TimeoutError("Incident service timed out (simulated outage). Try again later.")


SAFE_TOOLS: list[BaseTool] = [query_service_metrics, check_billing_balance]
SENSITIVE_TOOLS: list[BaseTool] = [scale_kubernetes_cluster, execute_wire_transfer]
ALL_TOOLS: list[BaseTool] = SAFE_TOOLS + SENSITIVE_TOOLS
FAILING_TOOL: BaseTool = fetch_incident_report


def is_sensitive(tool_obj: BaseTool) -> bool:
    return bool((tool_obj.metadata or {}).get("is_sensitive"))


def risk_label(tool_obj: BaseTool) -> str:
    return (tool_obj.metadata or {}).get("risk", "UNKNOWN")
