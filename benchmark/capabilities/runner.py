"""Invocation + deterministic scoring for the capability benchmark.

Reuses the proxy plumbing from ``benchmark.shared`` (``MODELS`` registry and
``resolve_proxy_credentials``) so models resolve exactly as the rest of the
benchmark suite. No LLM judge — every score is computed against ground truth.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel, ValidationError

from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError

from benchmark.shared import MODELS, resolve_proxy_credentials
from benchmark.capabilities.tasks import (
    ToolTask, StructTask, TOOL_TASKS, STRUCT_TASKS, args_match, norm,
)

# Native function-calling is the broadly-supported path through LiteLLM; some
# models also support "json_schema". Override with CAPABILITY_STRUCT_METHOD.
STRUCT_METHOD = os.environ.get("CAPABILITY_STRUCT_METHOD", "function_calling")
REQUEST_TIMEOUT = int(os.environ.get("CAPABILITY_TIMEOUT", "120"))


def build_client(model_name: str) -> ChatOpenAI:
    """Construct a ChatOpenAI bound to the proxy for ``model_name``."""
    cfg = MODELS.get(model_name, {"type": "litellm", "model": model_name})
    model_id = cfg.get("model", model_name)
    key, base_url = resolve_proxy_credentials()
    return ChatOpenAI(
        model=model_id,
        api_key=key,
        base_url=base_url,
        timeout=REQUEST_TIMEOUT,
        max_retries=1,
        temperature=0,
    )


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(RateLimitError),
)
def _invoke(runnable, prompt: str):
    return runnable.invoke(prompt)


# ===========================================================================
# Tool-calling
# ===========================================================================

def run_tool_task(client: ChatOpenAI, task: ToolTask) -> Dict[str, Any]:
    """Run one tool task. Returns a per-task result record."""
    rec: Dict[str, Any] = {
        "id": task.id, "category": task.category,
        "expected_tool": task.expected_tool,
        "valid_call": False, "selection_ok": False, "args_ok": False,
        "called_tool": None, "called_args": None, "error": None,
    }
    try:
        bound = client.bind_tools(task.tools)
        resp = _invoke(bound, task.prompt)
    except Exception as e:  # noqa: BLE001 — record, don't crash the sweep
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    calls = getattr(resp, "tool_calls", None) or []

    if task.expected_tool is None:
        # Abstention: success == no tool call fabricated. "valid_call" here means
        # the model produced a well-formed (empty) tool decision.
        rec["valid_call"] = True
        rec["selection_ok"] = len(calls) == 0
        rec["args_ok"] = rec["selection_ok"]
        if calls:
            rec["called_tool"] = calls[0].get("name")
            rec["called_args"] = calls[0].get("args")
        return rec

    if not calls:
        return rec  # expected a call, got none
    rec["valid_call"] = True
    first = calls[0]
    rec["called_tool"] = first.get("name")
    rec["called_args"] = first.get("args")
    # selection: the right tool appears anywhere in the call list
    rec["selection_ok"] = any(c.get("name") == task.expected_tool for c in calls)
    if rec["selection_ok"]:
        match = next(c for c in calls if c.get("name") == task.expected_tool)
        rec["args_ok"] = args_match(task.expected_args, match.get("args") or {})
    return rec


# ===========================================================================
# Structured output
# ===========================================================================

def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return {}


def compare_leaves(expected: Any, actual: Any) -> Tuple[int, int]:
    """Count (matched, total) leaf values, recursing into dicts/lists."""
    if isinstance(expected, dict):
        m = t = 0
        act = actual if isinstance(actual, dict) else {}
        for k, v in expected.items():
            dm, dt = compare_leaves(v, act.get(k))
            m += dm
            t += dt
        return m, t
    if isinstance(expected, list):
        m = t = 0
        act = actual if isinstance(actual, list) else []
        for i, v in enumerate(expected):
            dm, dt = compare_leaves(v, act[i] if i < len(act) else None)
            m += dm
            t += dt
        return m, t
    # leaf
    return (1 if norm(expected) == norm(actual) else 0), 1


def run_struct_task(client: ChatOpenAI, task: StructTask) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "id": task.id, "category": task.category,
        "schema_valid": False, "field_matched": 0, "field_total": 0,
        "exact_match": False, "got": None, "error": None,
    }
    try:
        structured = client.with_structured_output(task.schema, method=STRUCT_METHOD)
        out = _invoke(structured, task.prompt)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        # count all expected leaves as missed so field_accuracy is fair
        _, total = compare_leaves(task.expected, {})
        rec["field_total"] = total
        return rec

    rec["schema_valid"] = True
    got = _to_dict(out)
    rec["got"] = got
    matched, total = compare_leaves(task.expected, got)
    rec["field_matched"] = matched
    rec["field_total"] = total
    rec["exact_match"] = matched == total and total > 0
    return rec


# ===========================================================================
# Prompt-based path (no native tool / structured-output API)
# ===========================================================================
# For models the proxy doesn't expose function-calling for, we fall back to
# plain prompting + manual JSON parsing. This measures instruction-following
# JSON adherence rather than native API support — a strictly easier bar, so
# differences between the two modes are themselves informative.

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Any:
    """Best-effort: strip ``` fences and parse the first {...} block."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)
    m = _JSON_RE.search(cleaned)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None


def _describe_tools(tools: List[Type[BaseModel]]) -> str:
    lines = []
    for cls in tools:
        fields = ", ".join(
            f"{n} ({f.description or 'value'})" for n, f in cls.model_fields.items()
        )
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        lines.append(f"- {cls.__name__}: {doc} | args: {fields}")
    return "\n".join(lines)


def run_tool_task_prompt(client: ChatOpenAI, task: ToolTask) -> Dict[str, Any]:
    """Prompt-based tool selection: ask for a JSON tool decision, parse it."""
    rec: Dict[str, Any] = {
        "id": task.id, "category": task.category, "expected_tool": task.expected_tool,
        "valid_call": False, "selection_ok": False, "args_ok": False,
        "called_tool": None, "called_args": None, "error": None,
    }
    prompt = (
        "You have these tools:\n" + _describe_tools(task.tools) + "\n\n"
        "Decide whether a tool is needed for the request below. Reply with ONLY a "
        'JSON object: {"tool": "<ToolName or null>", "args": {<arg>: <value>, ...}}. '
        'If no tool fits or a required argument is missing, use {"tool": null, "args": {}}.\n\n'
        f"Request: {task.prompt}"
    )
    try:
        resp = _invoke(client, prompt)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    parsed = _extract_json(getattr(resp, "content", "") or str(resp))
    if not isinstance(parsed, dict) or "tool" not in parsed:
        return rec  # unparseable / no decision
    rec["valid_call"] = True
    tool = parsed.get("tool")
    tool = None if tool in (None, "null", "", "None") else tool
    rec["called_tool"] = tool
    rec["called_args"] = parsed.get("args")

    if task.expected_tool is None:
        rec["selection_ok"] = tool is None
        rec["args_ok"] = rec["selection_ok"]
        return rec
    rec["selection_ok"] = tool == task.expected_tool
    if rec["selection_ok"]:
        rec["args_ok"] = args_match(task.expected_args, parsed.get("args") or {})
    return rec


def run_struct_task_prompt(client: ChatOpenAI, task: StructTask) -> Dict[str, Any]:
    """Prompt-based structured output: ask for schema-conforming JSON, validate."""
    rec: Dict[str, Any] = {
        "id": task.id, "category": task.category,
        "schema_valid": False, "field_matched": 0, "field_total": 0,
        "exact_match": False, "got": None, "error": None,
    }
    _, total = compare_leaves(task.expected, {})
    rec["field_total"] = total
    schema_json = json.dumps(task.schema.model_json_schema(), indent=2)
    prompt = (
        "Extract the information into JSON that conforms EXACTLY to this JSON Schema "
        "(reply with ONLY the JSON object, no prose, no code fences):\n"
        f"{schema_json}\n\n{task.prompt}"
    )
    try:
        resp = _invoke(client, prompt)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    parsed = _extract_json(getattr(resp, "content", "") or str(resp))
    if not isinstance(parsed, dict):
        rec["error"] = "unparseable_json"
        return rec
    try:
        validated = task.schema.model_validate(parsed)  # schema conformance check
    except ValidationError as e:
        rec["error"] = f"schema_invalid: {e.error_count()} errors"
        rec["got"] = parsed
        return rec
    rec["schema_valid"] = True
    got = validated.model_dump()
    rec["got"] = got
    matched, _ = compare_leaves(task.expected, got)
    rec["field_matched"] = matched
    rec["exact_match"] = matched == total and total > 0
    return rec


# ===========================================================================
# Per-model driver + aggregation
# ===========================================================================

def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def run_model(model_name: str, mode: str = "native") -> Dict[str, Any]:
    """Run all tasks for one model in the given mode (``native`` | ``prompt``)."""
    client = build_client(model_name)
    t0 = time.time()

    if mode == "prompt":
        tool_recs = [run_tool_task_prompt(client, t) for t in TOOL_TASKS]
        struct_recs = [run_struct_task_prompt(client, s) for s in STRUCT_TASKS]
    else:
        tool_recs = [run_tool_task(client, t) for t in TOOL_TASKS]
        struct_recs = [run_struct_task(client, s) for s in STRUCT_TASKS]

    n_tool = len(tool_recs)
    n_struct = len(struct_recs)

    tool_score = {
        "valid_call_rate": _rate(sum(r["valid_call"] for r in tool_recs), n_tool),
        "selection_rate": _rate(sum(r["selection_ok"] for r in tool_recs), n_tool),
        "args_rate": _rate(sum(r["args_ok"] for r in tool_recs), n_tool),
        "abstention_rate": _rate(
            sum(r["selection_ok"] for r in tool_recs if r["expected_tool"] is None),
            sum(1 for r in tool_recs if r["expected_tool"] is None),
        ),
    }
    field_matched = sum(r["field_matched"] for r in struct_recs)
    field_total = sum(r["field_total"] for r in struct_recs)
    struct_score = {
        "schema_valid_rate": _rate(sum(r["schema_valid"] for r in struct_recs), n_struct),
        "field_accuracy": _rate(field_matched, field_total),
        "exact_match_rate": _rate(sum(r["exact_match"] for r in struct_recs), n_struct),
    }
    # Single headline number per family (arg correctness is the bar for tools;
    # field accuracy is the bar for structured output).
    overall = round((tool_score["args_rate"] + struct_score["field_accuracy"]) / 2, 4)

    return {
        "model": model_name,
        "mode": mode,
        "label": f"{model_name}[{mode}]",
        "duration_s": round(time.time() - t0, 1),
        "tool": tool_score,
        "struct": struct_score,
        "overall": overall,
        "tool_detail": tool_recs,
        "struct_detail": struct_recs,
    }
