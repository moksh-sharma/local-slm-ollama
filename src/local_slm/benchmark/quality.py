"""Deterministic quality checks without a judge LLM."""

from __future__ import annotations

import json
import re
from typing import Any

from local_slm.models import QualityTaskResult


def _check_contains(text: str, value: str) -> tuple[bool, str]:
    ok = value.lower() in text.lower()
    return ok, f"contains '{value}'" if ok else f"missing '{value}'"


def _check_contains_any(text: str, values: list[str]) -> tuple[bool, str]:
    lowered = text.lower()
    for value in values:
        if value.lower() in lowered:
            return True, f"matched '{value}'"
    return False, f"none of {values}"


def _check_regex(text: str, pattern: str, flags: str = "") -> tuple[bool, str]:
    flag_bits = re.MULTILINE if "multiline" in flags.lower() else 0
    ok = bool(re.search(pattern, text, flag_bits))
    return ok, f"regex /{pattern}/" + (" matched" if ok else " failed")


def _check_json_has(text: str, keys: list[str]) -> tuple[bool, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return False, "no JSON object found"
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc}"
    missing = [k for k in keys if k not in obj]
    if missing:
        return False, f"missing keys: {missing}"
    return True, f"json has keys {keys}"


def evaluate_task(task: dict[str, Any], response: str) -> QualityTaskResult:
    checks: list[dict[str, Any]] = task.get("checks", [])
    details: list[str] = []
    passed = True

    for check in checks:
        kind = check.get("type")
        if kind == "contains":
            ok, msg = _check_contains(response, check["value"])
        elif kind == "contains_any":
            ok, msg = _check_contains_any(response, check["values"])
        elif kind == "regex":
            ok, msg = _check_regex(response, check["pattern"], check.get("flags", ""))
        elif kind == "json_has":
            ok, msg = _check_json_has(response, check["keys"])
        else:
            ok, msg = False, f"unknown check type: {kind}"
        details.append(msg)
        passed = passed and ok

    return QualityTaskResult(
        task_id=task["id"],
        category=task.get("category", "general"),
        passed=passed,
        response=response.strip(),
        details="; ".join(details),
    )
