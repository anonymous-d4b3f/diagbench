"""
M4-T1: Solver action schema parser.
V2: Parses and validates the four allowed solver actions:
  - propose_design
  - declare_infeasible
  - request_missing_info
  - replan  (V2 new: model identifies dead-end and switches strategy)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
from diagbench.solver.response_json import extract_first_json_object

_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "action_schema.json"


def _load_schema() -> dict:
    with _SCHEMA_PATH.open() as fh:
        return json.load(fh)


SYSTEM_PROMPT_EXTENSION: str = (
    "You MUST reply with ONLY a JSON object. No explanation, no markdown, no text before or after.\n"
    "Return the shortest valid JSON object. Omit optional fields unless they are strictly necessary.\n"
    "For propose_design, candidate must include every design variable for the current task.\n"
    "You may respond with one of four action types:\n"
    '{"action_type": "propose_design", "candidate": {"var": value}, "confidence": 0.0-1.0}\n'
    '{"action_type": "declare_infeasible", "reason": "...", "conflicting_constraints": [...], "confidence": 0.0-1.0}\n'
    '{"action_type": "request_missing_info", "missing_fields": [...], "clarification_request": "...", "confidence": 0.0-1.0}\n'
    '{"action_type": "replan", "reason": "current direction is a dead-end because ...", "suggested_pivot": "...", "confidence": 0.0-1.0}'
)

def _extract_json(raw: str) -> Any:
    """Try to extract a JSON object from a raw string."""
    return extract_first_json_object(raw)


def parse_action(raw: str | dict) -> dict:
    """Parse and validate a solver action.

    Args:
        raw: Either a dict or a JSON string (possibly with surrounding text).

    Returns:
        Validated action dict with an ``action_type`` field.

    Raises:
        ValueError: If the input cannot be parsed or fails schema validation.
    """
    if isinstance(raw, dict):
        data = raw
    else:
        data = _extract_json(raw)

    schema = _load_schema()
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"Action schema validation failed: {exc.message}") from exc

    return data


def serialize_action(action: dict) -> str:
    """Validate and serialize a solver action to canonical JSON."""
    validated = parse_action(action)
    return json.dumps(validated, sort_keys=True, separators=(",", ":"))


def is_valid_action(raw: str | dict) -> bool:
    """Return True if *raw* can be parsed as a valid solver action."""
    try:
        parse_action(raw)
        return True
    except (ValueError, Exception):
        return False
