"""
Shared helpers for extracting the first valid JSON object from model output.
"""
from __future__ import annotations

import json
import re
from typing import Any


def _extract_balanced_object(raw: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(raw)):
        char = raw[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    return None


def extract_first_json_object(raw: str | dict[str, Any]) -> dict[str, Any]:
    """
    Parse a dict directly or extract the first valid JSON object from raw text.

    This is tolerant of provider wrappers like:
      - ```json ... ```
      - Here is the JSON: {...}
      - leading commentary before the first JSON block
    """
    if isinstance(raw, dict):
        return raw

    text = str(raw).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{", text):
        candidate = _extract_balanced_object(text, match.start())
        if candidate is None:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data

    raise ValueError(f"No valid JSON object found in input: {text[:200]!r}")
