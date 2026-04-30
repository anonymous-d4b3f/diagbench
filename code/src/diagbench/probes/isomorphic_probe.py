from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from diagbench.physics.oracle import PiezoelectricOracle


FORM_A = "A_selection"
FORM_B = "B_generation"
FORM_C = "C_completion"
FORM_ORDER = (FORM_A, FORM_B, FORM_C)


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


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def save_jsonl(rows: list[dict[str, Any]], path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate_signature(candidate: dict[str, float], *, precision: int = 6) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((name, round(float(value), precision)) for name, value in candidate.items()))


def normalize_candidate_for_task(
    task: dict[str, Any],
    candidate: dict[str, Any],
    *,
    required_variables: list[str] | None = None,
) -> dict[str, float]:
    required = required_variables or list(task["design_variables"])
    normalized: dict[str, float] = {}
    for name in required:
        if name not in candidate:
            raise ValueError(f"candidate missing variable '{name}'")
        normalized[name] = round(float(candidate[name]), 6)
    return normalized


def candidate_within_bounds(task: dict[str, Any], candidate: dict[str, float]) -> bool:
    for variable in task["design_variables"]:
        value = float(candidate[variable])
        bounds = task["variable_bounds"][variable]
        lower = float(bounds["min"])
        upper = float(bounds["max"])
        if value < lower or value > upper:
            return False
    return True


def _constraint_limit_map(task: dict[str, Any]) -> dict[str, float]:
    return {item["name"]: float(item["limit"]) for item in task.get("constraints", [])}


def _normalized_violation_from_slack(task: dict[str, Any], constraint_slack: dict[str, float]) -> float:
    limits = _constraint_limit_map(task)
    total = 0.0
    for name, slack in constraint_slack.items():
        if float(slack) >= 0:
            continue
        limit = abs(float(limits.get(name, 1.0))) or 1.0
        total += abs(float(slack)) / limit
    return round(total, 6)


def evaluate_candidate(task: dict[str, Any], candidate: dict[str, float], *, oracle: PiezoelectricOracle) -> dict[str, Any]:
    constraints = _constraint_limit_map(task)
    result = oracle.evaluate(
        candidate,
        task["excitation_context"],
        constraints=constraints,
        environment=task.get("environment_context", {}),
    )
    return {
        "candidate": {name: round(float(value), 6) for name, value in candidate.items()},
        "is_feasible": bool(result.is_feasible) and candidate_within_bounds(task, candidate),
        "constraint_slack": dict(result.constraint_slack),
        "objective_value": round(float(result.load_power_uw), 6),
        "total_normalized_violation": _normalized_violation_from_slack(task, result.constraint_slack),
        "resonant_freq_hz": round(float(result.resonant_freq_hz), 6),
        "tip_stress_mpa": round(float(result.tip_stress_mpa), 6),
        "tip_disp_mm": round(float(result.tip_disp_mm), 6),
        "freq_error_pct": round(float(result.freq_error_pct), 6),
    }


def aggregate_non_feasible_decoys(task: dict[str, Any], p4_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: dict[tuple[tuple[str, float], ...], dict[str, Any]] = {}
    for row in p4_rows:
        for candidate_row in row.get("candidate_pool", []):
            candidate = candidate_row.get("candidate")
            if not isinstance(candidate, dict):
                continue
            if candidate_row.get("is_feasible"):
                continue
            signature = candidate_signature(candidate)
            payload = {
                "candidate": {name: round(float(value), 6) for name, value in candidate.items()},
                "is_feasible": False,
                "constraint_slack": dict(candidate_row.get("constraint_slack", {})),
                "objective_value": float(candidate_row.get("objective_value", 0.0)),
                "total_normalized_violation": float(candidate_row.get("total_normalized_violation", 0.0)),
                "source_candidate_id": candidate_row.get("candidate_id"),
                "source_role": candidate_row.get("candidate_role") or candidate_row.get("candidate_category"),
            }
            existing = aggregated.get(signature)
            if existing is None or payload["total_normalized_violation"] < existing["total_normalized_violation"]:
                aggregated[signature] = payload
    return sorted(
        aggregated.values(),
        key=lambda item: (float(item["total_normalized_violation"]), -float(item["objective_value"])),
    )


def synthesize_edge_decoys(
    task: dict[str, Any],
    gold_candidate: dict[str, float],
    *,
    oracle: PiezoelectricOracle,
    existing_signatures: set[tuple[tuple[str, float], ...]],
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    seen = set(existing_signatures)
    design_variables = list(task["design_variables"])
    spans = {
        name: float(task["variable_bounds"][name]["max"]) - float(task["variable_bounds"][name]["min"])
        for name in design_variables
    }
    ranked_variables = sorted(design_variables, key=lambda name: spans[name], reverse=True)
    for variable in ranked_variables:
        bounds = task["variable_bounds"][variable]
        lower = float(bounds["min"])
        upper = float(bounds["max"])
        for value in (lower, upper, lower + 0.1 * (upper - lower), lower + 0.9 * (upper - lower)):
            candidate = dict(gold_candidate)
            candidate[variable] = round(value, 6)
            signature = candidate_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            evaluation = evaluate_candidate(task, candidate, oracle=oracle)
            if evaluation["is_feasible"]:
                continue
            proposals.append(evaluation)
    if len(ranked_variables) >= 2:
        first, second = ranked_variables[:2]
        first_bounds = task["variable_bounds"][first]
        second_bounds = task["variable_bounds"][second]
        pair_values = [
            (float(first_bounds["min"]), float(second_bounds["min"])),
            (float(first_bounds["max"]), float(second_bounds["max"])),
            (float(first_bounds["min"]), float(second_bounds["max"])),
            (float(first_bounds["max"]), float(second_bounds["min"])),
        ]
        for first_value, second_value in pair_values:
            candidate = dict(gold_candidate)
            candidate[first] = round(first_value, 6)
            candidate[second] = round(second_value, 6)
            signature = candidate_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            evaluation = evaluate_candidate(task, candidate, oracle=oracle)
            if evaluation["is_feasible"]:
                continue
            proposals.append(evaluation)
    proposals.sort(key=lambda item: (float(item["total_normalized_violation"]), -float(item["objective_value"])))
    return proposals


def derive_missing_variables(
    task: dict[str, Any],
    *,
    start_candidate: dict[str, float],
    gold_candidate: dict[str, float],
    count: int = 2,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for variable in task["design_variables"]:
        bounds = task["variable_bounds"][variable]
        span = max(float(bounds["max"]) - float(bounds["min"]), 1e-9)
        diff = abs(float(start_candidate[variable]) - float(gold_candidate[variable])) / span
        scored.append((diff, variable))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [name for _, name in scored[:count]]
    if len(selected) < count:
        for variable in task["design_variables"]:
            if variable not in selected:
                selected.append(variable)
            if len(selected) == count:
                break
    return selected


def prompt_style_for_model(model_id: str, explicit_style: str | None = None) -> str:
    if explicit_style:
        return explicit_style
    model_lower = model_id.lower()
    if "base" in model_lower:
        return "base_fewshot"
    return "strict_json"


def build_probe_prompt(task: dict[str, Any], *, prompt_style: str) -> tuple[str, str]:
    constraints_json = json.dumps(task["constraints"], ensure_ascii=False)
    bounds_json = json.dumps(task["variable_bounds"], ensure_ascii=False)
    excitation_json = json.dumps(task["excitation_context"], ensure_ascii=False)
    environment_json = json.dumps(task.get("environment_context", {}), ensure_ascii=False)
    design_variables = list(task["design_variables"])
    form = task["form"]

    if form == FORM_A:
        examples = ""
        if prompt_style == "base_fewshot":
            examples = (
                "Example output format:\n"
                '{"selected_candidate_id":"B","confidence":0.51}\n'
                '{"selected_candidate_id":"D","confidence":0.63}\n'
            )
        candidate_lines = []
        for candidate in task["candidate_pool"]:
            candidate_lines.append(f"{candidate['candidate_id']}: {json.dumps(candidate['candidate'], ensure_ascii=False)}")
        user_prompt = (
            "Engineering feasibility selection task.\n"
            "Exactly one candidate strictly satisfies all physical constraints.\n"
            "Select the ONLY feasible candidate.\n"
            f"Design variables: {json.dumps(design_variables, ensure_ascii=False)}\n"
            f"Variable bounds: {bounds_json}\n"
            f"Objective: {task['objective_direction']} {task['objective_name']}\n"
            f"Constraints: {constraints_json}\n"
            f"Excitation context: {excitation_json}\n"
            f"Environment context: {environment_json}\n"
            f"Candidates:\n" + "\n".join(candidate_lines) + "\n"
            "Return only JSON as {\"selected_candidate_id\":\"A\",\"confidence\":0.0}.\n"
            + examples
        )
        return (
            "Reply with only one JSON object. Pick the single feasible candidate id.",
            user_prompt,
        )

    if form == FORM_B:
        examples = ""
        if prompt_style == "base_fewshot":
            examples = (
                "Example output format:\n"
                '{"candidate":{"beam_length_mm":120.0,"beam_width_mm":12.0,"substrate_thickness_um":500.0,"piezo_thickness_um":120.0,"tip_mass_g":2.5,"load_resistance_ohm":150000.0},"confidence":0.40}\n'
            )
        user_prompt = (
            "Engineering constrained generation task.\n"
            "Modify the current design to produce a NEW full design candidate that strictly satisfies all constraints.\n"
            f"Design variables: {json.dumps(design_variables, ensure_ascii=False)}\n"
            f"Variable bounds: {bounds_json}\n"
            f"Objective: {task['objective_direction']} {task['objective_name']}\n"
            f"Constraints: {constraints_json}\n"
            f"Excitation context: {excitation_json}\n"
            f"Environment context: {environment_json}\n"
            f"Current violating design: {json.dumps(task['seed_candidate'], ensure_ascii=False)}\n"
            "Return only JSON as {\"candidate\":{...all design variables...},\"confidence\":0.0}.\n"
            + examples
        )
        return (
            "Reply with only one JSON object containing a full candidate.",
            user_prompt,
        )

    if form == FORM_C:
        missing_variables = task["missing_variables"]
        examples = ""
        if prompt_style == "base_fewshot":
            examples = (
                "Example output format:\n"
                '{"candidate":{"tip_mass_g":2.75,"load_resistance_ohm":180000.0},"confidence":0.44}\n'
            )
        user_prompt = (
            "Engineering constrained completion task.\n"
            "The fixed design parameters below must remain unchanged.\n"
            f"Design variables: {json.dumps(design_variables, ensure_ascii=False)}\n"
            f"Variable bounds: {bounds_json}\n"
            f"Objective: {task['objective_direction']} {task['objective_name']}\n"
            f"Constraints: {constraints_json}\n"
            f"Excitation context: {excitation_json}\n"
            f"Environment context: {environment_json}\n"
            f"Fixed known parameters: {json.dumps(task['known_candidate'], ensure_ascii=False)}\n"
            f"Missing variables you must determine exactly: {json.dumps(missing_variables, ensure_ascii=False)}\n"
            "Return only JSON as {\"candidate\":{...only missing variables...},\"confidence\":0.0}.\n"
            + examples
        )
        return (
            "Reply with only one JSON object containing values for the missing variables.",
            user_prompt,
        )

    raise ValueError(f"Unsupported probe form: {form}")


def parse_selection_response(raw: str, *, candidate_ids: list[str]) -> tuple[str | None, bool]:
    try:
        data = extract_first_json_object(raw)
        selected = data.get("selected_candidate_id")
        if selected is None:
            selected = data.get("candidate_id") or data.get("answer") or data.get("choice")
        if isinstance(selected, str):
            normalized = selected.strip().upper()
            if normalized in candidate_ids:
                return normalized, True
    except Exception:
        pass

    pattern = re.compile(
        r"(selected_candidate_id|candidate_id|answer|choice|selected)\s*[:=]?\s*[\"']?([A-Z])[\"']?",
        flags=re.IGNORECASE,
    )
    match = pattern.search(raw)
    if match:
        normalized = match.group(2).upper()
        if normalized in candidate_ids:
            return normalized, False
    hits = {candidate_id for candidate_id in candidate_ids if re.search(rf"\b{re.escape(candidate_id)}\b", raw)}
    if len(hits) == 1:
        return next(iter(hits)), False
    return None, False


def parse_candidate_response(
    raw: str,
    *,
    variables: list[str],
) -> tuple[dict[str, float] | None, bool]:
    try:
        data = extract_first_json_object(raw)
        candidate = data.get("candidate", data)
        if isinstance(candidate, dict):
            parsed: dict[str, float] = {}
            for variable in variables:
                value = candidate.get(variable)
                if isinstance(value, (int, float)):
                    parsed[variable] = round(float(value), 6)
            if len(parsed) == len(variables):
                return parsed, True
    except Exception:
        pass

    parsed: dict[str, float] = {}
    for variable in variables:
        pattern = re.compile(
            rf"[\"']?{re.escape(variable)}[\"']?\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            flags=re.IGNORECASE,
        )
        match = pattern.search(raw)
        if not match:
            continue
        parsed[variable] = round(float(match.group(1)), 6)
    if len(parsed) == len(variables):
        return parsed, False
    return None, False
