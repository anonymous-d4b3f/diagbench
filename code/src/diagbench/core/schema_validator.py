"""
M0 Schema Validator
Validates task, trajectory, scoring, audit, and run-manifest JSON objects
against their respective JSON Schema definitions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "jsonschema is required: pip install jsonschema"
    ) from e

# ---------------------------------------------------------------------------
# Locate schemas relative to this file
# ---------------------------------------------------------------------------
_SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "schemas"

_SCHEMA_FILES: dict[str, str] = {
    "task": "task_schema.json",
    "p1_task": "p1_task_schema.json",
    "canonical_anchor": "canonical_anchor_schema.json",
    "evidence_span": "evidence_span_schema.json",
    "completion_record": "completion_record_schema.json",
    "difficulty_annotation": "difficulty_annotation_schema.json",
    "trajectory": "trajectory_schema.json",
    "scoring": "scoring_schema.json",
    "audit": "audit_schema.json",
    "task_bank_manifest": "task_bank_manifest_schema.json",
    "run_manifest": "run_manifest_schema.json",
    "model_config": "model_config_schema.json",
    "solver_config": "solver_config_schema.json",
}


def _load_schema(name: str) -> dict:
    path = _SCHEMAS_DIR / _SCHEMA_FILES[name]
    with path.open() as fh:
        return json.load(fh)


class ValidationResult:
    """Container returned by every validate_* function."""

    def __init__(self, ok: bool, errors: list[str]) -> None:
        self.ok = ok
        self.errors = errors

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:  # pragma: no cover
        status = "OK" if self.ok else f"FAILED ({len(self.errors)} error(s))"
        return f"<ValidationResult {status}>"


def _join_errors(errors: list[str], extra_errors: list[str]) -> ValidationResult:
    merged = errors + extra_errors
    return ValidationResult(ok=not merged, errors=merged)


def _validate(schema_name: str, data: Any) -> ValidationResult:
    schema = _load_schema(schema_name)
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        msgs = [f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]
        return ValidationResult(ok=False, errors=msgs)
    return ValidationResult(ok=True, errors=[])


def _validate_task_semantics(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return []

    errors: list[str] = []
    design_variables = data.get("design_variables")
    variable_bounds = data.get("variable_bounds")
    best_known_feasible = data.get("best_known_feasible")

    if isinstance(design_variables, list) and isinstance(variable_bounds, dict):
        dv_set = set(design_variables)
        vb_set = set(variable_bounds.keys())
        if dv_set != vb_set:
            missing = sorted(dv_set - vb_set)
            extra = sorted(vb_set - dv_set)
            if missing:
                errors.append(f"variable_bounds: missing bounds for design_variables {missing}")
            if extra:
                errors.append(f"variable_bounds: contains keys not present in design_variables {extra}")

    if isinstance(variable_bounds, dict):
        for key, bounds in variable_bounds.items():
            if not isinstance(bounds, dict):
                continue
            min_value = bounds.get("min")
            max_value = bounds.get("max")
            if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)) and min_value > max_value:
                errors.append(f"variable_bounds.{key}: min must be <= max")

    if isinstance(design_variables, list) and isinstance(best_known_feasible, dict):
        dv_set = set(design_variables)
        bkf_set = set(best_known_feasible.keys())
        if dv_set != bkf_set:
            missing = sorted(dv_set - bkf_set)
            extra = sorted(bkf_set - dv_set)
            if missing:
                errors.append(f"best_known_feasible: missing values for design_variables {missing}")
            if extra:
                errors.append(f"best_known_feasible: contains keys not present in design_variables {extra}")

        if isinstance(variable_bounds, dict):
            for key, value in best_known_feasible.items():
                bounds = variable_bounds.get(key)
                if not isinstance(bounds, dict):
                    continue
                min_value = bounds.get("min")
                max_value = bounds.get("max")
                if isinstance(value, (int, float)) and isinstance(min_value, (int, float)) and value < min_value:
                    errors.append(f"best_known_feasible.{key}: value below declared lower bound")
                if isinstance(value, (int, float)) and isinstance(max_value, (int, float)) and value > max_value:
                    errors.append(f"best_known_feasible.{key}: value above declared upper bound")

    return errors


_P1_GOLD_ACTION = {
    "solvable_wide": "propose_design",
    "solvable_narrow": "propose_design",
    "solvable_anchor": "propose_design",
    "solvable_tight": "propose_design",
    "solvable_base": "propose_design",
    "solvable_boundary": "propose_design",
    "solvable_red_herring": "propose_design",
    "underspecified_nonkey": "propose_design",
    "missing_nonblocker": "propose_design",
    "infeasible_hard_conflict": "declare_infeasible",
    "infeasible_by_margin": "declare_infeasible",
    "infeasible_disguised": "declare_infeasible",
    "infeasible_structural": "declare_infeasible",
    "infeasible_margin": "declare_infeasible",
    "underspecified_key": "request_missing_info",
    "missing_blocker_obvious": "request_missing_info",
    # P1 v3 intentionally has mixed gold labels for this subtype.
    "missing_blocker_ambiguous": None,
}

_P1_MISSING_INFO_SUBTYPES = {
    "underspecified_key",
    "missing_blocker_obvious",
    "missing_blocker_ambiguous",
}


def _validate_p1_task_semantics(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return []

    errors: list[str] = []
    design_variables = data.get("design_variables")
    variable_bounds = data.get("variable_bounds")
    subtype = data.get("p1_subtype")
    gold_label = data.get("gold_label")
    missing_fields = data.get("missing_fields_ground_truth")
    if not missing_fields and isinstance(gold_label, dict):
        missing_fields = gold_label.get("missing_fields")
    missing_set = set(missing_fields or [])

    if isinstance(design_variables, list) and isinstance(variable_bounds, dict):
        dv_set = set(design_variables)
        vb_set = set(variable_bounds.keys())
        if dv_set != vb_set:
            missing = sorted(dv_set - vb_set)
            extra = sorted(vb_set - dv_set)
            allowed_missing_bounds = subtype in _P1_MISSING_INFO_SUBTYPES and set(missing).issubset(missing_set)
            if missing and not allowed_missing_bounds:
                errors.append(f"variable_bounds: missing bounds for design_variables {missing}")
            if extra:
                errors.append(f"variable_bounds: contains keys not present in design_variables {extra}")

    if isinstance(variable_bounds, dict):
        for key, bounds in variable_bounds.items():
            if not isinstance(bounds, dict):
                continue
            min_value = bounds.get("min")
            max_value = bounds.get("max")
            if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)) and min_value > max_value:
                errors.append(f"variable_bounds.{key}: min must be <= max")

    if subtype in _P1_GOLD_ACTION and _P1_GOLD_ACTION[subtype] is not None and isinstance(gold_label, dict):
        gold_action = gold_label.get("action_type")
        if gold_action != _P1_GOLD_ACTION[subtype]:
            errors.append(
                f"gold_label.action_type: expected {_P1_GOLD_ACTION[subtype]!r} for p1_subtype={subtype!r}"
            )

    if subtype not in _P1_MISSING_INFO_SUBTYPES and "excitation_context" not in data:
        errors.append("excitation_context: required for P1 subtypes without missing blocker fields")

    excitation = data.get("excitation_context")
    if subtype not in _P1_MISSING_INFO_SUBTYPES and isinstance(excitation, dict):
        for field in ("frequency_hz", "acceleration_g"):
            if field not in excitation:
                errors.append(f"excitation_context.{field}: required for p1_subtype={subtype!r}")
    elif subtype in _P1_MISSING_INFO_SUBTYPES and isinstance(excitation, dict):
        for field in ("frequency_hz", "acceleration_g"):
            field_path = f"excitation_context.{field}"
            if subtype == "missing_blocker_ambiguous":
                continue
            if field not in excitation and field_path not in missing_set:
                errors.append(f"{field_path}: required unless listed in missing_fields_ground_truth")

    if subtype in {"underspecified_key", "missing_blocker_obvious"}:
        if not missing_fields:
            errors.append(f"missing_fields_ground_truth: required for {subtype}")

    return errors


def validate_task(data: Any) -> ValidationResult:
    """Validate a task dict against task_schema.json."""
    if isinstance(data, dict) and data.get("task_type") == "p1_problem_recognition":
        return validate_p1_task(data)
    base = _validate("task", data)
    if not base.ok:
        return base
    return _join_errors(base.errors, _validate_task_semantics(data))


def validate_p1_task(data: Any) -> ValidationResult:
    """Validate a P1 task dict against p1_task_schema.json."""
    base = _validate("p1_task", data)
    if not base.ok:
        return base
    return _join_errors(base.errors, _validate_p1_task_semantics(data))


def validate_evidence_span(data: Any) -> ValidationResult:
    """Validate an evidence span dict against evidence_span_schema.json."""
    return _validate("evidence_span", data)


def validate_canonical_anchor(data: Any) -> ValidationResult:
    """Validate a canonical paper anchor dict against canonical_anchor_schema.json."""
    base = _validate("canonical_anchor", data)
    if not base.ok:
        return base

    extra_errors: list[str] = []
    for section_name in ["design_variables", "excitation", "environment", "observed_outputs"]:
        section = data.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for field_name, field_record in section.items():
            if not isinstance(field_record, dict):
                continue
            evidence = field_record.get("evidence", [])
            if not evidence and field_record.get("status") != "missing":
                extra_errors.append(f"{section_name}.{field_name}.evidence: required unless status=missing")
            for idx, span in enumerate(evidence):
                result = validate_evidence_span(span)
                extra_errors.extend(
                    [f"{section_name}.{field_name}.evidence[{idx}].{msg}" for msg in result.errors]
                )
            lower_bound = field_record.get("lower_bound")
            upper_bound = field_record.get("upper_bound")
            if (
                isinstance(lower_bound, (int, float))
                and isinstance(upper_bound, (int, float))
                and lower_bound > upper_bound
            ):
                extra_errors.append(
                    f"{section_name}.{field_name}: lower_bound must be <= upper_bound"
                )
    return _join_errors(base.errors, extra_errors)


def validate_completion_record(data: Any) -> ValidationResult:
    """Validate a completion record dict against completion_record_schema.json."""
    base = _validate("completion_record", data)
    if not base.ok:
        return base
    extra_errors: list[str] = []
    for idx, span in enumerate(data.get("evidence", [])):
        result = validate_evidence_span(span)
        extra_errors.extend([f"evidence[{idx}].{msg}" for msg in result.errors])
    ident = data.get("identifiability")
    if isinstance(ident, dict):
        if ident.get("is_identifiable") and data.get("decision") == "manual_review":
            extra_errors.append("decision: identifiable completion should not default to manual_review")
        if not ident.get("is_identifiable") and data.get("result_status") == "oracle_imputed":
            extra_errors.append("result_status: oracle_imputed requires is_identifiable=true")
    return _join_errors(base.errors, extra_errors)


def validate_difficulty_annotation(data: Any) -> ValidationResult:
    """Validate a difficulty annotation dict against difficulty_annotation_schema.json."""
    return _validate("difficulty_annotation", data)


def validate_trajectory(data: Any) -> ValidationResult:
    """Validate a trajectory dict against trajectory_schema.json."""
    return _validate("trajectory", data)


def validate_scoring(data: Any) -> ValidationResult:
    """Validate a scoring config dict against scoring_schema.json."""
    return _validate("scoring", data)


def validate_audit(data: Any) -> ValidationResult:
    """Validate an audit record dict against audit_schema.json."""
    return _validate("audit", data)


def validate_task_bank_manifest(data: Any) -> ValidationResult:
    """Validate a task-bank manifest dict against task_bank_manifest_schema.json."""
    return _validate("task_bank_manifest", data)


def validate_run_manifest(data: Any) -> ValidationResult:
    """Validate a run manifest dict against run_manifest_schema.json."""
    base = _validate("run_manifest", data)
    if not base.ok:
        return base

    extra_errors: list[str] = []
    model_config = data.get("model_config")
    if model_config is not None:
        model_result = validate_model_config(model_config)
        extra_errors.extend([f"model_config.{msg}" for msg in model_result.errors])

    solver_config = {
        "solver_config_version": data.get("solver_config_version"),
        "solver_policy": data.get("solver_policy"),
        "timeout_policy": data.get("timeout_policy"),
        "retry_policy": data.get("retry_policy"),
        "jitter_policy": data.get("jitter_policy"),
    }
    solver_result = validate_solver_config(solver_config)
    extra_errors.extend([f"solver_config.{msg}" for msg in solver_result.errors])
    return _join_errors(base.errors, extra_errors)


def validate_model_config(data: Any) -> ValidationResult:
    """Validate a model config dict against model_config_schema.json."""
    return _validate("model_config", data)


def validate_solver_config(data: Any) -> ValidationResult:
    """Validate a solver config dict against solver_config_schema.json."""
    return _validate("solver_config", data)


SCHEMA_REGISTRY: dict[str, Any] = {
    name: None for name in _SCHEMA_FILES  # lazy-loaded on first use
}
