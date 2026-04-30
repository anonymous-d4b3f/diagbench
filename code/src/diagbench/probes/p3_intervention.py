"""
Prompt-facing adapters for P3 intervention experiments.

These helpers intentionally operate on observed trajectory state only. They
never inspect hidden trap metadata, so the intervention stays on the prompt
surface rather than changing the underlying benchmark semantics.
"""
from __future__ import annotations

from typing import Any


HISTORY_MODES = frozenset({"raw_history", "state_summary"})
FEEDBACK_MODES = frozenset({"full_numeric", "coarse_feedback"})


def validate_history_mode(mode: str) -> str:
    if mode not in HISTORY_MODES:
        allowed = ", ".join(sorted(HISTORY_MODES))
        raise ValueError(f"Unsupported history_mode={mode!r}. Expected one of: {allowed}")
    return mode


def validate_feedback_mode(mode: str) -> str:
    if mode not in FEEDBACK_MODES:
        allowed = ", ".join(sorted(FEEDBACK_MODES))
        raise ValueError(f"Unsupported feedback_mode={mode!r}. Expected one of: {allowed}")
    return mode


def _objective_direction_signal(objective_delta: Any) -> str:
    if not isinstance(objective_delta, (int, float)):
        return "flat"
    if objective_delta > 1e-9:
        return "better"
    if objective_delta < -1e-9:
        return "worse"
    return "flat"


def _extract_verifier_response(step: dict[str, Any]) -> dict[str, Any]:
    response = step.get("verifier_response")
    if isinstance(response, dict):
        return response
    if step.get("action_type") == "propose_design":
        return step
    return {}


def _extract_violations(step: dict[str, Any]) -> list[str]:
    response = _extract_verifier_response(step)
    raw = response.get("violations", [])
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def _extract_boundary_state(step: dict[str, Any]) -> dict[str, Any]:
    response = _extract_verifier_response(step)
    boundary_state = response.get("boundary_state")
    if isinstance(boundary_state, dict):
        return dict(boundary_state)
    top_level = step.get("boundary_state")
    if isinstance(top_level, dict):
        return dict(top_level)
    return {}


def _build_coarse_feedback(step: dict[str, Any]) -> dict[str, Any]:
    response = _extract_verifier_response(step)
    return {
        "is_feasible": bool(response.get("is_feasible", response.get("feasible", False))),
        "violations": _extract_violations(step),
        "boundary_state": _extract_boundary_state(step),
        "objective_direction_signal": _objective_direction_signal(
            response.get("objective_delta", step.get("objective_delta"))
        ),
    }


def _compress_history_step(step: dict[str, Any]) -> dict[str, Any]:
    compressed: dict[str, Any] = {
        "step_index": step.get("step_index"),
        "action_type": step.get("action_type"),
        "proposal": step.get("proposal"),
        "verifier_response": None,
        "constraint_slack": None,
        "objective_delta": None,
        "boundary_state": None,
        "confidence": step.get("confidence"),
    }
    for key in (
        "analysis_summary",
        "reason",
        "suggested_pivot",
        "missing_fields",
        "clarification_request",
        "conflicting_constraints",
    ):
        if key in step:
            compressed[key] = step.get(key)
    if step.get("action_type") == "propose_design":
        coarse_feedback = _build_coarse_feedback(step)
        compressed["verifier_response"] = coarse_feedback
        compressed["boundary_state"] = coarse_feedback["boundary_state"]
    return compressed


def _proposal_steps(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        step
        for step in history
        if step.get("action_type") == "propose_design" and isinstance(step.get("proposal"), dict)
    ]


def _latest_feasible_step(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in reversed(_proposal_steps(history)):
        response = _extract_verifier_response(step)
        if bool(response.get("is_feasible", response.get("feasible", False))):
            return step
    return None


def _aggregate_objective_trend(signals: list[str]) -> str:
    if not signals:
        return "mixed"
    if all(signal == "better" for signal in signals):
        return "up"
    if all(signal == "worse" for signal in signals):
        return "down"
    return "mixed"


def _compare_violations(previous: set[str], current: set[str]) -> str:
    if current == previous:
        return "same"
    if current.issubset(previous):
        return "better"
    if previous.issubset(current):
        return "worse"
    if len(current) < len(previous):
        return "better"
    if len(current) > len(previous):
        return "worse"
    return "mixed"


def _aggregate_violation_trend(transitions: list[str]) -> str:
    if not transitions:
        return "mixed"
    if len(set(transitions)) == 1:
        return transitions[0]
    return "mixed"


def _build_state_summary(
    *,
    history: list[dict[str, Any]],
    step_index: int,
    max_attempts: int,
    feedback_mode: str,
) -> dict[str, Any]:
    proposal_steps = _proposal_steps(history)
    latest_proposal_step = proposal_steps[-1] if proposal_steps else None
    latest_feasible_step = _latest_feasible_step(history)
    latest_proposal = latest_proposal_step.get("proposal") if latest_proposal_step is not None else None

    objective_signals = [
        _objective_direction_signal(
            _extract_verifier_response(step).get("objective_delta", step.get("objective_delta"))
        )
        for step in proposal_steps
    ]
    objective_trend = _aggregate_objective_trend(objective_signals[-2:])

    violation_sets = [set(_extract_violations(step)) for step in proposal_steps]
    violation_transitions = [
        _compare_violations(previous, current)
        for previous, current in zip(violation_sets, violation_sets[1:])
    ]
    violation_trend = _aggregate_violation_trend(violation_transitions[-2:])

    feasible_seen = False
    post_feasible_deterioration = False
    for step in proposal_steps:
        response = _extract_verifier_response(step)
        is_feasible = bool(response.get("is_feasible", response.get("feasible", False)))
        if is_feasible:
            feasible_seen = True
            continue
        if feasible_seen:
            post_feasible_deterioration = True
            break

    latest_violations = set(_extract_violations(latest_proposal_step or {}))
    latest_feasible_violations = set(_extract_violations(latest_feasible_step or {}))
    new_violation_introduced = bool(latest_violations - latest_feasible_violations)

    summary: dict[str, Any] = {
        "history_mode": "state_summary",
        "feedback_mode": feedback_mode,
        "current_step_index": step_index,
        "remaining_budget": max(0, max_attempts - step_index),
        "latest_proposal": latest_proposal,
        "latest_verifier": None,
        "latest_feasible_proposal": (
            latest_feasible_step.get("proposal") if latest_feasible_step is not None else None
        ),
        "objective_trend": objective_trend,
        "violation_trend": violation_trend,
        "post_feasible_deterioration": post_feasible_deterioration,
        "new_violation_introduced": new_violation_introduced,
    }

    if latest_proposal_step is not None:
        if feedback_mode == "coarse_feedback":
            summary["latest_verifier"] = _build_coarse_feedback(latest_proposal_step)
        else:
            response = _extract_verifier_response(latest_proposal_step)
            summary["latest_verifier"] = {
                "is_feasible": bool(response.get("is_feasible", response.get("feasible", False))),
                "violations": _extract_violations(latest_proposal_step),
                "boundary_state": _extract_boundary_state(latest_proposal_step),
                "objective_value": response.get("objective_value"),
                "objective_delta": response.get("objective_delta", latest_proposal_step.get("objective_delta")),
            }

    if feedback_mode == "full_numeric":
        if latest_feasible_step is not None:
            latest_feasible_response = _extract_verifier_response(latest_feasible_step)
            summary["latest_feasible_objective"] = latest_feasible_response.get("objective_value")
        feasible_objectives = [
            _extract_verifier_response(step).get("objective_value")
            for step in proposal_steps
            if bool(_extract_verifier_response(step).get("is_feasible", _extract_verifier_response(step).get("feasible", False)))
            and isinstance(_extract_verifier_response(step).get("objective_value"), (int, float))
        ]
        summary["best_so_far_feasible_objective"] = max(feasible_objectives) if feasible_objectives else None

    return summary


def build_prompt_history(
    *,
    history: list[dict[str, Any]],
    step_index: int,
    max_attempts: int,
    history_mode: str,
    feedback_mode: str,
) -> list[dict[str, Any]] | dict[str, Any]:
    history_mode = validate_history_mode(history_mode)
    feedback_mode = validate_feedback_mode(feedback_mode)
    if history_mode == "raw_history":
        if feedback_mode == "full_numeric":
            return [dict(step) for step in history]
        return [_compress_history_step(step) for step in history]
    return _build_state_summary(
        history=history,
        step_index=step_index,
        max_attempts=max_attempts,
        feedback_mode=feedback_mode,
    )
