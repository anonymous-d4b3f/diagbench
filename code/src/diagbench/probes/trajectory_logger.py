"""
M2-T1: Trajectory logger.

Records the full iterative search history of a solver run as a trajectory that
conforms to ``trajectory_schema.json``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diagbench.core.schema_validator import validate_trajectory


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrajectoryStep:
    """One schema-compatible step in the iterative search."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = dict(payload)

    @property
    def action_type(self) -> str:
        return self.payload["action_type"]

    @property
    def proposal(self) -> dict[str, Any] | None:
        return self.payload.get("proposal")

    @property
    def verifier_response(self) -> dict[str, Any] | None:
        response = self.payload.get("verifier_response")
        return response if isinstance(response, dict) else None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryStep":
        return cls(data)


class Trajectory:
    """Full iterative search record for one task/model run."""

    def __init__(
        self,
        task_id: str,
        model_id: str,
        run_id: str,
        steps: list[TrajectoryStep],
        started_at: str,
        finished_at: str,
        terminal_action: str,
    ) -> None:
        self.task_id = task_id
        self.model_id = model_id
        self.run_id = run_id
        self.steps = steps
        self.started_at = started_at
        self.finished_at = finished_at
        self.terminal_action = terminal_action

    def proposals(self) -> list[dict[str, Any]]:
        return [
            step.proposal
            for step in self.steps
            if step.action_type == "propose_design" and isinstance(step.proposal, dict)
        ]

    def violations_per_step(self) -> list[list[str]]:
        violations: list[list[str]] = []
        for step in self.steps:
            response = step.verifier_response or {}
            raw = response.get("violations", [])
            violations.append(raw if isinstance(raw, list) else [])
        return violations

    def objective_per_step(self) -> list[float | None]:
        values: list[float | None] = []
        for step in self.steps:
            response = step.verifier_response or {}
            objective = response.get("objective_value")
            values.append(objective if isinstance(objective, (int, float)) else None)
        return values

    def is_feasible_per_step(self) -> list[bool]:
        statuses: list[bool] = []
        for step in self.steps:
            response = step.verifier_response or {}
            if "is_feasible" in response:
                statuses.append(bool(response["is_feasible"]))
            else:
                statuses.append(bool(response.get("feasible", False)))
        return statuses

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "run_id": self.run_id,
            "steps": [step.to_dict() for step in self.steps],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "terminal_action": self.terminal_action,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        return cls(
            task_id=data["task_id"],
            model_id=data["model_id"],
            run_id=data["run_id"],
            steps=[TrajectoryStep.from_dict(step) for step in data["steps"]],
            started_at=data.get("started_at", _utc_now_iso()),
            finished_at=data.get("finished_at", _utc_now_iso()),
            terminal_action=data.get("terminal_action", "budget_exhausted"),
        )


class TrajectoryLogger:
    """Stateful logger for one solver run."""

    def __init__(self, task_id: str, model_id: str, run_id: str) -> None:
        self.task_id = task_id
        self.model_id = model_id
        self.run_id = run_id
        self._steps: list[TrajectoryStep] = []
        self._started_at = _utc_now_iso()

    def current_steps(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self._steps]

    def _validate_step(self, step_payload: dict[str, Any]) -> None:
        scaffold = {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "run_id": self.run_id,
            "steps": [step_payload],
        }
        result = validate_trajectory(scaffold)
        if not result.ok:
            raise ValueError(f"Invalid trajectory step: {result.errors}")

    def _build_step_payload(
        self,
        *,
        step_index: int,
        action: dict[str, Any],
        verifier_response: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(action, dict):
            raise ValueError("action must be a dict")

        action_type = action.get("action_type")
        if action_type == "propose_design":
            if not isinstance(verifier_response, dict):
                raise ValueError("propose_design requires a verifier_response dict")
            payload: dict[str, Any] = {
                "step_index": step_index,
                "action_type": "propose_design",
                "proposal": action.get("candidate"),
                "verifier_response": verifier_response,
                "constraint_slack": verifier_response.get("constraint_slack"),
                "objective_delta": verifier_response.get("objective_delta"),
                "boundary_state": verifier_response.get("boundary_state"),
                "confidence": action.get("confidence"),
            }
            if "analysis_summary" in action:
                payload["analysis_summary"] = action["analysis_summary"]
            return payload

        if action_type == "declare_infeasible":
            payload = {
                "step_index": step_index,
                "action_type": "declare_infeasible",
                "proposal": None,
                "verifier_response": None,
                "constraint_slack": None,
                "objective_delta": None,
                "boundary_state": None,
                "reason": action.get("reason"),
                "confidence": action.get("confidence"),
            }
            if "conflicting_constraints" in action:
                payload["conflicting_constraints"] = action["conflicting_constraints"]
            return payload

        if action_type == "request_missing_info":
            payload = {
                "step_index": step_index,
                "action_type": "request_missing_info",
                "proposal": None,
                "verifier_response": None,
                "constraint_slack": None,
                "objective_delta": None,
                "boundary_state": None,
                "missing_fields": action.get("missing_fields"),
                "confidence": action.get("confidence"),
            }
            if "clarification_request" in action:
                payload["clarification_request"] = action["clarification_request"]
            return payload

        if action_type == "replan":
            payload = {
                "step_index": step_index,
                "action_type": "replan",
                "proposal": None,
                "verifier_response": None,
                "constraint_slack": None,
                "objective_delta": None,
                "boundary_state": None,
                "reason": action.get("reason"),
                "confidence": action.get("confidence"),
            }
            if "suggested_pivot" in action:
                payload["suggested_pivot"] = action["suggested_pivot"]
            return payload

        if action_type == "invalid_output":
            payload = {
                "step_index": step_index,
                "action_type": "invalid_output",
                "proposal": None,
                "verifier_response": None,
                "constraint_slack": None,
                "objective_delta": None,
                "boundary_state": None,
                "reason": action.get("error_message", "invalid model output"),
                "confidence": action.get("confidence"),
            }
            if "error_source" in action:
                payload["error_source"] = action["error_source"]
            return payload

        raise ValueError(f"Unsupported action_type: {action_type!r}")

    def log_step(
        self,
        action: dict[str, Any],
        verifier_response: dict[str, Any] | None = None,
    ) -> None:
        """Record one solver step and validate it immediately."""
        payload = self._build_step_payload(
            step_index=len(self._steps),
            action=action,
            verifier_response=verifier_response,
        )
        self._validate_step(payload)
        self._steps.append(TrajectoryStep(payload))

    def finalize(self, terminal_action: str = "budget_exhausted") -> Trajectory:
        """Seal the trajectory, validate it, and return it."""
        trajectory = Trajectory(
            task_id=self.task_id,
            model_id=self.model_id,
            run_id=self.run_id,
            steps=list(self._steps),
            started_at=self._started_at,
            finished_at=_utc_now_iso(),
            terminal_action=terminal_action,
        )
        result = validate_trajectory(trajectory.to_dict())
        if not result.ok:
            raise ValueError(f"Invalid trajectory: {result.errors}")
        return trajectory

    @staticmethod
    def save(trajectory: Trajectory, path: Path | str) -> None:
        path = Path(path)
        result = validate_trajectory(trajectory.to_dict())
        if not result.ok:
            raise ValueError(f"Invalid trajectory: {result.errors}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(trajectory.to_dict(), fh, indent=2)
            fh.write("\n")

    @staticmethod
    def load(path: Path | str) -> Trajectory:
        with Path(path).open() as fh:
            data = json.load(fh)
        result = validate_trajectory(data)
        if not result.ok:
            raise ValueError(f"Invalid trajectory artifact: {result.errors}")
        return Trajectory.from_dict(data)

    @staticmethod
    def save_batch(trajectories: list[Trajectory], path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for trajectory in trajectories:
                result = validate_trajectory(trajectory.to_dict())
                if not result.ok:
                    raise ValueError(f"Invalid trajectory: {result.errors}")
                fh.write(json.dumps(trajectory.to_dict()) + "\n")

    @staticmethod
    def load_batch(path: Path | str) -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        with Path(path).open() as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                result = validate_trajectory(data)
                if not result.ok:
                    raise ValueError(
                        f"Invalid trajectory artifact at line {line_num}: {result.errors}"
                    )
                trajectories.append(Trajectory.from_dict(data))
        return trajectories
