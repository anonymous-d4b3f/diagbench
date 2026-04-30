"""
P3 Evaluator: Error Diagnosis and Trap Escape (v1 — hardened).

Headline metric: trap_escape_rate
Secondary metrics: escape_time, dead_budget_rate, explicit_replan_rate,
                   escape_quality, constraint_cascade_rate

v1 hardening changes vs v0:
  - threshold raised from 0.02×range to 0.08×range
  - escape requires 2 consecutive steps maintaining escape direction
  - replan no longer counts as escape (only as diagnostic signal)
  - new: escape_quality — mean feasibility rate in post-escape steps
  - new: constraint_cascade_rate — fraction of escapes that enter a new violation
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diagbench.probes.trajectory_logger import TrajectoryLogger

# v1 escape threshold: 8% of variable range (was 2% in v0)
_ESCAPE_THRESHOLD_FRACTION = 0.08

# v1: require this many consecutive steps in escape direction
_ESCAPE_CONFIRMATION_STEPS = 2


@dataclass
class P3TaskResult:
    task_id: str
    runner_name: str
    trap_exited: bool
    explicit_replan: bool
    dead_budget_rate: float
    exit_step: int | None
    # v1 new fields
    escape_quality: float | None  # mean feasibility rate in post-escape steps
    constraint_cascade: bool      # True if escape triggered a new constraint violation

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "runner_name": self.runner_name,
            "trap_exited": self.trap_exited,
            "explicit_replan": self.explicit_replan,
            "dead_budget_rate": round(self.dead_budget_rate, 6),
            "exit_step": self.exit_step,
            "escape_quality": round(self.escape_quality, 4) if self.escape_quality is not None else None,
            "constraint_cascade": self.constraint_cascade,
        }


@dataclass
class P3Summary:
    runner_name: str
    n_tasks: int
    trap_escape_rate: float
    dead_budget_rate: float
    explicit_replan_rate: float
    escape_time: float | None
    # v1 new aggregate fields
    escape_quality: float | None          # mean across escaped tasks
    constraint_cascade_rate: float | None  # fraction of escapes that cascade

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner_name": self.runner_name,
            "n_tasks": self.n_tasks,
            "trap_escape_rate": round(self.trap_escape_rate, 4),
            "dead_budget_rate": round(self.dead_budget_rate, 4),
            "explicit_replan_rate": round(self.explicit_replan_rate, 4),
            "escape_time": round(self.escape_time, 2) if self.escape_time is not None else None,
            "escape_quality": round(self.escape_quality, 4) if self.escape_quality is not None else None,
            "constraint_cascade_rate": (
                round(self.constraint_cascade_rate, 4) if self.constraint_cascade_rate is not None else None
            ),
        }


class P3Evaluator:
    """v1 hardened P3 evaluator.

    Key differences from v0:
      - Escape requires the trap variable to move ≥ 8% of range in the escape
        direction AND that movement to be sustained for 2 consecutive propose_design
        steps.
      - A replan action is counted as a diagnostic signal (explicit_replan) but does
        NOT by itself count as a successful escape.
      - escape_quality measures whether post-escape proposals are actually feasible.
      - constraint_cascade detects if the escape move introduced a new violation.
    """

    def load_results(self, path: Path | str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with Path(path).open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def load_tasks(self, path: Path | str) -> dict[str, dict[str, Any]]:
        tasks: dict[str, dict[str, Any]] = {}
        with Path(path).open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    task = json.loads(line)
                    tasks[task["task_id"]] = task
        return tasks

    def load_trajectories(self, path: Path | str) -> dict[str, Any]:
        return {trajectory.task_id: trajectory for trajectory in TrajectoryLogger.load_batch(path)}

    def evaluate_task(
        self,
        *,
        task: dict[str, Any],
        trajectory: Any,
        runner_name: str,
    ) -> P3TaskResult:
        metadata = task["trap_metadata"]
        seed_count = int(metadata["seed_step_count"])
        trap_variable = metadata["trap_variable"]
        escape_direction = metadata["escape_direction"]
        query_budget = int(task["query_budget"])

        bounds = task["variable_bounds"][trap_variable]
        var_range = float(bounds["max"]) - float(bounds["min"])
        threshold = _ESCAPE_THRESHOLD_FRACTION * var_range

        seed_steps = trajectory.steps[:seed_count]
        last_seed_candidate = seed_steps[-1].proposal if seed_steps else None
        live_steps = trajectory.steps[seed_count:]

        # --- Track escape with confirmation ---
        exit_step: int | None = None
        explicit_replan = False
        trap_proposals_before_exit = 0
        consecutive_escape_count = 0

        # Collect all propose_design steps for post-escape analysis
        propose_steps: list[tuple[int, Any]] = []  # (relative_index, step)

        for relative_index, step in enumerate(live_steps, start=1):
            if step.action_type == "replan":
                explicit_replan = True
                # v1: replan does NOT count as escape — only as diagnostic
                continue

            if step.action_type != "propose_design":
                continue

            propose_steps.append((relative_index, step))

            if exit_step is not None:
                # Already escaped, just collecting post-escape steps
                continue

            trap_proposals_before_exit += 1

            if isinstance(last_seed_candidate, dict) and isinstance(step.proposal, dict):
                delta = float(step.proposal[trap_variable]) - float(last_seed_candidate[trap_variable])
                is_escape_direction = (
                    (escape_direction == "decrease" and delta <= -threshold)
                    or (escape_direction == "increase" and delta >= threshold)
                )

                if is_escape_direction:
                    consecutive_escape_count += 1
                    if consecutive_escape_count >= _ESCAPE_CONFIRMATION_STEPS:
                        exit_step = relative_index
                else:
                    consecutive_escape_count = 0

        trap_exited = exit_step is not None
        if not trap_exited:
            trap_proposals_before_exit = len(propose_steps)

        # --- Escape quality: feasibility rate in post-escape steps ---
        escape_quality: float | None = None
        constraint_cascade = False

        if trap_exited:
            # Find the index in propose_steps where escape was confirmed
            escape_propose_idx = None
            for idx, (ri, _) in enumerate(propose_steps):
                if ri == exit_step:
                    escape_propose_idx = idx
                    break

            if escape_propose_idx is not None:
                post_escape = propose_steps[escape_propose_idx:]
                if post_escape:
                    feasible_count = 0
                    new_violations_seen = False

                    # Get the set of violations at trap entry (last seed step)
                    seed_violations = set()
                    if seed_steps:
                        last_seed_dict = seed_steps[-1].to_dict()
                        seed_slack = last_seed_dict.get("constraint_slack") or {}
                        seed_violations = {
                            name for name, val in seed_slack.items()
                            if isinstance(val, (int, float)) and val < 0
                        }

                    for _, step in post_escape:
                        step_dict = step.to_dict()
                        vr = step_dict.get("verifier_response") or step_dict
                        is_feasible = vr.get("is_feasible", vr.get("feasible", False))
                        if is_feasible:
                            feasible_count += 1

                        # Check for new constraint violations (cascade)
                        post_slack = vr.get("constraint_slack") or {}
                        post_violations = {
                            name for name, val in post_slack.items()
                            if isinstance(val, (int, float)) and val < 0
                        }
                        new_violations = post_violations - seed_violations
                        if new_violations:
                            new_violations_seen = True

                    escape_quality = feasible_count / len(post_escape)
                    constraint_cascade = new_violations_seen

        return P3TaskResult(
            task_id=task["task_id"],
            runner_name=runner_name,
            trap_exited=trap_exited,
            explicit_replan=explicit_replan,
            dead_budget_rate=trap_proposals_before_exit / max(query_budget, 1),
            exit_step=exit_step,
            escape_quality=escape_quality,
            constraint_cascade=constraint_cascade,
        )

    def aggregate(self, task_results: list[P3TaskResult]) -> P3Summary:
        if not task_results:
            raise ValueError("Cannot aggregate empty P3 task results")
        runner_name = task_results[0].runner_name
        n_tasks = len(task_results)

        escaped = [r for r in task_results if r.trap_exited]
        escape_qualities = [r.escape_quality for r in escaped if r.escape_quality is not None]
        cascade_count = sum(1 for r in escaped if r.constraint_cascade)

        return P3Summary(
            runner_name=runner_name,
            n_tasks=n_tasks,
            trap_escape_rate=len(escaped) / n_tasks,
            dead_budget_rate=sum(r.dead_budget_rate for r in task_results) / n_tasks,
            explicit_replan_rate=sum(1 for r in task_results if r.explicit_replan) / n_tasks,
            escape_time=(
                statistics.mean(r.exit_step for r in escaped if r.exit_step is not None)
                if escaped
                else None
            ),
            escape_quality=(
                statistics.mean(escape_qualities) if escape_qualities else None
            ),
            constraint_cascade_rate=(
                cascade_count / len(escaped) if escaped else None
            ),
        )
