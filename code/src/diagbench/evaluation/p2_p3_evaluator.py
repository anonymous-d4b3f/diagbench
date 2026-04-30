"""
V3 evaluator for P2a Physical Anchoring and P2c Trajectory Improvement.

P2a is computed from main-loop result rows:
  - first_proposal_regret
  - first_proposal_is_feasible

P2c combines main-loop result rows with trajectory-derived diagnostics:
  - mean_best_so_far_auc
  - queries_to_feasible
  - final_regret
  - monotonicity_score
  - violation_reduction_consistency
  - normalized_improvement_per_query
  - improvement_rate (new in v3: fraction of steps where U_(t+1) > U_t)
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diagbench.evaluation.d1_evaluator import D1Evaluator, D1Result
from diagbench.probes.trajectory_logger import TrajectoryLogger


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _safe_round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _compute_utility(
    power: float,
    p_ref: float,
    v_freq: float = 0.0,
    v_stress: float = 0.0,
    v_disp: float = 0.0,
    *,
    lambda_: float = 0.5,
    w_f: float = 1.0,
    w_s: float = 1.0,
    w_d: float = 1.0,
) -> float:
    """Compute unified trajectory utility U_t per v3 blueprint §6.3."""
    if p_ref <= 0:
        return 0.0
    v_total = w_f * v_freq + w_s * v_stress + w_d * v_disp
    return (power / p_ref) - lambda_ * v_total


@dataclass
class P2ASummary:
    runner_name: str
    n_tasks: int
    initial_feasible_rate: float
    mean_initial_regret: float | None
    excellent_rate: float
    good_or_better_rate: float
    infeasible_first_step_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner_name": self.runner_name,
            "n_tasks": self.n_tasks,
            "initial_feasible_rate": _safe_round(self.initial_feasible_rate, 4),
            "mean_initial_regret": _safe_round(self.mean_initial_regret, 4),
            "excellent_rate": _safe_round(self.excellent_rate, 4),
            "good_or_better_rate": _safe_round(self.good_or_better_rate, 4),
            "infeasible_first_step_rate": _safe_round(self.infeasible_first_step_rate, 4),
        }


@dataclass
class P2CSummary:
    """P2c Trajectory Improvement summary."""
    runner_name: str
    n_tasks: int
    mean_best_so_far_auc: float | None
    mean_queries_to_feasible: float | None
    median_final_regret: float | None
    monotonicity_score: float | None
    violation_reduction_consistency: float | None
    normalized_improvement_per_query: float | None
    improvement_rate: float | None  # fraction of steps where U_(t+1) > U_t

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner_name": self.runner_name,
            "n_tasks": self.n_tasks,
            "mean_best_so_far_auc": _safe_round(self.mean_best_so_far_auc, 4),
            "mean_queries_to_feasible": _safe_round(self.mean_queries_to_feasible, 2),
            "median_final_regret": _safe_round(self.median_final_regret, 4),
            "monotonicity_score": _safe_round(self.monotonicity_score, 4),
            "violation_reduction_consistency": _safe_round(self.violation_reduction_consistency, 4),
            "normalized_improvement_per_query": _safe_round(self.normalized_improvement_per_query, 4),
            "improvement_rate": _safe_round(self.improvement_rate, 4),
        }


class P2P3Evaluator:
    """Builds v2-specific P2-A and P3 summaries from main-loop artifacts."""

    def __init__(self, *, budget_limit: int = 10) -> None:
        self._d1 = D1Evaluator(budget_limit=budget_limit)

    def load_results(self, path: Path | str) -> list[D1Result]:
        return self._d1.load_results(Path(path))

    def load_trajectories(self, path: Path | str) -> dict[str, Any]:
        trajectories = TrajectoryLogger.load_batch(path)
        return {trajectory.task_id: trajectory for trajectory in trajectories}

    def aggregate_p2a(self, results: list[D1Result]) -> P2ASummary:
        if not results:
            raise ValueError("Cannot aggregate empty result set for P2-A")

        runner_name = results[0].runner_name
        n_tasks = len(results)
        feasible_initial = [r for r in results if r.first_proposal_is_feasible is True]
        feasible_rate = len(feasible_initial) / n_tasks

        initial_regrets = [
            r.first_proposal_regret
            for r in feasible_initial
            if r.first_proposal_regret is not None
        ]
        excellent_rate = (
            sum(1 for value in initial_regrets if value <= 0.1) / n_tasks
        )
        good_or_better_rate = (
            sum(1 for value in initial_regrets if value <= 0.3) / n_tasks
        )
        infeasible_rate = sum(1 for r in results if r.first_proposal_is_feasible is False) / n_tasks

        return P2ASummary(
            runner_name=runner_name,
            n_tasks=n_tasks,
            initial_feasible_rate=feasible_rate,
            mean_initial_regret=_mean(initial_regrets),
            excellent_rate=excellent_rate,
            good_or_better_rate=good_or_better_rate,
            infeasible_first_step_rate=infeasible_rate,
        )

    def aggregate_p2c(
        self,
        *,
        results: list[D1Result],
        trajectories_by_task_id: dict[str, Any] | None = None,
    ) -> P2CSummary:
        if not results:
            raise ValueError("Cannot aggregate empty result set for P2c")

        d1_summary = self._d1.aggregate(results)
        runner_name = results[0].runner_name
        n_tasks = len(results)

        monotonicity_values: list[float] = []
        violation_reduction_values: list[float] = []
        normalized_improvement_values: list[float] = []

        trajectory_map = trajectories_by_task_id or {}
        result_map = {result.task_id: result for result in results}

        for task_id, result in result_map.items():
            if (
                result.first_proposal_objective is not None
                and result.objective_value is not None
                and result.bkf_objective_value not in (None, 0)
                and result.queries_used > 0
            ):
                improvement = max(0.0, result.objective_value - result.first_proposal_objective)
                normalized_improvement_values.append(
                    improvement / (float(result.bkf_objective_value) * float(result.queries_used))
                )

            trajectory = trajectory_map.get(task_id)
            if trajectory is None:
                continue

            monotonicity = self._trajectory_monotonicity(trajectory)
            if monotonicity is not None:
                monotonicity_values.append(monotonicity)

            violation_consistency = self._violation_reduction_consistency(trajectory)
            if violation_consistency is not None:
                violation_reduction_values.append(violation_consistency)

        # Compute improvement_rate from trajectories using unified utility
        improvement_rates: list[float] = []
        for task_id, result in result_map.items():
            trajectory = trajectory_map.get(task_id)
            if trajectory is None:
                continue
            rate = self._trajectory_improvement_rate(trajectory, result)
            if rate is not None:
                improvement_rates.append(rate)

        return P2CSummary(
            runner_name=runner_name,
            n_tasks=n_tasks,
            mean_best_so_far_auc=d1_summary.mean_best_so_far_auc,
            mean_queries_to_feasible=d1_summary.queries_to_feasible,
            median_final_regret=d1_summary.median_regret,
            monotonicity_score=_mean(monotonicity_values),
            violation_reduction_consistency=_mean(violation_reduction_values),
            normalized_improvement_per_query=_mean(normalized_improvement_values),
            improvement_rate=_mean(improvement_rates),
        )

    def _trajectory_monotonicity(self, trajectory: Any) -> float | None:
        objectives = [
            value
            for value in trajectory.objective_per_step()
            if value is not None
        ]
        if len(objectives) < 2:
            return None
        decreases = sum(
            1
            for index in range(1, len(objectives))
            if objectives[index] < objectives[index - 1]
        )
        return 1.0 - decreases / (len(objectives) - 1)

    def _violation_reduction_consistency(self, trajectory: Any) -> float | None:
        steps = [
            step
            for step in trajectory.steps
            if step.action_type == "propose_design"
        ]
        if len(steps) < 2:
            return None

        consistent = 0
        total = 0
        for current_step, next_step in zip(steps, steps[1:]):
            current_slack = current_step.to_dict().get("constraint_slack") or {}
            next_slack = next_step.to_dict().get("constraint_slack") or {}
            violated = [
                (name, float(value))
                for name, value in current_slack.items()
                if isinstance(value, (int, float)) and value < 0
            ]
            if not violated:
                continue
            dominant_name, dominant_value = min(violated, key=lambda item: item[1])
            if dominant_name not in next_slack or not isinstance(next_slack[dominant_name], (int, float)):
                continue
            total += 1
            if float(next_slack[dominant_name]) > dominant_value:
                consistent += 1
        if total == 0:
            return None
        return consistent / total

    def _trajectory_improvement_rate(self, trajectory: Any, result: D1Result) -> float | None:
        """Compute fraction of steps where U_(t+1) > U_t using unified utility."""
        steps = [
            step
            for step in trajectory.steps
            if step.action_type == "propose_design"
        ]
        if len(steps) < 2:
            return None

        p_ref = float(result.bkf_objective_value) if result.bkf_objective_value else None
        if p_ref is None or p_ref <= 0:
            return None

        utilities: list[float] = []
        for step in steps:
            step_dict = step.to_dict()
            power = float(step_dict.get("objective_value", 0.0) or 0.0)
            slack = step_dict.get("constraint_slack") or {}
            # Extract normalized violations (negative slack = violation)
            v_freq = max(0.0, -float(slack.get("freq_error_pct_limit", 0.0)))
            v_stress = max(0.0, -float(slack.get("stress_limit_mpa", 0.0)))
            v_disp = max(0.0, -float(slack.get("disp_limit_mm", 0.0)))
            u = _compute_utility(power, p_ref, v_freq, v_stress, v_disp)
            utilities.append(u)

        improvements = sum(
            1 for i in range(1, len(utilities)) if utilities[i] > utilities[i - 1]
        )
        return improvements / (len(utilities) - 1)


def load_result_dicts(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
