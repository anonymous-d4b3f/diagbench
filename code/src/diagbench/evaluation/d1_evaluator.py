"""
M1-T3: D1 Evaluator — computes the primary benchmark metrics from run results.

V2 metrics:
  feasible_rate            — fraction of tasks where model found a feasible design
  median_regret            — median normalized regret on feasible tasks only
  queries_to_feasible      — mean queries used on tasks that reached feasibility
  strict_attribution_rate  — fraction with strict_attribution=True
  best_so_far_auc          — V2 P3 core metric: area under best-feasible-obj curve
  first_feasible_step      — V2 P3: mean step index of first feasible proposal
"""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_REGRET_CAP = 1.0
_DEFAULT_BUDGET_LIMIT = 6    # matches task_bank.py query_budget default


@dataclass
class D1Result:
    task_id: str
    runner_name: str
    is_feasible: bool
    regret: float | None
    queries_used: int
    strict_attribution: bool
    objective_value: float | None
    bkf_objective_value: float | None
    first_proposal_objective: float | None = None
    first_proposal_is_feasible: bool | None = None
    first_proposal_regret: float | None = None
    # V2 new fields (optional — None in v1 results)
    objective_history: list[float | None] | None = None   # per-step feasible obj or None
    oracle_tier: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "D1Result":
        return cls(
            task_id=d["task_id"],
            runner_name=d.get("runner_name", "unknown"),
            is_feasible=bool(d["is_feasible"]),
            regret=float(d["regret"]) if d.get("regret") is not None else None,
            queries_used=int(d["queries_used"]),
            strict_attribution=bool(d.get("strict_attribution", False)),
            objective_value=d.get("objective_value"),
            bkf_objective_value=d.get("bkf_objective_value"),
            first_proposal_objective=d.get("first_proposal_objective"),
            first_proposal_is_feasible=d.get("first_proposal_is_feasible"),
            first_proposal_regret=(
                float(d["first_proposal_regret"])
                if d.get("first_proposal_regret") is not None
                else None
            ),
            objective_history=d.get("objective_history"),
            oracle_tier=d.get("oracle_tier"),
        )


def _compute_best_so_far_auc(
    objective_history: list[float | None],
    bkf_obj: float | None,
    budget: int,
) -> float | None:
    """
    Compute normalized best-so-far AUC for a single task.

    objective_history[i] = best feasible objective at step i (None if infeasible at step i).
    AUC is normalized to [0, 1] where 1.0 = reached RBKF on first step.

    Returns None if bkf_obj is missing or zero.
    """
    if not objective_history or bkf_obj is None or bkf_obj < 1e-12:
        return None

    # Build best-so-far curve over the budget
    padded = list(objective_history) + [None] * max(0, budget - len(objective_history))
    best_so_far = 0.0
    auc_sum = 0.0
    for step_val in padded:
        if step_val is not None:
            best_so_far = max(best_so_far, step_val)
        # Normalized best-so-far at this step
        auc_sum += best_so_far / bkf_obj

    return round(auc_sum / budget, 6) if budget > 0 else None


@dataclass
class D1Summary:
    runner_name: str
    n_tasks: int
    feasible_rate: float
    median_regret: float | None
    queries_to_feasible: float | None    # None if no feasible tasks
    strict_attribution_rate: float
    # V2 P3 metrics
    mean_best_so_far_auc: float | None   # mean across all tasks (None if no history)
    mean_first_feasible_step: float | None  # mean step of first feasible (feasible tasks only)

    def to_dict(self) -> dict:
        return {
            "runner_name": self.runner_name,
            "n_tasks": self.n_tasks,
            "feasible_rate": round(self.feasible_rate, 4),
            "median_regret": (
                round(self.median_regret, 4)
                if self.median_regret is not None
                else None
            ),
            "queries_to_feasible": (
                round(self.queries_to_feasible, 2)
                if self.queries_to_feasible is not None
                else None
            ),
            "strict_attribution_rate": round(self.strict_attribution_rate, 4),
            "mean_best_so_far_auc": (
                round(self.mean_best_so_far_auc, 4)
                if self.mean_best_so_far_auc is not None
                else None
            ),
            "mean_first_feasible_step": (
                round(self.mean_first_feasible_step, 2)
                if self.mean_first_feasible_step is not None
                else None
            ),
        }

    def to_table1_dict(self) -> dict:
        return {
            "runner_name": self.runner_name,
            "n_tasks": self.n_tasks,
            "feasible_rate": round(self.feasible_rate, 4),
            "median_regret": (
                round(self.median_regret, 4)
                if self.median_regret is not None
                else None
            ),
            "queries_to_feasible": (
                round(self.queries_to_feasible, 2)
                if self.queries_to_feasible is not None
                else None
            ),
            "strict_attribution_rate": round(self.strict_attribution_rate, 4),
        }


class D1Evaluator:
    """Evaluates a set of per-task run results against D1 + P3 metrics."""

    def __init__(
        self,
        regret_cap: float = _DEFAULT_REGRET_CAP,
        budget_limit: int = _DEFAULT_BUDGET_LIMIT,
    ) -> None:
        self.regret_cap = regret_cap
        self.budget_limit = budget_limit

    @classmethod
    def from_scoring_config(cls, config: dict) -> "D1Evaluator":
        return cls(
            regret_cap=float(config.get("regret_cap", _DEFAULT_REGRET_CAP)),
            budget_limit=int(config.get("budget_limit", _DEFAULT_BUDGET_LIMIT)),
        )

    def load_results(self, path: Path) -> list[D1Result]:
        results = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    results.append(D1Result.from_dict(json.loads(line)))
        return results

    def aggregate(self, results: list[D1Result]) -> D1Summary:
        if not results:
            raise ValueError("Cannot aggregate empty results")

        runner_name = results[0].runner_name
        n = len(results)

        feasible_results = [r for r in results if r.is_feasible]
        feasible_rate = len(feasible_results) / n

        feasible_regrets = [self._normalized_regret(r) for r in feasible_results]
        median_regret = statistics.median(feasible_regrets) if feasible_regrets else None

        queries_to_feasible = (
            statistics.mean(r.queries_used for r in feasible_results)
            if feasible_results
            else None
        )

        strict_attribution_rate = sum(1 for r in results if r.strict_attribution) / n

        # V2: best-so-far AUC (per task, then mean across tasks)
        auc_values = []
        for r in results:
            if r.objective_history is not None and r.bkf_objective_value is not None:
                auc = _compute_best_so_far_auc(
                    r.objective_history,
                    r.bkf_objective_value,
                    self.budget_limit,
                )
                if auc is not None:
                    auc_values.append(auc)
        mean_auc = statistics.mean(auc_values) if auc_values else None

        # V2: first feasible step (step index, 0-based; queries_used is 1-based)
        first_feasible_steps = [r.queries_used - 1 for r in feasible_results]
        mean_first_feasible = (
            statistics.mean(first_feasible_steps)
            if first_feasible_steps
            else None
        )

        return D1Summary(
            runner_name=runner_name,
            n_tasks=n,
            feasible_rate=feasible_rate,
            median_regret=median_regret,
            queries_to_feasible=queries_to_feasible,
            strict_attribution_rate=strict_attribution_rate,
            mean_best_so_far_auc=mean_auc,
            mean_first_feasible_step=mean_first_feasible,
        )

    def _normalized_regret(self, result: D1Result) -> float:
        if not result.is_feasible:
            raise ValueError(
                f"Infeasible result for task_id={result.task_id} should not enter regret aggregation"
            )
        if result.regret is None:
            raise ValueError(
                f"Feasible result for task_id={result.task_id} is missing regret"
            )
        return min(result.regret, self.regret_cap)


class Table1Exporter:
    """Exports a leaderboard table (Table 1) from a list of D1Summary objects."""

    COLUMNS = [
        "runner_name",
        "n_tasks",
        "feasible_rate",
        "median_regret",
        "queries_to_feasible",
        "strict_attribution_rate",
    ]

    def export_csv(self, summaries: list[D1Summary], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.COLUMNS)
            writer.writeheader()
            for s in summaries:
                writer.writerow(s.to_table1_dict())

    def export_json(self, summaries: list[D1Summary], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [s.to_table1_dict() for s in summaries]
        with path.open("w") as fh:
            json.dump({"table": "Table1", "rows": rows}, fh, indent=2)
