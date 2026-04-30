"""
P2b Evaluator: Final Design Quality.

Headline metric: mean_final_feasible_power_ratio = mean(P_T / P_ref) where
infeasible designs contribute 0.

This is the main score for P2 per the v3 benchmark blueprint.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


@dataclass
class P2BSummary:
    runner_name: str
    n_tasks: int
    mean_final_feasible_power_ratio: float  # headline metric
    feasible_rate: float
    mean_ratio_conditional: float | None  # mean ratio over feasible subset only

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner_name": self.runner_name,
            "n_tasks": self.n_tasks,
            "mean_final_feasible_power_ratio": round(self.mean_final_feasible_power_ratio, 6),
            "feasible_rate": round(self.feasible_rate, 4),
            "mean_ratio_conditional": (
                round(self.mean_ratio_conditional, 6)
                if self.mean_ratio_conditional is not None
                else None
            ),
        }


class P2BEvaluator:
    """
    Evaluates final design quality.

    Input per result row:
      {
        "task_id": str,
        "runner_name": str,
        "final_feasible": bool,
        "final_power": float | None,   # load power at final step (uW)
        "bkf_reference_power": float,   # BKF reference power (uW)
      }
    """

    def load_results(self, path: Path | str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with Path(path).open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def compute_ratio(self, row: dict[str, Any]) -> float:
        """Compute final feasible power ratio for a single result row."""
        if not row.get("final_feasible", False):
            return 0.0
        final_power = row.get("final_power")
        bkf_ref = row.get("bkf_reference_power")
        if final_power is None or bkf_ref is None or bkf_ref <= 0:
            return 0.0
        return float(final_power) / float(bkf_ref)

    def aggregate(self, rows: list[dict[str, Any]]) -> P2BSummary:
        if not rows:
            raise ValueError("Cannot aggregate empty P2b results")

        runner_name = rows[0].get("runner_name", "unknown")
        n_tasks = len(rows)

        ratios = [self.compute_ratio(row) for row in rows]
        feasible_ratios = [r for r in ratios if r > 0.0]

        feasible_count = sum(1 for row in rows if row.get("final_feasible", False))

        return P2BSummary(
            runner_name=runner_name,
            n_tasks=n_tasks,
            mean_final_feasible_power_ratio=statistics.mean(ratios) if ratios else 0.0,
            feasible_rate=feasible_count / n_tasks,
            mean_ratio_conditional=_mean(feasible_ratios),
        )
