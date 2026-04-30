"""
P4 Evaluator: Trade-off Ranking.

Headline metric: ranking_accuracy (Kendall tau)
Secondary metric: dominance_violation_rate
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _mean_bool(values: list[bool]) -> float | None:
    return _mean([1.0 if value else 0.0 for value in values]) if values else None


def _bool_field(row: dict[str, Any], key: str) -> bool | None:
    value = row.get(key)
    if value is None:
        return None
    return bool(value)


def _float_field(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    return float(value)


@dataclass
class P4Summary:
    runner_name: str
    n_tasks: int
    ranking_kendall_tau: float | None
    headline_metric_name: str
    headline_metric_value: float | None
    headline_metric_semantics: str
    dominance_violation_rate: float | None
    mean_dominated_pairs: float | None
    exact_match_rate: float | None
    top1_accuracy: float | None
    top2_set_accuracy: float | None
    policy_sensitive_pair_accuracy: float | None
    mean_policy_sensitive_pairs: float | None
    balanced_active_n_tasks: int
    balanced_active_bars: float | None
    balanced_active_ranking_kendall_tau: float | None
    balanced_active_exact_match_rate: float | None
    balanced_active_top1_accuracy: float | None
    balanced_active_top2_set_accuracy: float | None
    balanced_active_policy_sensitive_pair_accuracy: float | None
    parse_error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner_name": self.runner_name,
            "n_tasks": self.n_tasks,
            "ranking_kendall_tau": round(self.ranking_kendall_tau, 6) if self.ranking_kendall_tau is not None else None,
            "headline_metric_name": self.headline_metric_name,
            "headline_metric_value": round(self.headline_metric_value, 6) if self.headline_metric_value is not None else None,
            "headline_metric_semantics": self.headline_metric_semantics,
            "dominance_violation_rate": (
                round(self.dominance_violation_rate, 6) if self.dominance_violation_rate is not None else None
            ),
            "mean_dominated_pairs": (
                round(self.mean_dominated_pairs, 6) if self.mean_dominated_pairs is not None else None
            ),
            "exact_match_rate": round(self.exact_match_rate, 6) if self.exact_match_rate is not None else None,
            "top1_accuracy": round(self.top1_accuracy, 6) if self.top1_accuracy is not None else None,
            "top2_set_accuracy": round(self.top2_set_accuracy, 6) if self.top2_set_accuracy is not None else None,
            "policy_sensitive_pair_accuracy": (
                round(self.policy_sensitive_pair_accuracy, 6) if self.policy_sensitive_pair_accuracy is not None else None
            ),
            "mean_policy_sensitive_pairs": (
                round(self.mean_policy_sensitive_pairs, 6) if self.mean_policy_sensitive_pairs is not None else None
            ),
            "balanced_active_n_tasks": self.balanced_active_n_tasks,
            "balanced_active_bars": (
                round(self.balanced_active_bars, 6) if self.balanced_active_bars is not None else None
            ),
            "balanced_active_ranking_kendall_tau": (
                round(self.balanced_active_ranking_kendall_tau, 6)
                if self.balanced_active_ranking_kendall_tau is not None
                else None
            ),
            "balanced_active_exact_match_rate": (
                round(self.balanced_active_exact_match_rate, 6)
                if self.balanced_active_exact_match_rate is not None
                else None
            ),
            "balanced_active_top1_accuracy": (
                round(self.balanced_active_top1_accuracy, 6) if self.balanced_active_top1_accuracy is not None else None
            ),
            "balanced_active_top2_set_accuracy": (
                round(self.balanced_active_top2_set_accuracy, 6)
                if self.balanced_active_top2_set_accuracy is not None
                else None
            ),
            "balanced_active_policy_sensitive_pair_accuracy": (
                round(self.balanced_active_policy_sensitive_pair_accuracy, 6)
                if self.balanced_active_policy_sensitive_pair_accuracy is not None
                else None
            ),
            "parse_error_rate": round(self.parse_error_rate, 6),
        }


class P4Evaluator:
    @staticmethod
    def _balanced_active_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if bool(row.get("balanced_active_eval_eligible")):
                filtered.append(row)
                continue
            declared_profile = row.get("declared_profile")
            is_balanced = bool(row.get("is_balanced_view")) or declared_profile == "balanced"
            is_active = bool(row.get("is_active_policy_sensitive_row"))
            feasible_count = int(row.get("feasible_count", 0) or 0)
            if is_balanced and is_active and feasible_count >= 3:
                filtered.append(row)
        return filtered

    @staticmethod
    def _bars(
        *,
        tau: float | None,
        policy_pair_accuracy: float | None,
        exact_match_rate: float | None,
    ) -> float | None:
        if tau is None or policy_pair_accuracy is None or exact_match_rate is None:
            return None
        return 0.55 * tau + 0.25 * policy_pair_accuracy + 0.20 * exact_match_rate

    def load_results(self, path: Path | str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with Path(path).open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def aggregate(self, rows: list[dict[str, Any]]) -> P4Summary:
        if not rows:
            raise ValueError("Cannot aggregate empty P4 results")
        runner_name = rows[0].get("runner_name", "unknown")
        headline_metric_semantics = rows[0].get("headline_metric_semantics", "pareto")
        if headline_metric_semantics == "full":
            ranking_values = [float(row["full_kendall_tau"]) for row in rows if row.get("full_kendall_tau") is not None]
        else:
            ranking_values = [float(row["pareto_kendall_tau"]) for row in rows if row.get("pareto_kendall_tau") is not None]
        balanced_active_rows = self._balanced_active_rows(rows)
        balanced_active_tau = _mean(
            [float(row["full_kendall_tau"]) for row in balanced_active_rows if row.get("full_kendall_tau") is not None]
        )
        balanced_active_exact = _mean_bool(
            [bool(row["exact_match"]) for row in balanced_active_rows if row.get("exact_match") is not None]
        )
        balanced_active_top1 = _mean_bool(
            [bool(row["top1_accuracy"]) for row in balanced_active_rows if row.get("top1_accuracy") is not None]
        )
        balanced_active_top2 = _mean_bool(
            [bool(row["top2_set_accuracy"]) for row in balanced_active_rows if row.get("top2_set_accuracy") is not None]
        )
        balanced_active_policy = _mean(
            [
                float(row["policy_sensitive_pair_accuracy"])
                for row in balanced_active_rows
                if row.get("policy_sensitive_pair_accuracy") is not None
            ]
        )
        balanced_active_bars = self._bars(
            tau=balanced_active_tau,
            policy_pair_accuracy=balanced_active_policy,
            exact_match_rate=balanced_active_exact,
        )
        headline_metric_name = "ranking_kendall_tau"
        headline_metric_value = _mean(ranking_values)
        if headline_metric_semantics == "full" and balanced_active_bars is not None:
            headline_metric_name = "balanced_active_bars"
            headline_metric_value = balanced_active_bars
        return P4Summary(
            runner_name=runner_name,
            n_tasks=len(rows),
            ranking_kendall_tau=_mean(ranking_values),
            headline_metric_name=headline_metric_name,
            headline_metric_value=headline_metric_value,
            headline_metric_semantics=headline_metric_semantics,
            dominance_violation_rate=_mean(
                [float(row["pareto_violation_rate"]) for row in rows if row.get("pareto_violation_rate") is not None]
            ),
            mean_dominated_pairs=_mean([float(row["n_dominated_pairs"]) for row in rows if row.get("n_dominated_pairs") is not None]),
            exact_match_rate=_mean_bool([bool(row["exact_match"]) for row in rows if row.get("exact_match") is not None]),
            top1_accuracy=_mean_bool([bool(row["top1_accuracy"]) for row in rows if row.get("top1_accuracy") is not None]),
            top2_set_accuracy=_mean(
                [1.0 if row.get("top2_set_accuracy") else 0.0 for row in rows if row.get("top2_set_accuracy") is not None]
            ),
            policy_sensitive_pair_accuracy=_mean(
                [
                    float(row["policy_sensitive_pair_accuracy"])
                    for row in rows
                    if row.get("policy_sensitive_pair_accuracy") is not None
                ]
            ),
            mean_policy_sensitive_pairs=_mean(
                [float(row["policy_sensitive_pair_count"]) for row in rows if row.get("policy_sensitive_pair_count") is not None]
            ),
            balanced_active_n_tasks=len(balanced_active_rows),
            balanced_active_bars=balanced_active_bars,
            balanced_active_ranking_kendall_tau=balanced_active_tau,
            balanced_active_exact_match_rate=balanced_active_exact,
            balanced_active_top1_accuracy=balanced_active_top1,
            balanced_active_top2_set_accuracy=balanced_active_top2,
            balanced_active_policy_sensitive_pair_accuracy=balanced_active_policy,
            parse_error_rate=sum(1 for row in rows if row.get("is_parse_error")) / len(rows),
        )
