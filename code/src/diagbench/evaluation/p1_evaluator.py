"""
P1 Evaluator: Feasibility Recognition.

Computes:
  - p1_3class_macro_f1: Macro-F1 over 3 top-level labels (main paper metric)
      feasible / infeasible / underspecified
  - p1_6subtype_macro_f1: equal-weight F1 across P1 subtypes (appendix;
      field name is kept for backward compatibility)
  - false_refusal_rate: solvable tasks incorrectly declared infeasible
  - clarification_precision: when model asks, are the requested fields blocking?
  - (appendix) P1_weighted_F1: engineering-weighted version

Gold label mapping (6-subtype):
  solvable_wide/solvable_narrow/underspecified_nonkey → propose_design
  infeasible_*/                                       → declare_infeasible
  underspecified_key                                  → request_missing_info
  p1_v2 aliases:
    solvable_anchor/solvable_tight                    → propose_design
    infeasible_disguised/infeasible_structural        → declare_infeasible
  p1_v3 aliases:
    solvable_base/solvable_boundary/solvable_red_herring/
      missing_nonblocker                              → propose_design
    infeasible_margin/infeasible_structural           → declare_infeasible
    missing_blocker_obvious                           → request_missing_info
    missing_blocker_ambiguous                         → mixed per-row gold

3-class collapse:
  solvable_*, underspecified_nonkey                     → feasible
  infeasible_*                                          → infeasible
  underspecified_key                                    → underspecified
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Gold action expected for each subtype
_GOLD_ACTION = {
    "solvable_wide":            "propose_design",
    "solvable_narrow":          "propose_design",
    "solvable_anchor":          "propose_design",
    "solvable_tight":           "propose_design",
    "underspecified_nonkey":    "propose_design",
    "solvable_base":            "propose_design",
    "solvable_boundary":        "propose_design",
    "solvable_red_herring":     "propose_design",
    "missing_nonblocker":       "propose_design",
    "infeasible_hard_conflict": "declare_infeasible",
    "infeasible_by_margin":     "declare_infeasible",
    "infeasible_disguised":     "declare_infeasible",
    "infeasible_structural":    "declare_infeasible",
    "infeasible_margin":        "declare_infeasible",
    "underspecified_key":       "request_missing_info",
    "missing_blocker_obvious":  "request_missing_info",
    # P1 v3 intentionally mixes request_missing_info/propose_design inside
    # this subtype. Aggregation must use each row's gold_action instead.
    "missing_blocker_ambiguous": "request_missing_info",
}

# Solvable subtypes (false refusal is measured here)
_SOLVABLE_SUBTYPES = {
    "solvable_wide",
    "solvable_narrow",
    "solvable_anchor",
    "solvable_tight",
    "underspecified_nonkey",
    "solvable_base",
    "solvable_boundary",
    "solvable_red_herring",
    "missing_nonblocker",
}

# 3-class collapse: subtype → top-level label
_SUBTYPE_TO_3CLASS = {
    "solvable_wide":            "feasible",
    "solvable_narrow":          "feasible",
    "solvable_anchor":          "feasible",
    "solvable_tight":           "feasible",
    "underspecified_nonkey":    "feasible",
    "solvable_base":            "feasible",
    "solvable_boundary":        "feasible",
    "solvable_red_herring":     "feasible",
    "missing_nonblocker":       "feasible",
    "infeasible_hard_conflict": "infeasible",
    "infeasible_by_margin":     "infeasible",
    "infeasible_disguised":     "infeasible",
    "infeasible_structural":    "infeasible",
    "infeasible_margin":        "infeasible",
    "underspecified_key":       "underspecified",
    "missing_blocker_obvious":  "underspecified",
    # Predominant class only; per-row gold_action takes precedence below.
    "missing_blocker_ambiguous": "underspecified",
}

# 3-class gold action mapping
_3CLASS_GOLD_ACTION = {
    "feasible":       "propose_design",
    "infeasible":     "declare_infeasible",
    "underspecified": "request_missing_info",
}

# Engineering-weighted F1 weights (disclosed — see BENCHMARK_V2_BLUEPRINT.md appendix)
_ENGINEERING_WEIGHTS = {
    "solvable_wide":            0.5,
    "solvable_narrow":          2.0,
    "solvable_anchor":          0.5,
    "solvable_tight":           2.0,
    "solvable_base":            0.75,
    "solvable_boundary":        2.0,
    "solvable_red_herring":     2.0,
    "infeasible_hard_conflict": 1.5,
    "infeasible_by_margin":     3.0,
    "infeasible_disguised":     3.0,
    "infeasible_structural":    1.5,
    "infeasible_margin":        3.0,
    "underspecified_key":       2.0,
    "underspecified_nonkey":    0.5,
    "missing_blocker_obvious":  1.5,
    "missing_blocker_ambiguous": 2.0,
    "missing_nonblocker":       0.75,
}


_ACTION_TO_3CLASS = {
    "propose_design": "feasible",
    "declare_infeasible": "infeasible",
    "request_missing_info": "underspecified",
}


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    """Compute F1 score with zero-division guard."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _recognition_action(resp: dict) -> str:
    """Return the action intended by the model before candidate validation."""
    parsed = resp.get("parsed_action_type")
    if isinstance(parsed, str) and parsed:
        return parsed
    pred = resp.get("predicted_action", "unknown")
    if pred == "invalid_candidate":
        # invalid_candidate is emitted after a parsed propose_design could not
        # be converted to a verifier candidate. It should not pollute P1
        # problem-recognition intent metrics.
        return "propose_design"
    return pred


def _gold_3class(resp: dict) -> str | None:
    gold_action = resp.get("gold_action")
    if isinstance(gold_action, str) and gold_action in _ACTION_TO_3CLASS:
        return _ACTION_TO_3CLASS[gold_action]
    subtype = resp.get("p1_subtype", "unknown")
    return _SUBTYPE_TO_3CLASS.get(subtype)


def _pred_3class(resp: dict) -> str:
    return _ACTION_TO_3CLASS.get(_recognition_action(resp), "unknown")


@dataclass
class P1SubtypeResult:
    subtype: str
    n: int
    n_correct: int
    f1: float
    precision: float
    recall: float


@dataclass
class P1Summary:
    runner_name: str
    n_tasks: int
    p1_3class_macro_f1: float  # primary paper metric
    p1_6subtype_macro_f1: float  # appendix metric
    p1_weighted_f1: float
    subtype_results: list[P1SubtypeResult]
    false_refusal_rate: float | None   # % solvable tasks incorrectly refused
    clarification_precision: float | None  # % of requested fields that are truly blocking

    @property
    def p1_macro_f1(self) -> float:
        """Backward-compatible alias for older callers/tests."""
        return self.p1_3class_macro_f1

    def to_dict(self) -> dict:
        return {
            "runner_name": self.runner_name,
            "n_tasks": self.n_tasks,
            "p1_3class_macro_f1": round(self.p1_3class_macro_f1, 4),
            "p1_6subtype_macro_f1": round(self.p1_6subtype_macro_f1, 4),
            "p1_weighted_f1": round(self.p1_weighted_f1, 4),
            "false_refusal_rate": (
                round(self.false_refusal_rate, 4) if self.false_refusal_rate is not None else None
            ),
            "clarification_precision": (
                round(self.clarification_precision, 4)
                if self.clarification_precision is not None
                else None
            ),
            "subtype_breakdown": {
                r.subtype: {"f1": round(r.f1, 4), "n": r.n, "n_correct": r.n_correct}
                for r in self.subtype_results
            },
        }


class P1Evaluator:
    """
    Evaluates model responses on P1 problem recognition tasks.

    Input format per response dict:
      {
        "task_id": str,
        "p1_subtype": str,
        "gold_action": str,
        "predicted_action": str,
        "runner_name": str,
        "missing_fields_ground_truth": [...],  # optional
        "predicted_missing_fields": [...],      # optional
        "confidence": float | None,
      }
    """

    def evaluate_response(self, task: dict, action: dict) -> dict:
        """
        Evaluate a single P1 response.

        Args:
            task: P1 task dict (with gold_label, p1_subtype).
            action: Parsed action dict from the model.

        Returns:
            Result dict with is_correct, predicted_action, etc.
        """
        gold = task["gold_label"]["action_type"]
        predicted = action.get("action_type", "unknown")
        is_correct = gold == predicted

        # Clarification precision: if model requests info, check if the fields are blocking
        predicted_missing: list[str] = []
        clarification_precision: float | None = None
        if predicted == "request_missing_info":
            predicted_missing = action.get("missing_fields", [])
            blocking = set(task.get("missing_fields_ground_truth", []))
            if predicted_missing:
                correct_asks = sum(1 for f in predicted_missing if f in blocking)
                clarification_precision = correct_asks / len(predicted_missing)

        return {
            "task_id": task["task_id"],
            "p1_subtype": task.get("p1_subtype", "unknown"),
            "gold_action": gold,
            "predicted_action": predicted,
            "parsed_action_type": predicted,
            "is_correct": is_correct,
            "runner_name": action.get("runner_name", "unknown"),
            "confidence": action.get("confidence"),
            "clarification_precision": clarification_precision,
            "predicted_missing_fields": predicted_missing,
        }

    def aggregate(self, responses: list[dict]) -> P1Summary:
        """
        Aggregate a list of per-task evaluation responses into P1Summary.

        Args:
            responses: List of dicts from evaluate_response().

        Returns:
            P1Summary with macro F1, weighted F1, and false_refusal_rate.
        """
        if not responses:
            raise ValueError("Cannot aggregate empty P1 responses")

        runner_name = responses[0].get("runner_name", "unknown")
        n_total = len(responses)

        # Group by subtype
        by_subtype: dict[str, list[dict]] = {st: [] for st in _GOLD_ACTION}
        for resp in responses:
            st = resp.get("p1_subtype", "unknown")
            by_subtype.setdefault(st, []).append(resp)

        # Per-subtype recognition F1. P1 v3 contains mixed-gold subtypes, so
        # this uses each row's gold_action instead of assuming one label per
        # subtype.
        subtype_results: list[P1SubtypeResult] = []
        for subtype in sorted(by_subtype):
            group = by_subtype[subtype]
            n = len(group)
            if n == 0:
                subtype_results.append(P1SubtypeResult(subtype, 0, 0, 0.0, 0.0, 0.0))
                continue
            labels = sorted(
                {
                    r.get("gold_action")
                    for r in group
                    if r.get("gold_action") in _ACTION_TO_3CLASS
                }
            )
            per_label_f1: list[float] = []
            per_label_precision: list[float] = []
            per_label_recall: list[float] = []
            for label in labels:
                tp = sum(
                    1 for r in group
                    if r.get("gold_action") == label and _recognition_action(r) == label
                )
                fp = sum(
                    1 for r in group
                    if r.get("gold_action") != label and _recognition_action(r) == label
                )
                fn = sum(
                    1 for r in group
                    if r.get("gold_action") == label and _recognition_action(r) != label
                )
                per_label_f1.append(_safe_f1(tp, fp, fn))
                per_label_precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
                per_label_recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

            n_correct = sum(1 for r in group if _recognition_action(r) == r.get("gold_action"))
            f1 = statistics.mean(per_label_f1) if per_label_f1 else 0.0
            precision = statistics.mean(per_label_precision) if per_label_precision else 0.0
            recall = statistics.mean(per_label_recall) if per_label_recall else 0.0
            subtype_results.append(P1SubtypeResult(subtype, n, n_correct, f1, precision, recall))

        # Macro F1 (6-subtype, appendix)
        f1_values = [r.f1 for r in subtype_results if r.n > 0]
        macro_f1_6subtype = statistics.mean(f1_values) if f1_values else 0.0

        # 3-class Macro F1 (primary paper metric)
        macro_f1_3class = self._compute_3class_macro_f1(responses)

        # Weighted F1 (appendix)
        total_weight = sum(_ENGINEERING_WEIGHTS.get(r.subtype, 1.0) for r in subtype_results if r.n > 0)
        weighted_f1 = (
            sum(_ENGINEERING_WEIGHTS.get(r.subtype, 1.0) * r.f1 for r in subtype_results if r.n > 0)
            / total_weight
            if total_weight > 0 else 0.0
        )

        # False refusal rate (solvable tasks wrongly declared infeasible)
        solvable_resps = [
            r for r in responses
            if r.get("gold_action") == "propose_design"
            or r.get("p1_subtype") in _SOLVABLE_SUBTYPES
        ]
        if solvable_resps:
            n_refused = sum(1 for r in solvable_resps if _recognition_action(r) == "declare_infeasible")
            false_refusal_rate = n_refused / len(solvable_resps)
        else:
            false_refusal_rate = None

        # Clarification precision
        clarif_values = [
            r["clarification_precision"]
            for r in responses
            if r.get("clarification_precision") is not None
        ]
        clarification_precision = statistics.mean(clarif_values) if clarif_values else None

        return P1Summary(
            runner_name=runner_name,
            n_tasks=n_total,
            p1_3class_macro_f1=round(macro_f1_3class, 6),
            p1_6subtype_macro_f1=round(macro_f1_6subtype, 6),
            p1_weighted_f1=round(weighted_f1, 6),
            subtype_results=subtype_results,
            false_refusal_rate=false_refusal_rate,
            clarification_precision=clarification_precision,
        )

    def load_responses(self, path: Path) -> list[dict]:
        rows = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _compute_3class_macro_f1(self, responses: list[dict]) -> float:
        """Compute Macro-F1 over 3 top-level labels: feasible, infeasible, underspecified."""
        gold_3class_list: list[str] = []
        pred_3class_list: list[str] = []
        for resp in responses:
            gold_3 = _gold_3class(resp)
            if gold_3 is None:
                continue
            gold_3class_list.append(gold_3)
            pred_3class_list.append(_pred_3class(resp))

        if not gold_3class_list:
            return 0.0

        # Compute per-class F1 and average
        f1_values: list[float] = []
        for label in ("feasible", "infeasible", "underspecified"):
            tp = sum(1 for g, p in zip(gold_3class_list, pred_3class_list) if g == label and p == label)
            fp = sum(1 for g, p in zip(gold_3class_list, pred_3class_list) if g != label and p == label)
            fn = sum(1 for g, p in zip(gold_3class_list, pred_3class_list) if g == label and p != label)
            n_class = sum(1 for g in gold_3class_list if g == label)
            if n_class > 0:
                f1_values.append(_safe_f1(tp, fp, fn))

        return statistics.mean(f1_values) if f1_values else 0.0
