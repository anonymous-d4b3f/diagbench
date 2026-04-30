#!/usr/bin/env python3
"""Extract stable response-control profile metrics from existing P1-P4 runs.

This script is intentionally artifact-only: it does not call models or rerun
evaluations. It reads the current curated/main run JSONL files, emits raw
profile metrics, dimension-level scores, flat CSV tables, and a small LaTeX
row file that can be included in the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagbench.evaluation.p1_evaluator import P1Evaluator
from diagbench.evaluation.p3_evaluator import P3Evaluator
from diagbench.evaluation.p4_evaluator import P4Evaluator


SPLITS = ("dev", "test_id", "test_ood")
STAGES = ("p1_v3r4", "p2_v3r1", "p3_v3r1", "p4_full_v2")
ACTIONS = ("propose_design", "declare_infeasible", "request_missing_info")
EXPECTED_STAGE_ROWS = {
    "p1": 240,
    "p2": 208,
    "p3": 156,
    "p4": 159,
}

CORE_BUNDLE = ROOT / "artifacts" / "curated" / "main_table_bundle_20260417" / "eval_data"
CORE_STAGE_ROOTS = {
    "p1_v3r4": CORE_BUNDLE / "P1_v3r4",
    "p2_v3r1": CORE_BUNDLE / "P2_v3r1",
    "p3_v3r1": CORE_BUNDLE / "P3_v3r1",
    "p4_full_v2": CORE_BUNDLE / "P4_full_v2",
}
P2_TASK_DIR = ROOT / "data" / "p2_v3r1" / "splits"
P3_TASK_DIR = ROOT / "data" / "p3_v3r1"
P4_TASK_DIR = ROOT / "data" / "p4_full_v2"
P3_INTERVENTION_SUMMARY = ROOT / "results" / "analysis" / "p3_intervention_full_20260421_clean" / "p3_intervention_summary.json"

EXTENSION_ROOTS = {
    "model_K": ROOT / "artifacts" / "runs" / "full_20260425_deepseek_mimo_p1_p4" / "model_K",
    "model_L": ROOT / "artifacts" / "runs" / "full_20260425_deepseek_mimo_p1_p4" / "model_L",
    "model_M": (
        ROOT
        / "artifacts"
        / "runs"
        / "full_20260427_model_M_p1_p4"
        / "analysis"
        / "merged_clean_final_20260427"
    ),
}

ROW_REPAIR_DIRS = {
    # Mimo P3 test_ood initially had 14 HTTP-402 invalid rows. These retry rows
    # are the paper-facing repair view; merge by task_id and keep retry rows.
    ("model_L", "p3_v3r1", "test_ood"): (
        ROOT
        / "artifacts"
        / "runs"
        / "full_20260425_deepseek_mimo_p1_p4"
        / "model_L"
        / "retries"
        / "http402_invalid_output"
        / "p3_v3r1"
        / "test_ood",
    ),
}

DISPLAY_NAMES = {
    "model_E": "model_E",
    "model_D": "model_D",
    "model_H": "model_H",
    "model_K": "model_K",
    "gemini_3_1_pro": "model_B",
    "model_B": "model_B",
    "model_J": "model_J",
    "model_C": "model_C",
    "model_C_mini": "model_I",
    "model_M": "model_M",
    "llama_3_3_70b": "Llama-3.3-70B",
    "o4_mini": "model_F",
    "qwen3_6_plus": "model_G",
    "model_A": "model_A",
    "model_L": "model_L",
}

P1_GOLD_ACTION = {
    "solvable_wide": "propose_design",
    "solvable_narrow": "propose_design",
    "solvable_anchor": "propose_design",
    "solvable_tight": "propose_design",
    "underspecified_nonkey": "propose_design",
    "solvable_base": "propose_design",
    "solvable_boundary": "propose_design",
    "solvable_red_herring": "propose_design",
    "missing_nonblocker": "propose_design",
    "infeasible_hard_conflict": "declare_infeasible",
    "infeasible_by_margin": "declare_infeasible",
    "infeasible_disguised": "declare_infeasible",
    "infeasible_structural": "declare_infeasible",
    "infeasible_margin": "declare_infeasible",
    "underspecified_key": "request_missing_info",
    "missing_blocker_obvious": "request_missing_info",
    "missing_blocker_ambiguous": "request_missing_info",
}

# P1 was run before the later "preview" naming stabilized.
CORE_MODEL_ALIASES = {
    ("p1_v3r4", "gemini_3_1_pro"): "model_B",
}

CORE_STAGE_MODEL_OVERRIDES = {
    # The curated bundle accidentally retained a 50-row partial P4 artifact for
    # model_F. This run is the complete 93/36/30 P4-full-v2 repair used in the
    # paper-facing snapshot.
    ("o4_mini", "p4_full_v2"): ROOT / "artifacts" / "runs" / "p4_full_v2_live_20260416_model_F_hi",
}

DIMENSION_METRICS = {
    "action_prior": (
        "p1.action_distribution_alignment",
        "p1.macro_f1",
        "p1.missing_recall",
        "p1.infeasible_recall",
        "p1.propose_precision",
        "p1.non_invalid_rate",
    ),
    "edit_style": (
        "p2.bounded_local_edit_rate",
        "p2.feasibility_preservation_rate",
        "p2.directed_repair_rate",
        "p2.final_feasible_rate",
        "p2.non_destructive_edit_rate",
        "p2.protocol_valid_rate",
    ),
    "feedback_obedience": (
        "p2.violation_reduction_consistency",
        "p2.utility_improvement_rate",
        "p2.mean_best_so_far_auc",
        "p3.violation_reduction_consistency",
        "p3.post_feedback_feasible_rate",
    ),
    "state_trust": (
        "p3.trap_escape_rate",
        "p3.explicit_replan_rate",
        "p3.escape_quality",
        "p3.non_cascade_rate",
        "p3.non_dead_budget_rate",
        "p3.summary_success_delta",
        "p3.summary_cascade_reduction",
    ),
    "preference_execution": (
        "p4.full_tau_scaled",
        "p4.balanced_active_bars",
        "p4.balanced_active_policy_sensitive_pair_accuracy",
        "p4.exact_match_rate",
        "p4.top1_accuracy",
        "p4.non_pareto_violation_rate",
        "p4.non_parse_error_rate",
    ),
}

STAGE_TARGETS = {
    "p1_composite": "p1.composite",
    "p2_power_ratio": "p2.final_feasible_power_ratio",
    "p3_success": "p3.success_rate",
    "p3_trap_escape": "p3.trap_escape_rate",
    "p4_bars": "p4.balanced_active_bars",
    "p4_full_tau": "p4.full_tau",
}


@dataclass
class ModelArtifacts:
    model_key: str
    display_name: str
    source_group: str
    stage_dirs: dict[str, dict[str, Path]] = field(default_factory=dict)

    def add_split_dir(self, stage: str, split: str, path: Path) -> None:
        self.stage_dirs.setdefault(stage, {})[split] = path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def maybe_load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path) if path.exists() else []


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def safe_mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.mean(clean) if clean else None


def safe_median(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(clean) if clean else None


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def clamp01(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return max(0.0, min(1.0, value))


def bool_mean(values: list[Any]) -> float | None:
    clean = [bool(value) for value in values if value is not None]
    return safe_mean([1.0 if value else 0.0 for value in clean])


def round_float(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, digits)
    if isinstance(value, dict):
        return {key: round_float(val, digits) for key, val in value.items()}
    if isinstance(value, list):
        return [round_float(item, digits) for item in value]
    return value


def action_for_row(row: dict[str, Any]) -> str:
    parsed = row.get("parsed_action_type")
    if isinstance(parsed, str) and parsed:
        return parsed
    predicted = row.get("predicted_action")
    if predicted == "invalid_candidate":
        return "propose_design"
    if isinstance(predicted, str) and predicted:
        return predicted
    return "invalid_output"


def gold_action_for_row(row: dict[str, Any]) -> str:
    gold = row.get("gold_action")
    if isinstance(gold, str) and gold:
        return gold
    return P1_GOLD_ACTION.get(str(row.get("p1_subtype")), "unknown")


def normalized_entropy(counts: dict[str, int]) -> float | None:
    total = sum(counts.values())
    if total <= 0:
        return None
    probs = [count / total for count in counts.values() if count > 0]
    if len(counts) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(counts))


def distribution_alignment(pred: dict[str, float], gold: dict[str, float]) -> float:
    l1 = sum(abs(pred.get(action, 0.0) - gold.get(action, 0.0)) for action in ACTIONS)
    return clamp01(1.0 - 0.5 * l1) or 0.0


def result_candidates_for_stage(split_dir: Path, stage: str) -> tuple[Path, ...]:
    candidates = {
        "p1_v3r4": ("p1v2_results.jsonl", "p1_results.jsonl"),
        "p2_v3r1": ("main_results.jsonl",),
        "p3_v3r1": ("p3_results.jsonl",),
        "p4_full_v2": ("p4_results.jsonl",),
    }[stage]
    return tuple(split_dir / name for name in candidates if (split_dir / name).exists())


def result_path_for_stage(split_dir: Path, stage: str) -> Path | None:
    candidates = result_candidates_for_stage(split_dir, stage)
    return candidates[0] if candidates else None


def load_best_result_rows(split_dir: Path, stage: str) -> tuple[Path | None, list[dict[str, Any]], dict[str, int]]:
    """Choose the fullest candidate result file under a split directory.

    Several historical P1 reruns left both full and partial result files in the
    same split. The stable rule is to load all same-stage candidates, select the
    largest one, and record all candidate counts in provenance.
    """
    candidate_counts: dict[str, int] = {}
    best_path: Path | None = None
    best_rows: list[dict[str, Any]] = []
    for path in result_candidates_for_stage(split_dir, stage):
        rows = load_jsonl(path)
        candidate_counts[path.name] = len(rows)
        if len(rows) > len(best_rows):
            best_path = path
            best_rows = rows
    return best_path, best_rows, candidate_counts


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Stable de-duplication by task_id, keeping the last row encountered."""
    by_task: dict[str, dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    duplicates = 0
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            passthrough.append(row)
            continue
        if task_id in by_task:
            duplicates += 1
        by_task[task_id] = row
    return passthrough + list(by_task.values()), duplicates


def trajectory_path_for_stage(split_dir: Path, stage: str) -> Path | None:
    candidates = {
        "p1_v3r4": ("p1_trajectories.jsonl",),
        "p2_v3r1": ("main_trajectories.jsonl",),
        "p3_v3r1": ("p3_trajectories.jsonl",),
        "p4_full_v2": ("p4_traces.jsonl",),
    }[stage]
    for name in candidates:
        path = split_dir / name
        if path.exists():
            return path
    return None


def load_trajectory_dicts(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    trajectories: dict[str, dict[str, Any]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            trajectory = json.loads(line)
            task_id = trajectory.get("task_id")
            if isinstance(task_id, str):
                trajectories[task_id] = trajectory
    return trajectories


def iter_propose_steps(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in trajectory.get("steps", [])
        if step.get("action_type") == "propose_design" and isinstance(step.get("proposal"), dict)
    ]


def step_verifier(step: dict[str, Any]) -> dict[str, Any]:
    response = step.get("verifier_response")
    return response if isinstance(response, dict) else {}


def step_feasible(step: dict[str, Any]) -> bool | None:
    response = step_verifier(step)
    if "is_feasible" in response:
        return bool(response["is_feasible"])
    if "feasible" in response:
        return bool(response["feasible"])
    return None


def step_objective(step: dict[str, Any]) -> float | None:
    response = step_verifier(step)
    value = response.get("objective_value")
    return float(value) if isinstance(value, (int, float)) else None


def step_slack(step: dict[str, Any]) -> dict[str, float]:
    raw = step.get("constraint_slack")
    if not isinstance(raw, dict):
        raw = step_verifier(step).get("constraint_slack")
    if not isinstance(raw, dict):
        return {}
    return {key: float(value) for key, value in raw.items() if isinstance(value, (int, float))}


def violation_total(slack: dict[str, float]) -> float:
    return sum(max(0.0, -value) for value in slack.values())


def dominant_violation_name(slack: dict[str, float]) -> str | None:
    violated = [(name, value) for name, value in slack.items() if value < 0]
    if not violated:
        return None
    return min(violated, key=lambda item: item[1])[0]


def compute_utility(step: dict[str, Any], p_ref: float | None) -> float | None:
    if p_ref is None or p_ref <= 0:
        return None
    power = step_objective(step)
    if power is None:
        power = 0.0
    slack = step_slack(step)
    v_freq = max(0.0, -float(slack.get("freq_error_pct_limit", 0.0)))
    v_stress = max(0.0, -float(slack.get("stress_limit_mpa", 0.0)))
    v_disp = max(0.0, -float(slack.get("disp_limit_mm", 0.0)))
    return (float(power) / float(p_ref)) - 0.5 * (v_freq + v_stress + v_disp)


def normalized_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    task: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    if not task:
        return None, None
    bounds = task.get("variable_bounds")
    if not isinstance(bounds, dict):
        return None, None
    deltas: list[float] = []
    changed = 0
    usable = 0
    for name, raw_bound in bounds.items():
        if name not in before or name not in after:
            continue
        if not isinstance(raw_bound, dict):
            continue
        lo = raw_bound.get("min")
        hi = raw_bound.get("max")
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            continue
        span = float(hi) - float(lo)
        if span <= 0:
            continue
        try:
            delta = abs(float(after[name]) - float(before[name])) / span
        except (TypeError, ValueError):
            continue
        usable += 1
        if delta > 1e-12:
            changed += 1
        deltas.append(delta)
    if not deltas:
        return None, None
    return statistics.mean(deltas), changed / usable if usable else None


def p2_row_ratio(row: dict[str, Any]) -> float:
    if not bool(row.get("is_feasible")):
        return 0.0
    obj = row.get("objective_value")
    bkf = row.get("bkf_objective_value")
    if not isinstance(obj, (int, float)) or not isinstance(bkf, (int, float)) or bkf <= 0:
        return 0.0
    return float(obj) / float(bkf)


def compute_best_so_far_auc(row: dict[str, Any], budget: int = 6) -> float | None:
    history = row.get("objective_history")
    bkf = row.get("bkf_objective_value")
    if not isinstance(history, list) or not isinstance(bkf, (int, float)) or bkf <= 0:
        return None
    padded = list(history) + [None] * max(0, budget - len(history))
    best_so_far = 0.0
    total = 0.0
    for value in padded[:budget]:
        if isinstance(value, (int, float)):
            best_so_far = max(best_so_far, float(value))
        total += best_so_far / float(bkf)
    return total / budget if budget else None


def load_task_maps() -> dict[str, dict[str, dict[str, Any]]]:
    maps: dict[str, dict[str, dict[str, Any]]] = {"p2_v3r1": {}, "p3_v3r1": {}, "p4_full_v2": {}}
    for split in SPLITS:
        for row in maybe_load_jsonl(P2_TASK_DIR / f"p2_{split}_tasks.jsonl"):
            maps["p2_v3r1"][row["task_id"]] = row
        for row in maybe_load_jsonl(P3_TASK_DIR / f"p3_{split}_tasks.jsonl"):
            maps["p3_v3r1"][row["task_id"]] = row
        for row in maybe_load_jsonl(P4_TASK_DIR / f"p4_{split}_tasks.jsonl"):
            maps["p4_full_v2"][row["task_id"]] = row
    return maps


def discover_core_models() -> dict[str, ModelArtifacts]:
    artifacts: dict[str, ModelArtifacts] = {}
    for stage, stage_root in CORE_STAGE_ROOTS.items():
        if not stage_root.exists():
            continue
        for model_dir in sorted(path for path in stage_root.iterdir() if path.is_dir()):
            model_key = CORE_MODEL_ALIASES.get((stage, model_dir.name), model_dir.name)
            item = artifacts.setdefault(
                model_key,
                ModelArtifacts(
                    model_key=model_key,
                    display_name=DISPLAY_NAMES.get(model_key, model_key),
                    source_group="core_main_table_bundle_20260417",
                ),
            )
            for split in SPLITS:
                split_dir = model_dir / split
                if split_dir.exists():
                    item.add_split_dir(stage, split, split_dir)
    for (model_key, stage), stage_root in CORE_STAGE_MODEL_OVERRIDES.items():
        if not stage_root.exists():
            continue
        item = artifacts.setdefault(
            model_key,
            ModelArtifacts(
                model_key=model_key,
                display_name=DISPLAY_NAMES.get(model_key, model_key),
                source_group="core_main_table_bundle_20260417_with_overrides",
            ),
        )
        if item.source_group == "core_main_table_bundle_20260417":
            item.source_group = "core_main_table_bundle_20260417_with_overrides"
        for split in SPLITS:
            split_dir = stage_root / split
            if split_dir.exists():
                item.add_split_dir(stage, split, split_dir)
    return artifacts


def discover_extension_models() -> dict[str, ModelArtifacts]:
    artifacts: dict[str, ModelArtifacts] = {}
    for model_key, model_root in EXTENSION_ROOTS.items():
        if not model_root.exists():
            continue
        item = ModelArtifacts(
            model_key=model_key,
            display_name=DISPLAY_NAMES.get(model_key, model_key),
            source_group="thinking_extension_runs_20260425_20260427",
        )
        for stage in STAGES:
            for split in SPLITS:
                split_dir = model_root / stage / split
                if split_dir.exists():
                    item.add_split_dir(stage, split, split_dir)
        artifacts[model_key] = item
    return artifacts


def read_stage_rows(model_artifacts: ModelArtifacts, stage: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    all_rows: list[dict[str, Any]] = []
    per_split: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    for split, split_dir in sorted(model_artifacts.stage_dirs.get(stage, {}).items()):
        result_path, raw_rows, candidate_counts = load_best_result_rows(split_dir, stage)
        if result_path is None:
            provenance[split] = {"missing_result": str(split_dir)}
            continue
        repair_sources: list[dict[str, Any]] = []
        for repair_dir in ROW_REPAIR_DIRS.get((model_artifacts.model_key, stage, split), ()):
            repair_path, repair_rows, repair_candidate_counts = load_best_result_rows(repair_dir, stage)
            if repair_path is None:
                repair_sources.append({"missing_repair_result": str(repair_dir)})
                continue
            raw_rows.extend(repair_rows)
            repair_sources.append(
                {
                    "result_path": str(repair_path.relative_to(ROOT)),
                    "candidate_result_counts": repair_candidate_counts,
                    "n_rows": len(repair_rows),
                }
            )
        rows, duplicate_count = dedupe_rows(raw_rows)
        per_split[split] = rows
        all_rows.extend(rows)
        provenance[split] = {
            "result_path": str(result_path.relative_to(ROOT)),
            "candidate_result_counts": candidate_counts,
            "raw_n_rows": len(raw_rows),
            "n_rows": len(rows),
            "deduped_task_rows": duplicate_count,
        }
        if repair_sources:
            provenance[split]["repair_overrides"] = repair_sources
        trajectory_path = trajectory_path_for_stage(split_dir, stage)
        if trajectory_path is not None:
            provenance[split]["trajectory_path"] = str(trajectory_path.relative_to(ROOT))
    all_rows, all_duplicate_count = dedupe_rows(all_rows)
    if all_duplicate_count:
        provenance["_all_splits"] = {"deduped_task_rows": all_duplicate_count}
    return all_rows, provenance, per_split


def read_stage_trajectories(model_artifacts: ModelArtifacts, stage: str) -> dict[str, dict[str, Any]]:
    all_trajectories: dict[str, dict[str, Any]] = {}
    for split, split_dir in model_artifacts.stage_dirs.get(stage, {}).items():
        all_trajectories.update(load_trajectory_dicts(trajectory_path_for_stage(split_dir, stage)))
        for repair_dir in ROW_REPAIR_DIRS.get((model_artifacts.model_key, stage, split), ()):
            all_trajectories.update(load_trajectory_dicts(trajectory_path_for_stage(repair_dir, stage)))
    return all_trajectories


def compute_p1_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available": False, "n": 0}
    summary = P1Evaluator().aggregate(rows)
    n = len(rows)
    pred_counts = {action: 0 for action in ACTIONS}
    gold_counts = {action: 0 for action in ACTIONS}
    invalid_count = 0
    recalls: dict[str, float | None] = {}
    precisions: dict[str, float | None] = {}
    spurious: dict[str, float | None] = {}
    for row in rows:
        pred = action_for_row(row)
        gold = gold_action_for_row(row)
        if pred in pred_counts:
            pred_counts[pred] += 1
        else:
            invalid_count += 1
        if gold in gold_counts:
            gold_counts[gold] += 1
    for action in ACTIONS:
        gold_action_rows = [row for row in rows if gold_action_for_row(row) == action]
        pred_action_rows = [row for row in rows if action_for_row(row) == action]
        non_action_rows = [row for row in rows if gold_action_for_row(row) != action]
        recalls[action] = safe_div(sum(1 for row in gold_action_rows if action_for_row(row) == action), len(gold_action_rows))
        precisions[action] = safe_div(sum(1 for row in pred_action_rows if gold_action_for_row(row) == action), len(pred_action_rows))
        spurious[action] = safe_div(sum(1 for row in non_action_rows if action_for_row(row) == action), len(non_action_rows))
    pred_dist = {action: pred_counts[action] / n for action in ACTIONS}
    gold_dist = {action: gold_counts[action] / n for action in ACTIONS}
    acs = (recalls["propose_design"] or 0.0) * (1.0 - (spurious["propose_design"] or 0.0))
    mds = (recalls["request_missing_info"] or 0.0) * (1.0 - (spurious["request_missing_info"] or 0.0))
    ids = (recalls["declare_infeasible"] or 0.0) * (1.0 - (spurious["declare_infeasible"] or 0.0))
    composite = (
        0.40 * summary.p1_3class_macro_f1
        + 0.20 * acs
        + 0.15 * mds
        + 0.15 * ids
        + 0.10 * summary.p1_6subtype_macro_f1
    )
    return {
        "available": True,
        "n": n,
        "accuracy": safe_div(sum(1 for row in rows if bool(row.get("is_correct"))), n),
        "macro_f1": summary.p1_3class_macro_f1,
        "subtype_f1": summary.p1_6subtype_macro_f1,
        "composite": composite,
        "acs": acs,
        "mds": mds,
        "ids": ids,
        "action_entropy": normalized_entropy(pred_counts),
        "action_distribution_l1": sum(abs(pred_dist[action] - gold_dist[action]) for action in ACTIONS),
        "action_distribution_alignment": distribution_alignment(pred_dist, gold_dist),
        "predicted_action_rate": pred_dist,
        "gold_action_rate": gold_dist,
        "propose_rate": pred_dist["propose_design"],
        "request_rate": pred_dist["request_missing_info"],
        "infeasible_rate": pred_dist["declare_infeasible"],
        "invalid_action_rate": invalid_count / n,
        "non_invalid_rate": 1.0 - invalid_count / n,
        "spurious_propose_rate": spurious["propose_design"],
        "spurious_request_rate": spurious["request_missing_info"],
        "spurious_infeasible_rate": spurious["declare_infeasible"],
        "propose_recall": recalls["propose_design"],
        "missing_recall": recalls["request_missing_info"],
        "infeasible_recall": recalls["declare_infeasible"],
        "propose_precision": precisions["propose_design"],
        "missing_precision": precisions["request_missing_info"],
        "infeasible_precision": precisions["declare_infeasible"],
    }


def compute_violation_reduction_consistency(trajectories: dict[str, dict[str, Any]]) -> float | None:
    values: list[float] = []
    for trajectory in trajectories.values():
        steps = iter_propose_steps(trajectory)
        consistent = 0
        total = 0
        for current, nxt in zip(steps, steps[1:]):
            current_slack = step_slack(current)
            next_slack = step_slack(nxt)
            dominant = dominant_violation_name(current_slack)
            if dominant is None or dominant not in next_slack:
                continue
            total += 1
            if next_slack[dominant] > current_slack[dominant]:
                consistent += 1
        if total:
            values.append(consistent / total)
    return safe_mean(values)


def compute_trajectory_improvement_rate(rows_by_task: dict[str, dict[str, Any]], trajectories: dict[str, dict[str, Any]]) -> float | None:
    values: list[float] = []
    for task_id, trajectory in trajectories.items():
        row = rows_by_task.get(task_id)
        if row is None:
            continue
        bkf = row.get("bkf_objective_value")
        p_ref = float(bkf) if isinstance(bkf, (int, float)) and bkf > 0 else None
        steps = iter_propose_steps(trajectory)
        utilities = [compute_utility(step, p_ref) for step in steps]
        utilities = [value for value in utilities if value is not None]
        if len(utilities) < 2:
            continue
        values.append(sum(1 for prev, cur in zip(utilities, utilities[1:]) if cur > prev) / (len(utilities) - 1))
    return safe_mean(values)


def compute_p2_metrics(
    rows: list[dict[str, Any]],
    trajectories: dict[str, dict[str, Any]],
    task_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {"available": False, "n": 0}
    rows_by_task = {row["task_id"]: row for row in rows if isinstance(row.get("task_id"), str)}
    ratios = [p2_row_ratio(row) for row in rows]
    feasible_ratios = [ratio for ratio in ratios if ratio > 0]
    protocol_invalid = sum(1 for row in rows if row.get("final_action_type") == "invalid_output")
    auc_values = [compute_best_so_far_auc(row) for row in rows]
    deltas: list[float] = []
    vars_changed: list[float] = []
    bounded_local_edits = 0
    nonzero_edits = 0
    feasible_to_feasible = 0
    feasible_transitions = 0
    feasible_to_infeasible = 0
    infeasible_to_feasible = 0
    infeasible_transitions = 0
    first_to_final_deltas: list[float] = []
    post_feedback_feasible: list[float] = []
    for task_id, trajectory in trajectories.items():
        task = task_map.get(task_id)
        propose_steps = iter_propose_steps(trajectory)
        if len(propose_steps) >= 2:
            first = propose_steps[0].get("proposal")
            final = propose_steps[-1].get("proposal")
            if isinstance(first, dict) and isinstance(final, dict):
                first_final_delta, _ = normalized_delta(first, final, task)
                if first_final_delta is not None:
                    first_to_final_deltas.append(first_final_delta)
        for before_step, after_step in zip(propose_steps, propose_steps[1:]):
            before = before_step.get("proposal")
            after = after_step.get("proposal")
            if isinstance(before, dict) and isinstance(after, dict):
                delta, var_frac = normalized_delta(before, after, task)
                if delta is not None:
                    deltas.append(delta)
                    if delta > 1e-12:
                        nonzero_edits += 1
                        if delta <= 0.25:
                            bounded_local_edits += 1
                if var_frac is not None:
                    vars_changed.append(var_frac)
            before_feasible = step_feasible(before_step)
            after_feasible = step_feasible(after_step)
            if before_feasible is True:
                feasible_transitions += 1
                if after_feasible is True:
                    feasible_to_feasible += 1
                elif after_feasible is False:
                    feasible_to_infeasible += 1
            if before_feasible is False:
                infeasible_transitions += 1
                if after_feasible is True:
                    infeasible_to_feasible += 1
            if after_feasible is not None:
                post_feedback_feasible.append(1.0 if after_feasible else 0.0)
    destructive_edit_rate = safe_div(feasible_to_infeasible, feasible_transitions)
    return {
        "available": True,
        "n": len(rows),
        "n_trajectories": len(trajectories),
        "p2a_first_feasible": safe_div(sum(1 for row in rows if row.get("first_proposal_is_feasible") is True), len(rows)),
        "final_feasible_rate": safe_div(sum(1 for row in rows if bool(row.get("is_feasible"))), len(rows)),
        "final_feasible_power_ratio": safe_mean(ratios),
        "conditional_feasible_power_ratio": safe_mean(feasible_ratios),
        "mean_best_so_far_auc": safe_mean(auc_values),
        "mean_queries": safe_mean([float(row["queries_used"]) for row in rows if isinstance(row.get("queries_used"), (int, float))]),
        "protocol_invalid_rate": protocol_invalid / len(rows),
        "protocol_valid_rate": 1.0 - protocol_invalid / len(rows),
        "mean_normalized_step_delta": safe_mean(deltas),
        "median_normalized_step_delta": safe_median(deltas),
        "mean_first_to_final_delta": safe_mean(first_to_final_deltas),
        "bounded_local_edit_rate": safe_div(bounded_local_edits, nonzero_edits),
        "mean_variable_change_fraction": safe_mean(vars_changed),
        "feasibility_preservation_rate": safe_div(feasible_to_feasible, feasible_transitions),
        "destructive_edit_rate": destructive_edit_rate,
        "non_destructive_edit_rate": None if destructive_edit_rate is None else 1.0 - destructive_edit_rate,
        "directed_repair_rate": safe_div(infeasible_to_feasible, infeasible_transitions),
        "post_feedback_feasible_rate": safe_mean(post_feedback_feasible),
        "violation_reduction_consistency": compute_violation_reduction_consistency(trajectories),
        "utility_improvement_rate": compute_trajectory_improvement_rate(rows_by_task, trajectories),
    }


def p3_task_results_from_artifacts(
    model_artifacts: ModelArtifacts,
    rows: list[dict[str, Any]],
    task_map: dict[str, dict[str, Any]],
) -> list[Any]:
    evaluator = P3Evaluator()
    all_task_results: list[Any] = []
    for split, split_dir in model_artifacts.stage_dirs.get("p3_v3r1", {}).items():
        trajectory_paths = []
        trajectory_path = trajectory_path_for_stage(split_dir, "p3_v3r1")
        if trajectory_path is not None:
            trajectory_paths.append(trajectory_path)
        for repair_dir in ROW_REPAIR_DIRS.get((model_artifacts.model_key, "p3_v3r1", split), ()):
            repair_trajectory_path = trajectory_path_for_stage(repair_dir, "p3_v3r1")
            if repair_trajectory_path is not None:
                trajectory_paths.append(repair_trajectory_path)
        if not trajectory_paths:
            continue
        try:
            trajectories: dict[str, Any] = {}
            for path in trajectory_paths:
                trajectories.update(evaluator.load_trajectories(path))
        except Exception as exc:  # pragma: no cover - protects artifact audits.
            print(f"[warn] failed to load P3 trajectories for {model_artifacts.model_key}/{split}: {exc}", file=sys.stderr)
            continue
        split_task_ids = {row.get("task_id") for row in rows if row.get("split") == split}
        if not split_task_ids:
            split_task_ids = set(trajectories)
        runner_name = rows[0].get("runner_name", model_artifacts.display_name) if rows else model_artifacts.display_name
        for task_id in sorted(split_task_ids):
            if not isinstance(task_id, str) or task_id not in task_map or task_id not in trajectories:
                continue
            try:
                all_task_results.append(
                    evaluator.evaluate_task(
                        task=task_map[task_id],
                        trajectory=trajectories[task_id],
                        runner_name=runner_name,
                    )
                )
            except Exception as exc:  # pragma: no cover - keeps one malformed task from aborting.
                print(f"[warn] failed to evaluate P3 task {task_id}: {exc}", file=sys.stderr)
    return all_task_results


def load_intervention_effects() -> dict[str, dict[str, float]]:
    if not P3_INTERVENTION_SUMMARY.exists():
        return {}
    data = json.loads(P3_INTERVENTION_SUMMARY.read_text())
    reverse_display = {display: key for key, display in DISPLAY_NAMES.items()}
    effects: dict[str, dict[str, float]] = {}
    for display_name, effect_bundle in data.get("effects", {}).items():
        model_key = reverse_display.get(display_name)
        if model_key is None and display_name == "model_B":
            model_key = "model_B"
        if model_key is None:
            continue
        summary_effect = effect_bundle.get("summary_effect", {})
        effects[model_key] = {
            "summary_success_delta": summary_effect.get("p3_success_delta"),
            "summary_cascade_delta": summary_effect.get("constraint_cascade_rate_delta"),
            "summary_cascade_reduction": (
                -float(summary_effect["constraint_cascade_rate_delta"])
                if isinstance(summary_effect.get("constraint_cascade_rate_delta"), (int, float))
                else None
            ),
        }
    return effects


def compute_p3_metrics(
    model_artifacts: ModelArtifacts,
    rows: list[dict[str, Any]],
    trajectories: dict[str, dict[str, Any]],
    task_map: dict[str, dict[str, Any]],
    intervention_effects: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if not rows:
        return {"available": False, "n": 0}
    task_results = p3_task_results_from_artifacts(model_artifacts, rows, task_map)
    summary = P3Evaluator().aggregate(task_results) if task_results else None
    protocol_invalid = sum(1 for row in rows if row.get("final_action_type") == "invalid_output")
    effects = intervention_effects.get(model_artifacts.model_key, {})
    return {
        "available": True,
        "n": len(rows),
        "n_trajectories": len(trajectories),
        "n_escape_evaluated": len(task_results),
        "success_rate": safe_div(sum(1 for row in rows if bool(row.get("is_feasible"))), len(rows)),
        "first_recovery_feasible_rate": safe_div(sum(1 for row in rows if bool(row.get("first_proposal_is_feasible"))), len(rows)),
        "mean_queries": safe_mean([float(row["queries_used"]) for row in rows if isinstance(row.get("queries_used"), (int, float))]),
        "protocol_invalid_rate": protocol_invalid / len(rows),
        "protocol_valid_rate": 1.0 - protocol_invalid / len(rows),
        "trap_escape_rate": summary.trap_escape_rate if summary else None,
        "explicit_replan_rate": summary.explicit_replan_rate if summary else None,
        "dead_budget_rate": summary.dead_budget_rate if summary else None,
        "non_dead_budget_rate": None if summary is None else 1.0 - summary.dead_budget_rate,
        "escape_quality": summary.escape_quality if summary else None,
        "constraint_cascade_rate": summary.constraint_cascade_rate if summary else None,
        "non_cascade_rate": None if summary is None or summary.constraint_cascade_rate is None else 1.0 - summary.constraint_cascade_rate,
        "escape_time": summary.escape_time if summary else None,
        "post_feedback_feasible_rate": safe_mean(
            [
                1.0 if step_feasible(step) else 0.0
                for trajectory in trajectories.values()
                for step in iter_propose_steps(trajectory)[1:]
                if step_feasible(step) is not None
            ]
        ),
        "violation_reduction_consistency": compute_violation_reduction_consistency(trajectories),
        "summary_success_delta": effects.get("summary_success_delta"),
        "summary_cascade_delta": effects.get("summary_cascade_delta"),
        "summary_cascade_reduction": effects.get("summary_cascade_reduction"),
    }


def compute_p4_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available": False, "n": 0}
    summary = P4Evaluator().aggregate(rows)
    full_tau = summary.ranking_kendall_tau
    active_rows = P4Evaluator._balanced_active_rows(rows)
    active_policy = safe_mean(
        [
            float(row["policy_sensitive_pair_accuracy"])
            for row in active_rows
            if isinstance(row.get("policy_sensitive_pair_accuracy"), (int, float))
        ]
    )
    all_policy = safe_mean(
        [
            float(row["policy_sensitive_pair_accuracy"])
            for row in rows
            if isinstance(row.get("policy_sensitive_pair_accuracy"), (int, float))
        ]
    )
    return {
        "available": True,
        "n": len(rows),
        "full_tau": full_tau,
        "full_tau_scaled": None if full_tau is None else (full_tau + 1.0) / 2.0,
        "pareto_tau": safe_mean([float(row["pareto_kendall_tau"]) for row in rows if isinstance(row.get("pareto_kendall_tau"), (int, float))]),
        "balanced_active_n": summary.balanced_active_n_tasks,
        "balanced_active_bars": summary.balanced_active_bars,
        "balanced_active_tau": summary.balanced_active_ranking_kendall_tau,
        "balanced_active_policy_sensitive_pair_accuracy": summary.balanced_active_policy_sensitive_pair_accuracy or active_policy,
        "all_policy_sensitive_pair_accuracy": summary.policy_sensitive_pair_accuracy or all_policy,
        "exact_match_rate": summary.exact_match_rate,
        "top1_accuracy": summary.top1_accuracy,
        "top2_set_accuracy": summary.top2_set_accuracy,
        "pareto_violation_rate": summary.dominance_violation_rate,
        "non_pareto_violation_rate": None if summary.dominance_violation_rate is None else 1.0 - summary.dominance_violation_rate,
        "parse_error_rate": summary.parse_error_rate,
        "non_parse_error_rate": 1.0 - summary.parse_error_rate,
        "full_tau_exact_gap": (
            None
            if full_tau is None or summary.exact_match_rate is None
            else ((full_tau + 1.0) / 2.0) - summary.exact_match_rate
        ),
        "bars_exact_gap": (
            None
            if summary.balanced_active_bars is None or summary.exact_match_rate is None
            else summary.balanced_active_bars - summary.exact_match_rate
        ),
    }


def get_nested(metrics: dict[str, Any], dotted_key: str) -> float | None:
    current: Any = metrics
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, (int, float)) and math.isfinite(float(current)):
        return float(current)
    return None


def dimension_score(stage_metrics: dict[str, Any], metric_keys: tuple[str, ...]) -> tuple[float | None, int, int]:
    values = [clamp01(get_nested(stage_metrics, key)) for key in metric_keys]
    clean = [value for value in values if value is not None]
    return safe_mean(clean), len(clean), len(metric_keys)


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        for original_idx, _ in indexed[index:end]:
            ranks[original_idx] = avg_rank
        index = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0 or y_var <= 0:
        return None
    return cov / math.sqrt(x_var * y_var)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(rank_values(xs), rank_values(ys))


def compute_validity(model_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    correlations: dict[str, dict[str, Any]] = {}
    for dim in DIMENSION_METRICS:
        correlations[dim] = {}
        for target_name, target_key in STAGE_TARGETS.items():
            xs: list[float] = []
            ys: list[float] = []
            models: list[str] = []
            for model_key, payload in model_outputs.items():
                score = payload.get("dimension_scores", {}).get(dim, {}).get("score")
                target = get_nested(payload.get("stage_metrics", {}), target_key)
                if score is None or target is None:
                    continue
                xs.append(float(score))
                ys.append(float(target))
                models.append(model_key)
            correlations[dim][target_name] = {
                "n_models": len(xs),
                "spearman_r": spearman(xs, ys),
                "pearson_r": pearson(xs, ys),
                "models": models,
            }
    return {"correlations": round_float(correlations)}


def complete_p1_p4_models(model_outputs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    complete: dict[str, dict[str, Any]] = {}
    for model_key, payload in model_outputs.items():
        stage_metrics = payload.get("stage_metrics", {})
        is_complete = True
        for stage_key, expected in EXPECTED_STAGE_ROWS.items():
            stage = stage_metrics.get(stage_key, {})
            if stage.get("n") != expected:
                is_complete = False
                break
        if is_complete:
            complete[model_key] = payload
    return complete


def build_extraction_audit(model_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    audit_models: dict[str, Any] = {}
    for model_key, payload in model_outputs.items():
        stage_metrics = payload["stage_metrics"]
        stage_audit: dict[str, Any] = {}
        for short_stage, expected in EXPECTED_STAGE_ROWS.items():
            metrics = stage_metrics.get(short_stage, {})
            n_rows = metrics.get("n", 0)
            stage_audit[short_stage] = {
                "n_rows": n_rows,
                "expected_rows": expected,
                "complete": n_rows == expected,
                "available": bool(metrics.get("available")),
            }
        deduped_rows = 0
        for stage_provenance in payload.get("provenance", {}).values():
            if not isinstance(stage_provenance, dict):
                continue
            for split_provenance in stage_provenance.values():
                if isinstance(split_provenance, dict):
                    deduped_rows += int(split_provenance.get("deduped_task_rows", 0) or 0)
        audit_models[model_key] = {
            "display_name": payload["display_name"],
            "source_group": payload["source_group"],
            "stages": stage_audit,
            "deduped_task_rows": deduped_rows,
        }
    incomplete = {
        model_key: {
            stage: info
            for stage, info in payload["stages"].items()
            if not info["complete"] and info["n_rows"] != 0
        }
        for model_key, payload in audit_models.items()
    }
    missing = {
        model_key: {
            stage: info
            for stage, info in payload["stages"].items()
            if info["n_rows"] == 0
        }
        for model_key, payload in audit_models.items()
    }
    return {
        "expected_stage_rows": EXPECTED_STAGE_ROWS,
        "models": audit_models,
        "models_with_partial_nonzero_stages": {key: value for key, value in incomplete.items() if value},
        "models_with_missing_stages": {key: value for key, value in missing.items() if value},
    }


def flatten_metrics(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_metrics(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, (int, float, str)) or value is None or isinstance(value, bool):
        out[prefix] = value


def csv_safe(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return value


def write_profile_scores_csv(path: Path, model_outputs: dict[str, dict[str, Any]]) -> None:
    fields = [
        "model_key",
        "display_name",
        "source_group",
        "p1_n",
        "p2_n",
        "p3_n",
        "p4_n",
        "action_prior",
        "edit_style",
        "feedback_obedience",
        "state_trust",
        "preference_execution",
        "p1_composite",
        "p2_final_feasible_power_ratio",
        "p3_success_rate",
        "p3_trap_escape_rate",
        "p4_balanced_active_bars",
        "p4_full_tau",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for model_key in sorted(model_outputs):
            payload = model_outputs[model_key]
            stage = payload["stage_metrics"]
            dims = payload["dimension_scores"]
            row = {
                "model_key": model_key,
                "display_name": payload["display_name"],
                "source_group": payload["source_group"],
                "p1_n": get_nested(stage, "p1.n"),
                "p2_n": get_nested(stage, "p2.n"),
                "p3_n": get_nested(stage, "p3.n"),
                "p4_n": get_nested(stage, "p4.n"),
                "action_prior": dims.get("action_prior", {}).get("score"),
                "edit_style": dims.get("edit_style", {}).get("score"),
                "feedback_obedience": dims.get("feedback_obedience", {}).get("score"),
                "state_trust": dims.get("state_trust", {}).get("score"),
                "preference_execution": dims.get("preference_execution", {}).get("score"),
                "p1_composite": get_nested(stage, "p1.composite"),
                "p2_final_feasible_power_ratio": get_nested(stage, "p2.final_feasible_power_ratio"),
                "p3_success_rate": get_nested(stage, "p3.success_rate"),
                "p3_trap_escape_rate": get_nested(stage, "p3.trap_escape_rate"),
                "p4_balanced_active_bars": get_nested(stage, "p4.balanced_active_bars"),
                "p4_full_tau": get_nested(stage, "p4.full_tau"),
            }
            writer.writerow({key: csv_safe(value) for key, value in row.items()})


def write_flat_metrics_csv(path: Path, model_outputs: dict[str, dict[str, Any]]) -> None:
    flat_rows: list[dict[str, Any]] = []
    fieldnames = {"model_key", "display_name", "source_group"}
    for model_key, payload in model_outputs.items():
        flat: dict[str, Any] = {
            "model_key": model_key,
            "display_name": payload["display_name"],
            "source_group": payload["source_group"],
        }
        flatten_metrics("stage_metrics", payload["stage_metrics"], flat)
        flatten_metrics("dimension_scores", payload["dimension_scores"], flat)
        flat_rows.append(flat)
        fieldnames.update(flat)
    ordered = ["model_key", "display_name", "source_group"] + sorted(fieldnames - {"model_key", "display_name", "source_group"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered)
        writer.writeheader()
        for row in sorted(flat_rows, key=lambda item: item["model_key"]):
            writer.writerow({key: csv_safe(row.get(key)) for key in ordered})


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def latex_num(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "--"


def write_latex_rows(path: Path, model_outputs: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(
        model_outputs,
        key=lambda key: (
            model_outputs[key].get("dimension_scores", {}).get("action_prior", {}).get("score") is None,
            -(model_outputs[key].get("dimension_scores", {}).get("action_prior", {}).get("score") or -1.0),
            model_outputs[key]["display_name"],
        ),
    )
    lines = [
        "% Auto-generated by scripts/quantify_response_control_profiles.py",
        "% Columns: Model, Action prior, Edit style, Feedback obedience, State trust, Preference execution",
    ]
    for model_key in ordered:
        payload = model_outputs[model_key]
        dims = payload["dimension_scores"]
        lines.append(
            " & ".join(
                [
                    latex_escape(payload["display_name"]),
                    latex_num(dims.get("action_prior", {}).get("score")),
                    latex_num(dims.get("edit_style", {}).get("score")),
                    latex_num(dims.get("feedback_obedience", {}).get("score")),
                    latex_num(dims.get("state_trust", {}).get("score")),
                    latex_num(dims.get("preference_execution", {}).get("score")),
                ]
            )
            + r" \\"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def compute_model_profiles(include_extensions: bool) -> dict[str, dict[str, Any]]:
    task_maps = load_task_maps()
    intervention_effects = load_intervention_effects()
    discovered = discover_core_models()
    if include_extensions:
        discovered.update(discover_extension_models())
    outputs: dict[str, dict[str, Any]] = {}
    for model_key, artifact in sorted(discovered.items()):
        stage_metrics: dict[str, Any] = {}
        provenance: dict[str, Any] = {}
        p1_rows, p1_prov, _ = read_stage_rows(artifact, "p1_v3r4")
        p2_rows, p2_prov, _ = read_stage_rows(artifact, "p2_v3r1")
        p3_rows, p3_prov, _ = read_stage_rows(artifact, "p3_v3r1")
        p4_rows, p4_prov, _ = read_stage_rows(artifact, "p4_full_v2")
        p2_trajectories = read_stage_trajectories(artifact, "p2_v3r1")
        p3_trajectories = read_stage_trajectories(artifact, "p3_v3r1")
        stage_metrics["p1"] = compute_p1_metrics(p1_rows)
        stage_metrics["p2"] = compute_p2_metrics(p2_rows, p2_trajectories, task_maps["p2_v3r1"])
        stage_metrics["p3"] = compute_p3_metrics(
            artifact,
            p3_rows,
            p3_trajectories,
            task_maps["p3_v3r1"],
            intervention_effects,
        )
        stage_metrics["p4"] = compute_p4_metrics(p4_rows)
        provenance["p1_v3r4"] = p1_prov
        provenance["p2_v3r1"] = p2_prov
        provenance["p3_v3r1"] = p3_prov
        provenance["p4_full_v2"] = p4_prov
        dimension_scores: dict[str, Any] = {}
        for dimension, metric_keys in DIMENSION_METRICS.items():
            score, available, total = dimension_score(stage_metrics, metric_keys)
            dimension_scores[dimension] = {
                "score": score,
                "available_metrics": available,
                "total_metrics": total,
                "coverage": available / total if total else None,
                "metrics": list(metric_keys),
            }
        outputs[model_key] = {
            "model_key": model_key,
            "display_name": artifact.display_name,
            "source_group": artifact.source_group,
            "stage_metrics": stage_metrics,
            "dimension_scores": dimension_scores,
            "provenance": provenance,
        }
    return round_float(outputs)


def build_metric_definitions() -> dict[str, Any]:
    return {
        "score_range": "All dimension scores are simple means of available oriented indicators in [0, 1].",
        "missing_policy": "Missing indicators are reported as null and excluded from the dimension mean; coverage records how many were available.",
        "bounded_local_edit_rate": "Fraction of non-zero consecutive proposal edits with mean normalized variable delta <= 0.25.",
        "directed_repair_rate": "P(next proposal feasible | current proposal infeasible) over consecutive propose_design steps.",
        "feasibility_preservation_rate": "P(next proposal feasible | current proposal feasible) over consecutive propose_design steps.",
        "violation_reduction_consistency": "Fraction of consecutive proposal pairs that improve the currently dominant violated slack.",
        "summary_success_delta": "Optional P3 intervention effect: state_summary_full_numeric success minus raw_history_full_numeric success.",
        "summary_cascade_reduction": "Optional P3 intervention effect: raw-history cascade rate minus state-summary cascade rate.",
        "dimension_metrics": DIMENSION_METRICS,
        "stage_targets": STAGE_TARGETS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "analysis" / "response_control_profiles",
        help="Directory for JSON/CSV/TeX outputs.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Only read the curated main-table bundle; skip DeepSeek-V4/Mimo/Hy3 extension runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    model_outputs = compute_model_profiles(include_extensions=not args.core_only)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "include_extensions": not args.core_only,
        "metric_definitions": build_metric_definitions(),
        "models": model_outputs,
    }
    validity = compute_validity(model_outputs)
    complete_outputs = complete_p1_p4_models(model_outputs)
    complete_validity = compute_validity(complete_outputs)
    extraction_audit = build_extraction_audit(model_outputs)
    payload["validity"] = validity
    payload["complete_p1_p4_model_keys"] = sorted(complete_outputs)
    payload["complete_p1_p4_validity"] = complete_validity
    payload["extraction_audit"] = extraction_audit
    write_json(output_dir / "profile_metrics.json", payload)
    write_json(output_dir / "profile_validity.json", validity)
    write_json(output_dir / "profile_validity_complete_p1_p4.json", complete_validity)
    write_json(output_dir / "profile_extraction_audit.json", extraction_audit)
    write_profile_scores_csv(output_dir / "profile_scores.csv", model_outputs)
    write_profile_scores_csv(output_dir / "profile_scores_complete_p1_p4.csv", complete_outputs)
    write_flat_metrics_csv(output_dir / "profile_metrics_flat.csv", model_outputs)
    write_latex_rows(output_dir / "response_control_profile_table.tex", model_outputs)
    write_latex_rows(output_dir / "response_control_profile_table_complete_p1_p4.tex", complete_outputs)
    print(f"Wrote {len(model_outputs)} model profiles to {output_dir}")
    print(f"Wrote {len(complete_outputs)} complete P1-P4 model profiles")
    print(f"- {output_dir / 'profile_metrics.json'}")
    print(f"- {output_dir / 'profile_scores.csv'}")
    print(f"- {output_dir / 'profile_scores_complete_p1_p4.csv'}")
    print(f"- {output_dir / 'profile_metrics_flat.csv'}")
    print(f"- {output_dir / 'profile_validity.json'}")
    print(f"- {output_dir / 'profile_validity_complete_p1_p4.json'}")
    print(f"- {output_dir / 'profile_extraction_audit.json'}")
    print(f"- {output_dir / 'response_control_profile_table.tex'}")
    print(f"- {output_dir / 'response_control_profile_table_complete_p1_p4.tex'}")


if __name__ == "__main__":
    main()
