"""Offline evaluator for circuit pilot model/scripted outputs."""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from diagbench.domains.circuit.builder import CIRCUIT_PILOT_VERSION, DOMAIN
from diagbench.domains.circuit.oracle import CircuitOracle
from diagbench.solver.response_json import extract_first_json_object


ACTIONS = ("propose_design", "declare_infeasible", "request_missing_info")
FIELD_ALIASES = {
    "input_voltage_v": "vin_v",
    "source_voltage_v": "vin_v",
    "input_vpp_v": "input_vpp_v",
    "input_voltage_peak_v": "input_vpp_v",
    "input_amplitude_v": "input_vpp_v",
    "load_resistance_ohm": "load_ohm",
    "rl_ohm": "load_ohm",
    "led_forward_voltage_v": "led_vf_v",
    "vf_v": "led_vf_v",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.mean(clean) if clean else None


def _bool_mean(values: list[Any]) -> float | None:
    clean = [value for value in values if value is not None]
    return _safe_mean([1.0 if bool(value) else 0.0 for value in clean])


def _inverse_rate(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return 1.0 - float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round(val, digits) for key, val in value.items()}
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


def _entropy(labels: list[str], universe: tuple[str, ...]) -> float | None:
    if not labels:
        return None
    total = len(labels)
    probs = [labels.count(label) / total for label in universe if labels.count(label)]
    if len(universe) <= 1:
        return 0.0
    return -sum(p * math.log(p) for p in probs) / math.log(len(universe))


def _macro_f1(gold: list[str], pred: list[str]) -> float:
    scores: list[float] = []
    for label in ACTIONS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return statistics.mean(scores)


def _normalize_field_name(value: Any) -> str:
    text = str(value).strip().lower()
    if "." in text:
        text = text.split(".")[-1]
    text = text.replace("-", "_").replace(" ", "_")
    return FIELD_ALIASES.get(text, text)


def _field_exact_score(gold_fields: list[Any], pred_fields: list[Any]) -> float | None:
    if not gold_fields:
        return None
    gold = {_normalize_field_name(item) for item in gold_fields}
    pred = {_normalize_field_name(item) for item in pred_fields}
    return 1.0 if gold == pred else 0.0


def _action_text(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    fields = [action.get("reason", ""), action.get("clarification_request", "")]
    proof = action.get("proof")
    if isinstance(proof, dict):
        fields.append(json.dumps(proof, sort_keys=True))
    elif proof is not None:
        fields.append(str(proof))
    return " ".join(str(item) for item in fields).lower()


def _proof_match_score(task: dict[str, Any], action: dict[str, Any] | None) -> float | None:
    if task.get("gold_label", {}).get("action_type") != "declare_infeasible":
        return None
    requirements = task.get("proof_requirements") or task.get("oracle_metadata", {}).get("proof", {})
    if not isinstance(requirements, dict):
        return None
    text = _action_text(action)
    if not text:
        return 0.0
    score = 0.0
    blockers = requirements.get("blocking_constraints") or requirements.get("blocking_constraint") or requirements.get("metric")
    if isinstance(blockers, str):
        blockers = [blockers]
    blocker_tokens = [
        str(item).lower().replace("_", " ")
        for item in (blockers or [])
        if item is not None
    ]
    if blocker_tokens and any(token in text or token.replace(" ", "_") in text for token in blocker_tokens):
        score += 0.50
    metrics = requirements.get("metrics") or requirements.get("metric")
    if isinstance(metrics, str):
        metrics = [metrics]
    metric_tokens = [
        str(item).lower().replace("_", " ")
        for item in (metrics or [])
        if item is not None
    ]
    if metric_tokens and any(token in text or token.replace(" ", "_") in text for token in metric_tokens):
        score += 0.25
    proof = action.get("proof") if isinstance(action, dict) else None
    has_numeric_bound = isinstance(proof, dict) and any(
        key in proof for key in ("computed_bound", "required_bound", "margin_ratio", "max_achievable", "min_required")
    )
    if has_numeric_bound or re.search(r"\d", text):
        score += 0.25
    return min(score, 1.0)


def _kendall_tau(pred: list[str], gold: list[str]) -> float | None:
    if set(pred) != set(gold) or len(pred) < 2:
        return None
    pred_rank = {cid: idx for idx, cid in enumerate(pred)}
    gold_rank = {cid: idx for idx, cid in enumerate(gold)}
    concordant = 0
    discordant = 0
    ids = list(gold)
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            pred_order = pred_rank[left] < pred_rank[right]
            gold_order = gold_rank[left] < gold_rank[right]
            if pred_order == gold_order:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def _parse_jsonish(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return extract_first_json_object(str(raw))


def _parse_action(row: dict[str, Any], allowed_variables: list[str] | None = None) -> tuple[dict[str, Any] | None, bool]:
    try:
        if isinstance(row.get("parsed_action"), dict):
            data = row["parsed_action"]
        elif isinstance(row.get("action"), dict):
            data = row["action"]
        elif "raw_output" in row:
            data = _parse_jsonish(row["raw_output"])
        else:
            data = row
        action_type = data.get("action_type") or data.get("state_action")
        if action_type is None and allowed_variables:
            allowed = set(allowed_variables)
            candidate_like = {key: value for key, value in data.items() if key in allowed}
            metadata_keys = {"confidence", "reason"}
            if candidate_like and set(data).issubset(allowed | metadata_keys):
                data = {
                    "action_type": "propose_design",
                    "candidate": candidate_like,
                    "reason": data.get("reason", "normalized bare candidate object"),
                    "confidence": data.get("confidence", 0.0),
                }
                action_type = "propose_design"
        if action_type == "reset_history":
            action_type = "replan"
        if action_type not in {"propose_design", "declare_infeasible", "request_missing_info", "replan"}:
            raise ValueError(f"unknown action_type={action_type!r}")
        parsed = {**data, "action_type": action_type}
        if action_type == "propose_design" and not isinstance(parsed.get("candidate"), dict):
            raise ValueError("propose_design missing candidate")
        return parsed, False
    except Exception:
        return None, True


def _parse_steps(row: dict[str, Any], allowed_variables: list[str] | None = None) -> tuple[list[dict[str, Any]], bool]:
    if isinstance(row.get("steps"), list):
        parsed: list[dict[str, Any]] = []
        had_error = False
        for step in row["steps"]:
            action, error = _parse_action(step, allowed_variables)
            had_error = had_error or error
            if action is not None:
                parsed.append(action)
        return parsed, had_error
    action, error = _parse_action(row, allowed_variables)
    return ([action] if action is not None else []), error


def _parse_ranking(row: dict[str, Any]) -> tuple[list[str] | None, bool]:
    try:
        if isinstance(row.get("parsed_response"), dict):
            data = row["parsed_response"]
        elif isinstance(row.get("ranking"), list):
            data = {"ranking": row["ranking"]}
        elif "raw_output" in row:
            data = _parse_jsonish(row["raw_output"])
        else:
            data = row
        ranking = data.get("ranking") or data.get("ranked_candidates")
        if not isinstance(ranking, list):
            raise ValueError("missing ranking")
        normalized = [str(item).strip() for item in ranking]
        if len(normalized) != len(set(normalized)):
            raise ValueError("duplicate ranking ids")
        return normalized, False
    except Exception:
        return None, True


def _log_edit_delta(left: dict[str, float], right: dict[str, float], variables: list[str]) -> float:
    values: list[float] = []
    for variable in variables:
        if variable not in left or variable not in right:
            continue
        a = max(abs(float(left[variable])), 1e-12)
        b = max(abs(float(right[variable])), 1e-12)
        values.append(abs(math.log(b / a)))
    return statistics.mean(values) if values else 0.0


class CircuitPilotEvaluator:
    def __init__(self) -> None:
        self.oracle = CircuitOracle()

    def load_tasks(self, tasks_dir: Path | str) -> dict[str, list[dict[str, Any]]]:
        root = Path(tasks_dir)
        return {
            "P1": _load_jsonl(root / "p1_tasks.jsonl"),
            "P2": _load_jsonl(root / "p2_tasks.jsonl"),
            "P3": _load_jsonl(root / "p3_tasks.jsonl"),
            "P4": _load_jsonl(root / "p4_tasks.jsonl"),
        }

    def evaluate_directory(self, *, results_dir: Path | str, tasks_dir: Path | str) -> dict[str, Any]:
        tasks = self.load_tasks(tasks_dir)
        pilot_version = self._pilot_version(tasks)
        root = Path(results_dir)
        result_rows = {
            "P1": _load_jsonl(root / "p1_results.jsonl"),
            "P2": _load_jsonl(root / "p2_results.jsonl"),
            "P3": _load_jsonl(root / "p3_results.jsonl"),
            "P4": _load_jsonl(root / "p4_results.jsonl"),
        }
        runner_name = self._runner_name(result_rows)
        stage = {
            "P1": self.evaluate_p1(tasks["P1"], result_rows["P1"]),
            "P2": self.evaluate_p2(tasks["P2"], result_rows["P2"]),
            "P3": self.evaluate_p3(tasks["P3"], result_rows["P3"]),
            "P4": self.evaluate_p4(tasks["P4"], result_rows["P4"]),
        }
        profiles = self.profile_scores(stage)
        return {
            "domain": DOMAIN,
            "pilot_version": pilot_version,
            "runner_name": runner_name,
            "stage_metrics": stage,
            "profile_scores": profiles,
        }

    @staticmethod
    def _pilot_version(tasks: dict[str, list[dict[str, Any]]]) -> str:
        versions = {
            str(task.get("pilot_version"))
            for probe_tasks in tasks.values()
            for task in probe_tasks
            if task.get("pilot_version")
        }
        return sorted(versions)[0] if versions else CIRCUIT_PILOT_VERSION

    @staticmethod
    def _runner_name(result_rows: dict[str, list[dict[str, Any]]]) -> str:
        for rows in result_rows.values():
            for row in rows:
                if row.get("runner_name"):
                    return str(row["runner_name"])
        return "unknown"

    def evaluate_p1(self, tasks: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not tasks:
            return {
                "n_tasks": 0,
                "accuracy": None,
                "macro_f1": None,
                "action_entropy": None,
                "spurious_propose_rate": None,
                "unsafe_propose_rate": None,
                "request_recall": None,
                "infeasible_recall": None,
                "feasible_narrow_refusal_rate": None,
                "proposal_feasible_rate": None,
                "missing_field_exact_rate": None,
                "infeasible_proof_score": None,
                "acceptance_credibility": None,
                "missing_discipline": None,
                "infeasibility_discipline": None,
                "worst_action_recall": None,
                "action_imbalance": None,
                "credible_triage_score": None,
                "parse_error_rate": None,
            }
        row_map = {row.get("task_id"): row for row in rows}
        gold: list[str] = []
        pred: list[str] = []
        parse_errors = 0
        feasible_narrow_refusals = 0
        proposal_feasible: list[bool] = []
        missing_field_scores: list[float] = []
        infeasible_proof_scores: list[float] = []
        for task in tasks:
            row = row_map.get(task["task_id"], {})
            action, parse_error = _parse_action(row, task["design_variables"])
            parse_errors += int(parse_error)
            gold_action = task["gold_label"]["action_type"]
            pred_action = action["action_type"] if action is not None else "invalid_output"
            gold.append(gold_action)
            pred.append(pred_action)
            if task["subtype"] == "feasible_narrow" and pred_action != "propose_design":
                feasible_narrow_refusals += 1
            if gold_action == "propose_design":
                feasible = False
                if action is not None and action.get("action_type") == "propose_design" and isinstance(action.get("candidate"), dict):
                    try:
                        feasible = bool(self.oracle.evaluate(task, action["candidate"]).feasible)
                    except Exception:
                        feasible = False
                proposal_feasible.append(feasible)
            if gold_action == "request_missing_info":
                score = _field_exact_score(
                    task.get("missing_fields_ground_truth") or task.get("gold_label", {}).get("missing_fields", []),
                    action.get("missing_fields", []) if isinstance(action, dict) else [],
                )
                if score is not None:
                    missing_field_scores.append(score)
            if gold_action == "declare_infeasible":
                score = _proof_match_score(task, action)
                if score is not None:
                    infeasible_proof_scores.append(score)
        valid_pred = [p if p in ACTIONS else "invalid_output" for p in pred]
        recalls = {label: self._recall(gold, pred, label) for label in ACTIONS}
        spurious = {
            label: self._rate(g != label and p == label for g, p in zip(gold, pred))
            for label in ACTIONS
        }
        acceptance_credibility = recalls["propose_design"] * (1.0 - spurious["propose_design"])
        missing_discipline = recalls["request_missing_info"] * (1.0 - spurious["request_missing_info"])
        infeasibility_discipline = recalls["declare_infeasible"] * (1.0 - spurious["declare_infeasible"])
        pred_distribution = {label: pred.count(label) / len(tasks) for label in ACTIONS}
        gold_distribution = {label: gold.count(label) / len(tasks) for label in ACTIONS}
        action_imbalance = sum(abs(pred_distribution[label] - gold_distribution[label]) for label in ACTIONS)
        proposal_feasible_rate = _bool_mean(proposal_feasible)
        missing_field_exact_rate = _safe_mean(missing_field_scores)
        infeasible_proof_score = _safe_mean(infeasible_proof_scores)
        subtype_f1_proxy = _macro_f1(gold, pred)
        credible_triage_score = _safe_mean([
            0.30 * _macro_f1(gold, pred),
            0.18 * acceptance_credibility,
            0.15 * missing_discipline,
            0.15 * infeasibility_discipline,
            0.10 * subtype_f1_proxy,
            0.07 * (missing_field_exact_rate if missing_field_exact_rate is not None else 0.0),
            0.05 * (infeasible_proof_score if infeasible_proof_score is not None else 0.0),
        ])
        if credible_triage_score is not None:
            credible_triage_score *= 7.0
        return {
            "n_tasks": len(tasks),
            "accuracy": sum(1 for g, p in zip(gold, pred) if g == p) / len(tasks),
            "macro_f1": _macro_f1(gold, pred),
            "action_entropy": _entropy(valid_pred, ("propose_design", "declare_infeasible", "request_missing_info", "invalid_output")),
            "spurious_propose_rate": self._rate(g != "propose_design" and p == "propose_design" for g, p in zip(gold, pred)),
            "unsafe_propose_rate": self._rate(g == "declare_infeasible" and p == "propose_design" for g, p in zip(gold, pred)),
            "request_recall": self._recall(gold, pred, "request_missing_info"),
            "infeasible_recall": self._recall(gold, pred, "declare_infeasible"),
            "feasible_narrow_refusal_rate": feasible_narrow_refusals / max(sum(1 for task in tasks if task["subtype"] == "feasible_narrow"), 1),
            "proposal_feasible_rate": proposal_feasible_rate,
            "missing_field_exact_rate": missing_field_exact_rate,
            "infeasible_proof_score": infeasible_proof_score,
            "acceptance_credibility": acceptance_credibility,
            "missing_discipline": missing_discipline,
            "infeasibility_discipline": infeasibility_discipline,
            "worst_action_recall": min(recalls.values()),
            "action_imbalance": action_imbalance,
            "credible_triage_score": credible_triage_score,
            "parse_error_rate": parse_errors / len(tasks),
        }

    def evaluate_p2(self, tasks: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
        return self._evaluate_repair_probe(tasks, rows, probe="P2")

    def evaluate_p3(self, tasks: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
        row_map = {row.get("task_id"): row for row in rows}
        task_results: list[dict[str, Any]] = []
        for task in tasks:
            row = row_map.get(task["task_id"], {})
            steps, parse_error = _parse_steps(row, task["design_variables"])
            baseline = self.oracle.evaluate(task, task["corrupted_design"])
            evals: list[dict[str, Any]] = []
            candidates: list[dict[str, float]] = []
            explicit_replan = False
            reset_history = False
            for step in steps:
                if step["action_type"] == "replan":
                    explicit_replan = True
                    text = f"{step.get('reason', '')} {step.get('suggested_pivot', '')}".lower()
                    reset_history = reset_history or "reset" in text or "discard" in text
                    continue
                if step["action_type"] != "propose_design":
                    continue
                try:
                    candidate = {key: float(value) for key, value in step["candidate"].items()}
                    result = self.oracle.evaluate(task, candidate)
                except Exception:
                    parse_error = True
                    continue
                candidates.append(candidate)
                evals.append(result.to_dict())
            escape_threshold = baseline.total_violation * float(task["escape_criterion"].get("relative_total_violation", 0.5))
            escape_index = next(
                (idx for idx, result in enumerate(evals) if float(result["total_violation"]) <= escape_threshold),
                None,
            )
            escaped = escape_index is not None
            final = evals[-1] if evals else None
            meaningful_edits = [
                _log_edit_delta(task["corrupted_design"] if idx == 0 else candidates[idx - 1], candidate, task["design_variables"])
                for idx, candidate in enumerate(candidates)
            ]
            cascade = False
            if escaped and escape_index is not None:
                post = evals[escape_index:]
                for left, right in zip(post, post[1:]):
                    if float(right["total_violation"]) > float(left["total_violation"]) + 1e-9:
                        cascade = True
            task_results.append(
                {
                    "parse_error": parse_error,
                    "escaped": escaped,
                    "explicit_replan": explicit_replan,
                    "reset_history": reset_history,
                    "cascade": cascade,
                    "dead_budget": not candidates or max(meaningful_edits or [0.0]) < 0.01,
                    "final_success": bool(final and final["feasible"]),
                    "recovery_quality": float(final["objective_score"]) if final else 0.0,
                    "violation_reduction": bool(final and float(final["total_violation"]) < baseline.total_violation),
                }
            )
        return {
            "n_tasks": len(tasks),
            "escape_rate": _bool_mean([item["escaped"] for item in task_results]),
            "explicit_replan_rate": _bool_mean([item["explicit_replan"] for item in task_results]),
            "reset_history_rate": _bool_mean([item["reset_history"] for item in task_results]),
            "cascade_rate": _bool_mean([item["cascade"] for item in task_results if item["escaped"]]),
            "dead_budget_rate": _bool_mean([item["dead_budget"] for item in task_results]),
            "final_success": _bool_mean([item["final_success"] for item in task_results]),
            "recovery_quality": _safe_mean([item["recovery_quality"] for item in task_results]),
            "violation_reduction_consistency": _bool_mean([item["violation_reduction"] for item in task_results]),
            "raw_history_vs_state_summary_delta": None,
            "parse_error_rate": _bool_mean([item["parse_error"] for item in task_results]),
        }

    def _evaluate_repair_probe(self, tasks: list[dict[str, Any]], rows: list[dict[str, Any]], *, probe: str) -> dict[str, Any]:
        row_map = {row.get("task_id"): row for row in rows}
        final_feasible: list[bool] = []
        final_scores: list[float] = []
        reduction_consistency: list[float] = []
        directed_repair: list[float] = []
        preservation: list[bool] = []
        edit_deltas: list[float] = []
        no_ops: list[bool] = []
        over_edits: list[bool] = []
        query_counts: list[int] = []
        parse_errors: list[bool] = []
        for task in tasks:
            row = row_map.get(task["task_id"], {})
            steps, parse_error = _parse_steps(row, task["design_variables"])
            parse_errors.append(parse_error)
            current_design = task.get("initial_design") or task.get("corrupted_design")
            current_eval = self.oracle.evaluate(task, current_design).to_dict()
            evals = [current_eval]
            designs = [current_design]
            for step in steps:
                if step["action_type"] != "propose_design":
                    continue
                try:
                    design = {key: float(value) for key, value in step["candidate"].items()}
                    result = self.oracle.evaluate(task, design).to_dict()
                except Exception:
                    parse_errors[-1] = True
                    continue
                designs.append(design)
                evals.append(result)
            query_counts.append(max(0, len(evals) - 1))
            final = evals[-1]
            final_feasible.append(bool(final["feasible"]))
            final_scores.append(float(final["objective_score"]))
            reductions = []
            repairs = []
            for left, right in zip(evals, evals[1:]):
                if float(left["total_violation"]) > 0:
                    reductions.append(float(right["total_violation"]) < float(left["total_violation"]) - 1e-9)
                    dominant = self._dominant_violation(left)
                    if dominant is not None:
                        right_same = self._violation_by_metric(right, dominant["metric"])
                        repairs.append(right_same < float(dominant["normalized_violation"]))
                if left["feasible"]:
                    preservation.append(bool(right["feasible"]))
            reduction_consistency.append(_bool_mean(reductions) or 0.0)
            directed_repair.append(_bool_mean(repairs) or 0.0)
            deltas = [_log_edit_delta(left, right, task["design_variables"]) for left, right in zip(designs, designs[1:])]
            if deltas:
                edit_deltas.extend(deltas)
                no_ops.extend(delta < 0.01 for delta in deltas)
                over_edits.extend(delta > 0.75 for delta in deltas)
        return {
            "n_tasks": len(tasks),
            "final_feasible_rate": _bool_mean(final_feasible),
            "final_objective_score": _safe_mean(final_scores),
            "violation_reduction_consistency": _safe_mean(reduction_consistency),
            "directed_repair_rate": _safe_mean(directed_repair),
            "feasibility_preservation": _bool_mean(preservation),
            "mean_log_edit_delta": _safe_mean(edit_deltas),
            "no_op_rate": _bool_mean(no_ops),
            "over_edit_rate": _bool_mean(over_edits),
            "query_count": _safe_mean([float(value) for value in query_counts]),
            "parse_error_rate": _bool_mean(parse_errors),
        }

    @staticmethod
    def _dominant_violation(result: dict[str, Any]) -> dict[str, Any] | None:
        violations = result.get("violations") or []
        if not violations:
            return None
        return max(violations, key=lambda item: float(item["normalized_violation"]))

    @staticmethod
    def _violation_by_metric(result: dict[str, Any], metric: str) -> float:
        for violation in result.get("violations") or []:
            if violation.get("metric") == metric:
                return float(violation["normalized_violation"])
        return 0.0

    def evaluate_p4(self, tasks: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
        row_map = {row.get("task_id"): row for row in rows}
        tau_values: list[float] = []
        tau_scaled: list[float] = []
        exact: list[bool] = []
        top1: list[bool] = []
        top2: list[bool] = []
        pairwise: list[float] = []
        flip_acc: list[float] = []
        bars: list[float] = []
        parse_errors: list[bool] = []
        for task in tasks:
            ranking, parse_error = _parse_ranking(row_map.get(task["task_id"], {}))
            gold = task["oracle_reference_ranking"]
            parse_error = parse_error or ranking is None or set(ranking or []) != set(gold)
            parse_errors.append(parse_error)
            if parse_error or ranking is None:
                tau_values.append(-1.0)
                tau_scaled.append(0.0)
                exact.append(False)
                top1.append(False)
                top2.append(False)
                pairwise.append(0.0)
                flip_acc.append(0.0)
                bars.append(0.0)
                continue
            tau = _kendall_tau(ranking, gold)
            tau = float(tau if tau is not None else -1.0)
            pair = self._pairwise_accuracy(ranking, gold)
            flip = self._policy_flip_accuracy(ranking, task)
            ex = ranking == gold
            t1 = ranking[0] == gold[0]
            t2 = set(ranking[:2]) == set(gold[:2])
            scaled = (tau + 1.0) / 2.0
            tau_values.append(tau)
            tau_scaled.append(scaled)
            exact.append(ex)
            top1.append(t1)
            top2.append(t2)
            pairwise.append(pair)
            flip_acc.append(flip)
            bars.append(0.55 * scaled + 0.25 * flip + 0.20 * (1.0 if ex else 0.0))
        return {
            "n_tasks": len(tasks),
            "full_kendall_tau": _safe_mean(tau_values),
            "full_tau_scaled": _safe_mean(tau_scaled),
            "exact_match": _bool_mean(exact),
            "top1_accuracy": _bool_mean(top1),
            "top2_set_accuracy": _bool_mean(top2),
            "pairwise_accuracy": _safe_mean(pairwise),
            "policy_flip_accuracy": _safe_mean(flip_acc),
            "bars": _safe_mean(bars),
            "parse_error_rate": _bool_mean(parse_errors),
        }

    @staticmethod
    def _pairwise_accuracy(pred: list[str], gold: list[str]) -> float:
        pred_rank = {cid: idx for idx, cid in enumerate(pred)}
        gold_rank = {cid: idx for idx, cid in enumerate(gold)}
        total = 0
        correct = 0
        for idx, left in enumerate(gold):
            for right in gold[idx + 1 :]:
                total += 1
                correct += int((pred_rank[left] < pred_rank[right]) == (gold_rank[left] < gold_rank[right]))
        return correct / total if total else 0.0

    @staticmethod
    def _policy_flip_accuracy(pred: list[str], task: dict[str, Any]) -> float:
        pred_rank = {cid: idx for idx, cid in enumerate(pred)}
        total = 0
        correct = 0
        for pair in task.get("policy_flip_pairs", []):
            better = pair["policy_better"]
            other = pair["right"] if pair["left"] == better else pair["left"]
            if better not in pred_rank or other not in pred_rank:
                continue
            total += 1
            correct += int(pred_rank[better] < pred_rank[other])
        return correct / total if total else 0.0

    @staticmethod
    def _rate(values: Any) -> float:
        items = list(values)
        return sum(1 for value in items if value) / len(items) if items else 0.0

    @staticmethod
    def _recall(gold: list[str], pred: list[str], label: str) -> float:
        denom = sum(1 for item in gold if item == label)
        if not denom:
            return 0.0
        return sum(1 for g, p in zip(gold, pred) if g == label and p == label) / denom

    @staticmethod
    def profile_scores(stage: dict[str, dict[str, Any]]) -> dict[str, float | None]:
        p1 = stage["P1"]
        p2 = stage["P2"]
        p3 = stage["P3"]
        p4 = stage["P4"]
        return {
            "action_prior": _safe_mean([
                p1.get("accuracy"),
                p1.get("macro_f1"),
                _inverse_rate(p1.get("unsafe_propose_rate")),
                _inverse_rate(p1.get("feasible_narrow_refusal_rate")),
            ]),
            "edit_style": _safe_mean([
                p2.get("final_feasible_rate"),
                p2.get("feasibility_preservation"),
                _inverse_rate(p2.get("over_edit_rate")),
                _inverse_rate(p2.get("no_op_rate")),
            ]),
            "feedback_obedience": _safe_mean([
                p2.get("violation_reduction_consistency"),
                p2.get("directed_repair_rate"),
                p3.get("violation_reduction_consistency"),
            ]),
            "state_trust": _safe_mean([
                p3.get("escape_rate"),
                p3.get("explicit_replan_rate"),
                _inverse_rate(p3.get("cascade_rate")),
                _inverse_rate(p3.get("dead_budget_rate")),
                p3.get("final_success"),
            ]),
            "preference_execution": _safe_mean([
                p4.get("full_tau_scaled"),
                p4.get("policy_flip_accuracy"),
                p4.get("exact_match"),
                p4.get("top1_accuracy"),
                _inverse_rate(p4.get("parse_error_rate")),
            ]),
        }

    def write_outputs(self, *, summary: dict[str, Any], out_dir: Path | str, overwrite: bool = False) -> None:
        out = Path(out_dir)
        if out.exists() and not overwrite:
            raise FileExistsError(f"Output directory exists: {out}")
        out.mkdir(parents=True, exist_ok=True)
        summary = _round(summary)
        _write_json(out / "summary.json", summary)
        _write_json(out / "audit_failures.json", self._audit_failures(summary))
        self._write_profile_csv(out / "profile_scores.csv", summary)
        (out / "pilot_table.md").write_text(self._pilot_table(summary) + "\n")

    @staticmethod
    def _audit_failures(summary: dict[str, Any]) -> dict[str, Any]:
        failures = []
        for probe, metrics in summary["stage_metrics"].items():
            if float(metrics.get("parse_error_rate") or 0.0) > 0:
                failures.append({"probe": probe, "metric": "parse_error_rate", "value": metrics["parse_error_rate"]})
        return {"runner_name": summary["runner_name"], "failures": failures}

    @staticmethod
    def _write_profile_csv(path: Path, summary: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["runner_name", "action_prior", "edit_style", "feedback_obedience", "state_trust", "preference_execution"],
            )
            writer.writeheader()
            row = {"runner_name": summary["runner_name"], **summary["profile_scores"]}
            writer.writerow(row)

    @staticmethod
    def _pilot_table(summary: dict[str, Any]) -> str:
        def fmt(value: Any) -> str:
            if value is None:
                return "NA"
            try:
                number = float(value)
            except (TypeError, ValueError):
                return "NA"
            return f"{number:.3f}" if math.isfinite(number) else "NA"

        stage = summary["stage_metrics"]
        profiles = summary["profile_scores"]
        pilot_version = summary.get("pilot_version", CIRCUIT_PILOT_VERSION)
        title = str(pilot_version).replace("_", " ")
        lines = [
            f"# {title} Summary: {summary['runner_name']}",
            "",
            "| Probe | Headline | Parse errors |",
            "|---|---:|---:|",
            f"| P1 | accuracy {fmt(stage['P1']['accuracy'])}, macro-F1 {fmt(stage['P1']['macro_f1'])} | {fmt(stage['P1']['parse_error_rate'])} |",
            f"| P2 | final feasible {fmt(stage['P2']['final_feasible_rate'])}, objective {fmt(stage['P2']['final_objective_score'])} | {fmt(stage['P2']['parse_error_rate'])} |",
            f"| P3 | success {fmt(stage['P3']['final_success'])}, escape {fmt(stage['P3']['escape_rate'])} | {fmt(stage['P3']['parse_error_rate'])} |",
            f"| P4 | tau {fmt(stage['P4']['full_kendall_tau'])}, flip {fmt(stage['P4']['policy_flip_accuracy'])} | {fmt(stage['P4']['parse_error_rate'])} |",
            "",
            "| Profile | Score |",
            "|---|---:|",
        ]
        for key, value in profiles.items():
            lines.append(f"| {key} | {fmt(value)} |")
        return "\n".join(lines)
