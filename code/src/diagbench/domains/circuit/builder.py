"""Task and audit generation for the 56-task circuit pilot."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diagbench.domains.circuit.oracle import CircuitOracle


CIRCUIT_PILOT_VERSION = "circuit_pilot_v1"
DOMAIN = "circuit"
TASK_COUNTS = {"P1": 16, "P2": 16, "P3": 12, "P4": 12}


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _task_hash(task: dict[str, Any]) -> str:
    return _sha256_text(_canonical_json(task))


def _manifest(tasks: list[dict[str, Any]], *, seed: int, probe: str, artifact_path: Path) -> dict[str, Any]:
    return {
        "domain": DOMAIN,
        "pilot_version": CIRCUIT_PILOT_VERSION,
        "probe": probe,
        "n_tasks": len(tasks),
        "seed": seed,
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256_text("\n".join(_canonical_json(task) for task in tasks)),
        "task_ids": [task["task_id"] for task in tasks],
    }


def _base_task(
    *,
    task_id: str,
    probe: str,
    family: str,
    subtype: str,
    spec: dict[str, Any],
    design_variables: list[str],
    variable_bounds: dict[str, dict[str, float | str]],
    constraints: list[dict[str, Any]],
    objective: dict[str, Any],
    query_budget: int,
    split: str = "test_audit",
    best_known_feasible: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = {
        "task_id": task_id,
        "domain": DOMAIN,
        "pilot_version": CIRCUIT_PILOT_VERSION,
        "probe": probe,
        "family": family,
        "subtype": subtype,
        "split": split,
        "spec": spec,
        "design_variables": design_variables,
        "variable_bounds": variable_bounds,
        "constraints": constraints,
        "objective": objective,
        "query_budget": query_budget,
        "best_known_feasible": best_known_feasible,
        "oracle_metadata": {
            "oracle": "closed_form_circuit_oracle",
            "oracle_version": "circuit_oracle_v0.1",
            "unit_system": "SI",
        },
    }
    if extra:
        task.update(extra)
    task["task_sha256"] = _task_hash({k: v for k, v in task.items() if k != "task_sha256"})
    return task


def _target_log(name: str, metric: str, target: float, tolerance_rel: float, unit: str) -> dict[str, Any]:
    return {"name": name, "metric": metric, "type": "target_log", "target": target, "tolerance_rel": tolerance_rel, "unit": unit}


def _target_rel(name: str, metric: str, target: float, tolerance_rel: float, unit: str) -> dict[str, Any]:
    return {"name": name, "metric": metric, "type": "target_rel", "target": target, "tolerance_rel": tolerance_rel, "unit": unit}


def _upper(name: str, metric: str, limit: float, unit: str, scale: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "metric": metric, "type": "upper_bound", "limit": limit, "unit": unit}
    if scale is not None:
        out["scale"] = scale
    return out


def _lower(name: str, metric: str, limit: float, unit: str, scale: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "metric": metric, "type": "lower_bound", "limit": limit, "unit": unit}
    if scale is not None:
        out["scale"] = scale
    return out


def _rc_task_spec(target_fc_hz: float = 1000.0, vin_v: float = 5.0) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    spec = {"filter_type": "lowpass", "target_fc_hz": target_fc_hz, "vin_v": vin_v}
    variables = ["R_ohm", "C_f"]
    bounds = {
        "R_ohm": {"min": 1000.0, "max": 100000.0, "unit": "ohm"},
        "C_f": {"min": 1e-9, "max": 1e-6, "unit": "F"},
    }
    constraints = [
        _target_log("cutoff_frequency", "fc_hz", target_fc_hz, 0.05, "Hz"),
        _upper("source_current", "source_current_a", 0.003, "A", scale=0.003),
    ]
    objective = {"name": "low_source_current", "metric": "source_current_a", "direction": "minimize", "best": 0.00005, "worst": 0.003}
    return spec, variables, bounds, constraints, objective


def _divider_task_spec(target_vout_v: float = 2.5, load_ohm: float = 100000.0) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    spec = {"vin_v": 5.0, "target_vout_v": target_vout_v, "load_ohm": load_ohm}
    variables = ["R1_ohm", "R2_ohm"]
    bounds = {
        "R1_ohm": {"min": 1000.0, "max": 200000.0, "unit": "ohm"},
        "R2_ohm": {"min": 1000.0, "max": 200000.0, "unit": "ohm"},
    }
    constraints = [
        _target_rel("output_voltage", "vout_v", target_vout_v, 0.03, "V"),
        _upper("divider_current", "divider_current_a", 0.001, "A", scale=0.001),
    ]
    objective = {"name": "low_power", "metric": "power_w", "direction": "minimize", "best": 0.00002, "worst": 0.005}
    return spec, variables, bounds, constraints, objective


def _led_task_spec(target_current_a: float = 0.01) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    spec = {"supply_v": 5.0, "led_vf_v": 2.0, "target_current_a": target_current_a, "resistor_power_rating_w": 0.25}
    variables = ["R_ohm"]
    bounds = {"R_ohm": {"min": 50.0, "max": 2000.0, "unit": "ohm"}}
    constraints = [
        _target_rel("led_current", "led_current_a", target_current_a, 0.08, "A"),
        _upper("resistor_power", "resistor_power_w", 0.25, "W", scale=0.25),
    ]
    objective = {"name": "power_margin", "metric": "safety_margin_w", "direction": "maximize", "best": 0.25, "worst": 0.0}
    return spec, variables, bounds, constraints, objective


def _opamp_task_spec(target_gain: float = 10.0, mode: str = "non_inverting") -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if mode == "inverting":
        variables = ["Rf_ohm", "Rin_ohm", "Rload_ohm"]
        bounds = {
            "Rf_ohm": {"min": 1000.0, "max": 200000.0, "unit": "ohm"},
            "Rin_ohm": {"min": 1000.0, "max": 100000.0, "unit": "ohm"},
            "Rload_ohm": {"min": 1000.0, "max": 50000.0, "unit": "ohm"},
        }
    else:
        variables = ["Rf_ohm", "Rg_ohm", "Rload_ohm"]
        bounds = {
            "Rf_ohm": {"min": 1000.0, "max": 200000.0, "unit": "ohm"},
            "Rg_ohm": {"min": 1000.0, "max": 100000.0, "unit": "ohm"},
            "Rload_ohm": {"min": 1000.0, "max": 50000.0, "unit": "ohm"},
        }
    spec = {
        "mode": mode,
        "target_gain": target_gain,
        "gbw_hz": 10_000_000.0,
        "min_bandwidth_hz": 100_000.0,
        "vcc_v": 5.0,
        "vsat_v": 0.7,
        "output_current_limit_a": 0.02,
        "load_ohm": 10000.0,
        "input_vpp_v": 0.2,
    }
    constraints = [
        _target_log("closed_loop_gain", "gain_v_per_v", target_gain, 0.05, "V/V"),
        _lower("minimum_bandwidth", "bandwidth_hz", 100_000.0, "Hz", scale=100_000.0),
        _lower("swing_margin", "swing_margin_v", 0.2, "V", scale=1.0),
        _lower("input_impedance", "input_impedance_ohm", 8000.0, "ohm", scale=8000.0),
    ]
    objective = {"name": "bandwidth_margin", "metric": "bandwidth_hz", "direction": "maximize", "best": 2_000_000.0, "worst": 100_000.0}
    return spec, variables, bounds, constraints, objective


def _regulator_task_spec(target_vout_v: float = 3.3) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    spec = {"vin_v": 5.0, "target_vout_v": target_vout_v, "load_current_a": 0.2, "ambient_c": 25.0}
    variables = ["vout_v", "dropout_v", "thermal_resistance_c_per_w"]
    bounds = {
        "vout_v": {"min": 1.2, "max": 5.0, "unit": "V"},
        "dropout_v": {"min": 0.1, "max": 1.5, "unit": "V"},
        "thermal_resistance_c_per_w": {"min": 20.0, "max": 120.0, "unit": "C/W"},
    }
    constraints = [
        _target_rel("output_voltage", "vout_v", target_vout_v, 0.03, "V"),
        _lower("dropout_margin", "dropout_margin_v", 0.3, "V", scale=1.0),
        _upper("junction_temp", "junction_temp_c", 85.0, "C", scale=85.0),
    ]
    objective = {"name": "low_power_loss", "metric": "power_loss_w", "direction": "minimize", "best": 0.1, "worst": 1.0}
    return spec, variables, bounds, constraints, objective


@dataclass
class CircuitPilotBuilder:
    seed: int = 1701

    def __post_init__(self) -> None:
        self.oracle = CircuitOracle()
        self.rng = random.Random(self.seed)

    def build(self) -> dict[str, list[dict[str, Any]]]:
        tasks = {
            "P1": self.build_p1_tasks(),
            "P2": self.build_p2_tasks(),
            "P3": self.build_p3_tasks(),
            "P4": self.build_p4_tasks(),
        }
        for probe, expected in TASK_COUNTS.items():
            if len(tasks[probe]) != expected:
                raise AssertionError(f"{probe} expected {expected} tasks, got {len(tasks[probe])}")
        return tasks

    def write(self, *, out_dir: Path, audit_dir: Path, overwrite: bool = False) -> dict[str, list[dict[str, Any]]]:
        if out_dir.exists() and not overwrite:
            raise FileExistsError(f"Output directory exists: {out_dir}")
        if audit_dir.exists() and not overwrite:
            raise FileExistsError(f"Audit directory exists: {audit_dir}")
        tasks_by_probe = self.build()
        out_dir.mkdir(parents=True, exist_ok=True)
        audit_dir.mkdir(parents=True, exist_ok=True)
        for probe, tasks in tasks_by_probe.items():
            task_path = out_dir / f"{probe.lower()}_tasks.jsonl"
            _write_jsonl(task_path, tasks)
            _write_json(out_dir / f"{probe.lower()}_manifest.json", _manifest(tasks, seed=self.seed, probe=probe, artifact_path=task_path))
            for task in tasks:
                self.write_audit_bundle(task=task, audit_root=audit_dir)
        _write_json(out_dir / "dataset_summary.json", self.dataset_summary(tasks_by_probe))
        return tasks_by_probe

    def dataset_summary(self, tasks_by_probe: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return {
            "domain": DOMAIN,
            "pilot_version": CIRCUIT_PILOT_VERSION,
            "seed": self.seed,
            "task_counts": {probe: len(tasks) for probe, tasks in tasks_by_probe.items()},
            "families": sorted({task["family"] for tasks in tasks_by_probe.values() for task in tasks}),
            "purpose": "cross-domain construct-validity check for P1-P4 response-control profiles",
        }

    def build_p1_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        specs = [
            ("rc_filter", _rc_task_spec(), {"R_ohm": 10000.0, "C_f": 1.591549e-8}),
            ("loaded_divider", _divider_task_spec(), {"R1_ohm": 10000.0, "R2_ohm": 11111.111111}),
            ("led_current_limit", _led_task_spec(), {"R_ohm": 300.0}),
            ("op_amp_amplifier", _opamp_task_spec(), {"Rf_ohm": 90000.0, "Rg_ohm": 10000.0, "Rload_ohm": 10000.0}),
        ]
        for idx, (family, parts, ref) in enumerate(specs):
            spec, variables, bounds, constraints, objective = parts
            tasks.append(
                _base_task(
                    task_id=f"{CIRCUIT_PILOT_VERSION}::P1::propose_design::{idx:02d}",
                    probe="P1",
                    family=family,
                    subtype="propose_design",
                    spec=spec,
                    design_variables=variables,
                    variable_bounds=bounds,
                    constraints=constraints,
                    objective=objective,
                    query_budget=1,
                    best_known_feasible=ref,
                    extra={
                        "gold_label": {"action_type": "propose_design"},
                        "reference_design": ref,
                        "oracle_metadata": {
                            "oracle": "closed_form_circuit_oracle",
                            "oracle_version": "circuit_oracle_v0.1",
                            "unit_system": "SI",
                            "proof": {"proof_type": "reference_design_feasible", "reference_design": ref},
                        },
                    },
                )
            )

        infeasible_cases = [
            ("led_current_limit", _led_task_spec(0.1), {"supply_v": 3.3, "led_vf_v": 3.0, "target_current_a": 0.1, "resistor_power_rating_w": 0.25}, "max_current = (3.3 - 3.0)/50 = 6mA < required 100mA"),
            ("op_amp_amplifier", _opamp_task_spec(1000.0), {"mode": "non_inverting", "target_gain": 1000.0, "gbw_hz": 1_000_000.0, "min_bandwidth_hz": 10_000.0, "vcc_v": 5.0, "vsat_v": 0.7, "output_current_limit_a": 0.02, "load_ohm": 10000.0, "input_vpp_v": 0.2}, "GBW/gain = 1kHz < required 10kHz at gain 1000"),
            ("linear_regulator", _regulator_task_spec(5.0), {"vin_v": 5.0, "target_vout_v": 5.0, "load_current_a": 0.2, "ambient_c": 25.0}, "dropout margin is negative for Vin=Vout=5V"),
            ("loaded_divider", _divider_task_spec(10.0), {"vin_v": 5.0, "target_vout_v": 10.0, "load_ohm": 100000.0}, "passive divider cannot exceed Vin=5V"),
        ]
        for idx, (family, parts, spec_override, proof) in enumerate(infeasible_cases):
            spec, variables, bounds, constraints, objective = parts
            spec = {**spec, **spec_override}
            if family == "loaded_divider":
                constraints = [_target_rel("output_voltage", "vout_v", 10.0, 0.03, "V")]
            tasks.append(
                _base_task(
                    task_id=f"{CIRCUIT_PILOT_VERSION}::P1::declare_infeasible::{idx:02d}",
                    probe="P1",
                    family=family,
                    subtype="declare_infeasible",
                    spec=spec,
                    design_variables=variables,
                    variable_bounds=bounds,
                    constraints=constraints,
                    objective=objective,
                    query_budget=1,
                    best_known_feasible=None,
                    extra={
                        "gold_label": {"action_type": "declare_infeasible", "reason": proof},
                        "oracle_metadata": {
                            "oracle": "closed_form_circuit_oracle",
                            "oracle_version": "circuit_oracle_v0.1",
                            "unit_system": "SI",
                            "proof": {"proof_type": "closed_form_infeasibility", "argument": proof},
                        },
                    },
                )
            )

        missing_cases = [
            ("op_amp_amplifier", _opamp_task_spec(), ["spec.vcc_v", "spec.output_current_limit_a"], "cannot check swing without supply/current limits"),
            ("loaded_divider", _divider_task_spec(), ["spec.load_ohm"], "loaded divider output depends on load"),
            ("led_current_limit", _led_task_spec(), ["spec.supply_v", "spec.led_vf_v"], "current requires supply and forward voltage"),
            ("rc_filter", _rc_task_spec(), ["spec.target_fc_hz"], "cutoff target is required"),
        ]
        for idx, (family, parts, missing, reason) in enumerate(missing_cases):
            spec, variables, bounds, constraints, objective = parts
            for field in missing:
                _, key = field.split(".", 1)
                spec.pop(key, None)
            tasks.append(
                _base_task(
                    task_id=f"{CIRCUIT_PILOT_VERSION}::P1::request_missing_info::{idx:02d}",
                    probe="P1",
                    family=family,
                    subtype="request_missing_info",
                    spec=spec,
                    design_variables=variables,
                    variable_bounds=bounds,
                    constraints=constraints,
                    objective=objective,
                    query_budget=1,
                    best_known_feasible=None,
                    extra={
                        "gold_label": {"action_type": "request_missing_info", "missing_fields": missing, "reason": reason},
                        "missing_fields_ground_truth": missing,
                        "oracle_metadata": {
                            "oracle": "closed_form_circuit_oracle",
                            "oracle_version": "circuit_oracle_v0.1",
                            "unit_system": "SI",
                            "proof": {"proof_type": "missing_blocker", "missing_fields": missing, "argument": reason},
                        },
                    },
                )
            )

        narrow_cases = [
            ("rc_filter", _rc_task_spec(1000.0), {"R_ohm": 15915.494309, "C_f": 1e-8}),
            ("op_amp_amplifier", _opamp_task_spec(20.0, mode="inverting"), {"Rf_ohm": 200000.0, "Rin_ohm": 10000.0, "Rload_ohm": 20000.0}),
            ("loaded_divider", _divider_task_spec(1.25, load_ohm=10000.0), {"R1_ohm": 30000.0, "R2_ohm": 10000.0}),
            ("linear_regulator", _regulator_task_spec(4.2), {"vout_v": 4.2, "dropout_v": 0.3, "thermal_resistance_c_per_w": 35.0}),
        ]
        for idx, (family, parts, ref) in enumerate(narrow_cases):
            spec, variables, bounds, constraints, objective = parts
            tasks.append(
                _base_task(
                    task_id=f"{CIRCUIT_PILOT_VERSION}::P1::feasible_narrow::{idx:02d}",
                    probe="P1",
                    family=family,
                    subtype="feasible_narrow",
                    spec=spec,
                    design_variables=variables,
                    variable_bounds=bounds,
                    constraints=constraints,
                    objective=objective,
                    query_budget=1,
                    best_known_feasible=ref,
                    extra={
                        "gold_label": {"action_type": "propose_design"},
                        "reference_design": ref,
                        "oracle_metadata": {
                            "oracle": "closed_form_circuit_oracle",
                            "oracle_version": "circuit_oracle_v0.1",
                            "unit_system": "SI",
                            "proof": {"proof_type": "narrow_reference_design_feasible", "reference_design": ref},
                        },
                    },
                )
            )
        return tasks

    def build_p2_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        variants = [
            ("rc_filter", _rc_task_spec(1000.0), {"R_ohm": 10000.0, "C_f": 1e-9}, {"R_ohm": 10000.0, "C_f": 1.591549e-8}),
            ("rc_filter", _rc_task_spec(2500.0), {"R_ohm": 80000.0, "C_f": 1e-8}, {"R_ohm": 20000.0, "C_f": 3.183099e-9}),
            ("rc_filter", _rc_task_spec(400.0), {"R_ohm": 5000.0, "C_f": 1e-8}, {"R_ohm": 33000.0, "C_f": 1.205681e-8}),
            ("rc_filter", _rc_task_spec(1500.0), {"R_ohm": 1000.0, "C_f": 1e-8}, {"R_ohm": 22000.0, "C_f": 4.822877e-9}),
            ("loaded_divider", _divider_task_spec(2.5), {"R1_ohm": 1000.0, "R2_ohm": 1000.0}, {"R1_ohm": 10000.0, "R2_ohm": 11111.111111}),
            ("loaded_divider", _divider_task_spec(1.8, load_ohm=47000.0), {"R1_ohm": 10000.0, "R2_ohm": 10000.0}, {"R1_ohm": 18000.0, "R2_ohm": 13000.0}),
            ("loaded_divider", _divider_task_spec(3.3, load_ohm=22000.0), {"R1_ohm": 22000.0, "R2_ohm": 10000.0}, {"R1_ohm": 6800.0, "R2_ohm": 33000.0}),
            ("loaded_divider", _divider_task_spec(1.2, load_ohm=10000.0), {"R1_ohm": 10000.0, "R2_ohm": 10000.0}, {"R1_ohm": 30000.0, "R2_ohm": 180000.0}),
            ("op_amp_amplifier", _opamp_task_spec(10.0), {"Rf_ohm": 10000.0, "Rg_ohm": 10000.0, "Rload_ohm": 1000.0}, {"Rf_ohm": 90000.0, "Rg_ohm": 10000.0, "Rload_ohm": 10000.0}),
            ("op_amp_amplifier", _opamp_task_spec(20.0, mode="inverting"), {"Rf_ohm": 10000.0, "Rin_ohm": 10000.0, "Rload_ohm": 1000.0}, {"Rf_ohm": 200000.0, "Rin_ohm": 10000.0, "Rload_ohm": 20000.0}),
            ("op_amp_amplifier", _opamp_task_spec(5.0), {"Rf_ohm": 200000.0, "Rg_ohm": 1000.0, "Rload_ohm": 1000.0}, {"Rf_ohm": 40000.0, "Rg_ohm": 10000.0, "Rload_ohm": 20000.0}),
            ("op_amp_amplifier", _opamp_task_spec(15.0, mode="inverting"), {"Rf_ohm": 150000.0, "Rin_ohm": 1000.0, "Rload_ohm": 1000.0}, {"Rf_ohm": 150000.0, "Rin_ohm": 10000.0, "Rload_ohm": 15000.0}),
            ("led_current_limit", _led_task_spec(0.01), {"R_ohm": 50.0}, {"R_ohm": 300.0}),
            ("led_current_limit", _led_task_spec(0.005), {"R_ohm": 2000.0}, {"R_ohm": 600.0}),
            ("linear_regulator", _regulator_task_spec(3.3), {"vout_v": 4.8, "dropout_v": 1.0, "thermal_resistance_c_per_w": 120.0}, {"vout_v": 3.3, "dropout_v": 0.3, "thermal_resistance_c_per_w": 45.0}),
            ("linear_regulator", _regulator_task_spec(2.5), {"vout_v": 4.6, "dropout_v": 0.8, "thermal_resistance_c_per_w": 110.0}, {"vout_v": 2.5, "dropout_v": 0.25, "thermal_resistance_c_per_w": 35.0}),
        ]
        for idx, (family, parts, initial, best) in enumerate(variants):
            spec, variables, bounds, constraints, objective = parts
            task = _base_task(
                task_id=f"{CIRCUIT_PILOT_VERSION}::P2::{idx:02d}",
                probe="P2",
                family=family,
                subtype="iterative_repair",
                spec=spec,
                design_variables=variables,
                variable_bounds=bounds,
                constraints=constraints,
                objective=objective,
                query_budget=4,
                best_known_feasible=best,
                extra={"initial_design": initial, "allowed_edits": variables},
            )
            task["initial_oracle_result"] = self.oracle.evaluate(task, initial).to_dict()
            task["oracle_reference_result"] = self.oracle.evaluate(task, best).to_dict()
            tasks.append(task)
        return tasks

    def build_p3_tasks(self) -> list[dict[str, Any]]:
        p2_sources = self.build_p2_tasks()
        corruption_types = [
            "unit_flip",
            "wrong_formula_direction",
            "false_feasibility",
            "topology_trap",
            "verifier_ignored",
            "progressive_contamination",
        ]
        tasks: list[dict[str, Any]] = []
        for idx, corruption in enumerate(corruption_types * 2):
            source = p2_sources[(idx * 3) % len(p2_sources)]
            bad = dict(source["initial_design"])
            best = dict(source["best_known_feasible"])
            family = source["family"]
            corrupted_history = self._corrupted_history(corruption=corruption, task=source, bad=bad)
            task = _base_task(
                task_id=f"{CIRCUIT_PILOT_VERSION}::P3::{corruption}::{idx:02d}",
                probe="P3",
                family=family,
                subtype=corruption,
                spec=source["spec"],
                design_variables=source["design_variables"],
                variable_bounds=source["variable_bounds"],
                constraints=source["constraints"],
                objective=source["objective"],
                query_budget=4,
                best_known_feasible=best,
                extra={
                    "corruption_type": corruption,
                    "corrupted_design": bad,
                    "corrupted_history": corrupted_history,
                    "hidden_truth": {
                        "corrupted_oracle_result": self.oracle.evaluate(source, bad).to_dict(),
                        "recovery_design": best,
                        "recovery_oracle_result": self.oracle.evaluate(source, best).to_dict(),
                    },
                    "escape_criterion": {
                        "type": "violation_reduction",
                        "relative_total_violation": 0.5,
                        "description": "Escape when a proposal cuts total violation by at least half relative to the corrupted design.",
                    },
                },
            )
            tasks.append(task)
        return tasks

    def _corrupted_history(self, *, corruption: str, task: dict[str, Any], bad: dict[str, float]) -> list[dict[str, Any]]:
        result = self.oracle.evaluate(task, bad).to_dict()
        text = {
            "unit_flip": "Previous analysis treated nF as uF, so the numeric capacitance looked much larger than it is.",
            "wrong_formula_direction": "Previous analysis claimed reducing the RC product lowers cutoff frequency, which reverses the formula direction.",
            "false_feasibility": "Previous analysis declared the design feasible without recomputing the verifier metrics.",
            "topology_trap": "Previous analysis kept the same topology even though the dominant violation is structural for the current setting.",
            "verifier_ignored": "Previous analysis explained away the verifier failure as harmless instead of acting on it.",
            "progressive_contamination": "The first steps were reasonable, then a small unit/sign mistake entered mid-trajectory and subsequent steps trusted it.",
        }[corruption]
        return [
            {
                "step": 0,
                "action_type": "propose_design",
                "candidate": bad,
                "analysis_summary": text,
                "verifier_response": result,
            },
            {
                "step": 1,
                "action_type": "propose_design",
                "candidate": bad,
                "analysis_summary": "Continue the same repair direction based on the prior trajectory.",
                "verifier_response": result,
            },
        ]

    def build_p4_tasks(self) -> list[dict[str, Any]]:
        pools = self._p4_pools()
        tasks: list[dict[str, Any]] = []
        for pool_idx, pool in enumerate(pools):
            base = pool["base"]
            candidates = pool["candidates"]
            policies = pool["policies"]
            candidate_metrics = []
            for candidate in candidates:
                result = self.oracle.evaluate(base, candidate["design"]).to_dict()
                if not result["feasible"]:
                    raise AssertionError(f"P4 candidate is not feasible in pool {pool_idx}: {candidate['candidate_id']}")
                candidate_metrics.append({**candidate, "oracle_result": result, "metrics": result["metrics"]})
            rankings = {
                policy["name"]: self._rank_candidates(candidate_metrics, policy)
                for policy in policies
            }
            pairwise = {
                policy["name"]: self._pairwise_matrix(rankings[policy["name"]])
                for policy in policies
            }
            flip_pairs = self._policy_flip_pairs(pairwise[policies[0]["name"]], pairwise[policies[1]["name"]])
            if len(flip_pairs) < 3:
                raise AssertionError(f"P4 pool {pool_idx} has only {len(flip_pairs)} flip pairs")
            for policy in policies:
                paired = policies[1] if policy is policies[0] else policies[0]
                current_flip_pairs = (
                    flip_pairs
                    if policy is policies[0]
                    else [
                        {
                            "left": pair["left"],
                            "right": pair["right"],
                            "policy_better": pair["paired_policy_better"],
                            "paired_policy_better": pair["policy_better"],
                        }
                        for pair in flip_pairs
                    ]
                )
                task = _base_task(
                    task_id=f"{CIRCUIT_PILOT_VERSION}::P4::{pool_idx:02d}::{policy['name']}",
                    probe="P4",
                    family=base["family"],
                    subtype="policy_conditioned_ranking",
                    spec=base["spec"],
                    design_variables=base["design_variables"],
                    variable_bounds=base["variable_bounds"],
                    constraints=base["constraints"],
                    objective={"name": policy["display_name"], "direction": "maximize", "metric": "policy_score"},
                    query_budget=1,
                    best_known_feasible=base["best_known_feasible"],
                    extra={
                        "candidate_pool": candidate_metrics,
                        "policy": policy,
                        "paired_policy_name": paired["name"],
                        "oracle_reference_ranking": rankings[policy["name"]],
                        "pairwise_matrix": pairwise[policy["name"]],
                        "paired_policy_pairwise_matrix": pairwise[paired["name"]],
                        "policy_flip_pairs": current_flip_pairs,
                    },
                )
                tasks.append(task)
        return tasks

    def _p4_pools(self) -> list[dict[str, Any]]:
        return [
            self._p4_pool("rc_filter", _rc_task_spec(1000.0), [
                ("A", {"R_ohm": 100000.0, "C_f": 1.591549e-9}),
                ("B", {"R_ohm": 10000.0, "C_f": 1.591549e-8}),
                ("C", {"R_ohm": 4700.0, "C_f": 3.386274e-8}),
                ("D", {"R_ohm": 33000.0, "C_f": 4.822877e-9}),
                ("E", {"R_ohm": 68000.0, "C_f": 2.34e-9}),
            ], self._policies("source_current_a", "fc_error_rel", "component_cost", "robustness_margin")),
            self._p4_pool("loaded_divider", _divider_task_spec(2.5), [
                ("A", {"R1_ohm": 10000.0, "R2_ohm": 11111.111111}),
                ("B", {"R1_ohm": 47000.0, "R2_ohm": 88679.245283}),
                ("C", {"R1_ohm": 3300.0, "R2_ohm": 3412.616339}),
                ("D", {"R1_ohm": 18000.0, "R2_ohm": 21951.219512}),
                ("E", {"R1_ohm": 65000.0, "R2_ohm": 185714.285714}),
            ], self._policies("power_w", "vout_error_rel", "component_cost", "robustness_margin")),
            self._p4_pool("op_amp_amplifier", _opamp_task_spec(10.0), [
                ("A", {"Rf_ohm": 90000.0, "Rg_ohm": 10000.0, "Rload_ohm": 10000.0}),
                ("B", {"Rf_ohm": 45000.0, "Rg_ohm": 5000.0, "Rload_ohm": 20000.0}),
                ("C", {"Rf_ohm": 180000.0, "Rg_ohm": 20000.0, "Rload_ohm": 50000.0}),
                ("D", {"Rf_ohm": 81000.0, "Rg_ohm": 9000.0, "Rload_ohm": 15000.0}),
                ("E", {"Rf_ohm": 99000.0, "Rg_ohm": 11000.0, "Rload_ohm": 30000.0}),
            ], self._policies("component_cost", "gain_error_rel", "bandwidth_hz", "robustness_margin", primary_direction="minimize", secondary_direction="minimize", tertiary_direction="maximize")),
            self._p4_pool("linear_regulator", _regulator_task_spec(3.3), [
                ("A", {"vout_v": 3.3, "dropout_v": 0.3, "thermal_resistance_c_per_w": 45.0}),
                ("B", {"vout_v": 3.28, "dropout_v": 0.15, "thermal_resistance_c_per_w": 30.0}),
                ("C", {"vout_v": 3.36, "dropout_v": 0.4, "thermal_resistance_c_per_w": 25.0}),
                ("D", {"vout_v": 3.22, "dropout_v": 0.2, "thermal_resistance_c_per_w": 60.0}),
                ("E", {"vout_v": 3.38, "dropout_v": 0.5, "thermal_resistance_c_per_w": 35.0}),
            ], self._policies("power_loss_w", "vout_error_rel", "component_cost", "robustness_margin")),
            self._p4_pool("led_current_limit", _led_task_spec(0.01), [
                ("A", {"R_ohm": 300.0}),
                ("B", {"R_ohm": 325.0}),
                ("C", {"R_ohm": 278.0}),
                ("D", {"R_ohm": 315.0}),
                ("E", {"R_ohm": 285.0}),
            ], self._led_policies()),
            self._p4_pool("rc_filter", _rc_task_spec(2500.0), [
                ("A", {"R_ohm": 18000.0, "C_f": 3.589828160850528e-9}),
                ("B", {"R_ohm": 33000.0, "C_f": 1.8519847923420549e-9}),
                ("C", {"R_ohm": 62000.0, "C_f": 1.0268060844638409e-9}),
                ("D", {"R_ohm": 12000.0, "C_f": 5.384742241275792e-9}),
                ("E", {"R_ohm": 56000.0, "C_f": 1.119768706753692e-9}),
            ], self._policies("source_current_a", "fc_error_rel", "component_cost", "robustness_margin")),
        ]

    @staticmethod
    def _policies(
        primary_metric: str,
        secondary_metric: str,
        tertiary_metric: str,
        margin_metric: str,
        *,
        primary_direction: str = "minimize",
        secondary_direction: str = "minimize",
        tertiary_direction: str = "minimize",
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": "efficiency_first",
                "display_name": "Efficiency-first",
                "description": f"Prioritize {primary_metric}, then {secondary_metric}.",
                "terms": [
                    {"metric": primary_metric, "direction": primary_direction, "weight": 0.55},
                    {"metric": secondary_metric, "direction": secondary_direction, "weight": 0.25},
                    {"metric": tertiary_metric, "direction": tertiary_direction, "weight": 0.20},
                ],
            },
            {
                "name": "accuracy_first",
                "display_name": "Accuracy-first",
                "description": f"Prioritize {secondary_metric}, then robustness margin, then cost/current.",
                "terms": [
                    {"metric": secondary_metric, "direction": secondary_direction, "weight": 0.55},
                    {"metric": margin_metric, "direction": "maximize", "weight": 0.25},
                    {"metric": tertiary_metric, "direction": tertiary_direction, "weight": 0.20},
                ],
            },
        ]

    @staticmethod
    def _led_policies() -> list[dict[str, Any]]:
        return [
            {
                "name": "safety_first",
                "display_name": "Safety-first",
                "description": "Prioritize low resistor power, then component cost.",
                "terms": [
                    {"metric": "resistor_power_w", "direction": "minimize", "weight": 0.90},
                    {"metric": "component_cost", "direction": "minimize", "weight": 0.10},
                ],
            },
            {
                "name": "brightness_accuracy_first",
                "display_name": "Brightness-accuracy-first",
                "description": "Prioritize current accuracy, then power margin.",
                "terms": [
                    {"metric": "led_current_error_rel", "direction": "minimize", "weight": 0.90},
                    {"metric": "robustness_margin", "direction": "maximize", "weight": 0.10},
                ],
            },
        ]

    @staticmethod
    def _p4_pool(family: str, parts: tuple[Any, ...], candidates: list[tuple[str, dict[str, float]]], policies: list[dict[str, Any]]) -> dict[str, Any]:
        spec, variables, bounds, constraints, objective = parts
        base = _base_task(
            task_id=f"{CIRCUIT_PILOT_VERSION}::P4::pool::{family}",
            probe="P4",
            family=family,
            subtype="p4_pool_base",
            spec=spec,
            design_variables=variables,
            variable_bounds=bounds,
            constraints=constraints,
            objective=objective,
            query_budget=1,
            best_known_feasible=candidates[0][1],
        )
        return {
            "base": base,
            "candidates": [{"candidate_id": cid, "design": design} for cid, design in candidates],
            "policies": policies,
        }

    @staticmethod
    def _rank_candidates(candidates: list[dict[str, Any]], policy: dict[str, Any]) -> list[str]:
        scores = CircuitPilotBuilder._policy_scores(candidates, policy)
        return [cid for cid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]

    @staticmethod
    def _policy_scores(candidates: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, float]:
        scores = {candidate["candidate_id"]: 0.0 for candidate in candidates}
        for term in policy["terms"]:
            metric = term["metric"]
            values = [float(candidate["metrics"].get(metric, 0.0)) for candidate in candidates]
            lo = min(values)
            hi = max(values)
            span = max(hi - lo, 1e-12)
            for candidate, value in zip(candidates, values):
                normalized = (value - lo) / span
                if term["direction"] == "minimize":
                    normalized = 1.0 - normalized
                scores[candidate["candidate_id"]] += float(term["weight"]) * normalized
        return {cid: round(score, 12) for cid, score in scores.items()}

    @staticmethod
    def _pairwise_matrix(ranking: list[str]) -> dict[str, str]:
        rank = {cid: idx for idx, cid in enumerate(ranking)}
        matrix: dict[str, str] = {}
        ids = list(ranking)
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1 :]:
                better, worse = (left, right) if rank[left] < rank[right] else (right, left)
                matrix[f"{left}>{right}"] = better
                matrix[f"{right}>{left}"] = better
                matrix[f"{better}|{worse}"] = better
        return matrix

    @staticmethod
    def _policy_flip_pairs(left: dict[str, str], right: dict[str, str]) -> list[dict[str, str]]:
        pairs: list[dict[str, str]] = []
        seen: set[frozenset[str]] = set()
        for key, better_left in left.items():
            if "|" not in key:
                continue
            a, b = key.split("|", 1)
            pair_key = frozenset({a, b})
            if pair_key in seen:
                continue
            seen.add(pair_key)
            better_right = right.get(f"{a}|{b}") or right.get(f"{b}|{a}")
            if better_right and better_right != better_left:
                pairs.append({"left": a, "right": b, "policy_better": better_left, "paired_policy_better": better_right})
        return pairs

    def write_audit_bundle(self, *, task: dict[str, Any], audit_root: Path) -> None:
        task_dir = audit_root / task["probe"].lower() / task["task_id"].replace("::", "__")
        task_dir.mkdir(parents=True, exist_ok=True)
        _write_json(task_dir / "task.json", task)
        if task["probe"] in {"P2", "P3"}:
            initial = task.get("initial_design") or task.get("corrupted_design")
            trace = self.oracle.evaluate(task, initial).to_dict() if isinstance(initial, dict) else {}
            expected = self.oracle.evaluate(task, task["best_known_feasible"]).to_dict() if isinstance(task.get("best_known_feasible"), dict) else {}
        elif task["probe"] == "P4":
            trace = {"candidate_count": len(task["candidate_pool"]), "policy_flip_pairs": task["policy_flip_pairs"]}
            expected = {"oracle_reference_ranking": task["oracle_reference_ranking"]}
            _write_json(task_dir / "candidate_metrics.json", task["candidate_pool"])
            _write_json(task_dir / "policy_scores.json", {"ranking": task["oracle_reference_ranking"], "policy": task["policy"]})
            _write_json(task_dir / "pairwise_matrix_current.json", task["pairwise_matrix"])
            _write_json(task_dir / "pairwise_matrix_paired.json", task["paired_policy_pairwise_matrix"])
            _write_json(task_dir / "policy_flip_pairs.json", task["policy_flip_pairs"])
        else:
            trace = self.oracle.proof_for_task(task)
            expected = {"gold_label": task.get("gold_label"), "reference_design": task.get("reference_design")}
        _write_json(task_dir / "oracle_trace.json", trace)
        _write_json(task_dir / "oracle_expected.json", expected)
        _write_json(task_dir / "score_trace.json", {"status": "not_scored", "note": "Filled by evaluator for model outputs."})
        (task_dir / "audit.md").write_text(self._audit_markdown(task, trace, expected) + "\n")

    @staticmethod
    def _audit_markdown(task: dict[str, Any], trace: dict[str, Any], expected: dict[str, Any]) -> str:
        return (
            f"# {task['task_id']}\n\n"
            f"- Probe: {task['probe']}\n"
            f"- Family: {task['family']}\n"
            f"- Subtype: {task['subtype']}\n"
            f"- Purpose: cross-domain construct-validity audit, not a standalone circuit benchmark.\n\n"
            "## Expected\n\n"
            f"```json\n{json.dumps(expected, indent=2, sort_keys=True)}\n```\n\n"
            "## Oracle Trace\n\n"
            f"```json\n{json.dumps(trace, indent=2, sort_keys=True)}\n```\n"
        )

    def write_scripted_oracle_results(self, *, tasks_by_probe: dict[str, list[dict[str, Any]]], out_dir: Path, overwrite: bool = False) -> None:
        if out_dir.exists() and not overwrite:
            raise FileExistsError(f"Scripted output directory exists: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_dir / "p1_results.jsonl", [self._scripted_p1_row(task) for task in tasks_by_probe["P1"]])
        _write_jsonl(out_dir / "p2_results.jsonl", [self._scripted_repair_row(task) for task in tasks_by_probe["P2"]])
        _write_jsonl(out_dir / "p3_results.jsonl", [self._scripted_p3_row(task) for task in tasks_by_probe["P3"]])
        _write_jsonl(out_dir / "p4_results.jsonl", [self._scripted_p4_row(task) for task in tasks_by_probe["P4"]])
        _write_json(out_dir / "run_manifest.json", {"runner_name": "scripted_oracle", "domain": DOMAIN, "pilot_version": CIRCUIT_PILOT_VERSION})

    @staticmethod
    def _scripted_p1_row(task: dict[str, Any]) -> dict[str, Any]:
        action = dict(task["gold_label"])
        if action["action_type"] == "propose_design":
            action["candidate"] = task.get("reference_design") or task.get("best_known_feasible")
        return {"task_id": task["task_id"], "runner_name": "scripted_oracle", "parsed_action": action}

    @staticmethod
    def _scripted_repair_row(task: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": task["task_id"],
            "runner_name": "scripted_oracle",
            "steps": [
                {
                    "action_type": "propose_design",
                    "candidate": task["best_known_feasible"],
                    "reason": "scripted oracle repair",
                }
            ],
        }

    @staticmethod
    def _scripted_p3_row(task: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": task["task_id"],
            "runner_name": "scripted_oracle",
            "steps": [
                {
                    "action_type": "replan",
                    "reason": "discard corrupted trajectory and recompute from verifier state",
                    "suggested_pivot": "reset_history",
                },
                {
                    "action_type": "propose_design",
                    "candidate": task["best_known_feasible"],
                    "reason": "scripted oracle recovery",
                },
            ],
        }

    @staticmethod
    def _scripted_p4_row(task: dict[str, Any]) -> dict[str, Any]:
        return {"task_id": task["task_id"], "runner_name": "scripted_oracle", "ranking": task["oracle_reference_ranking"], "confidence": 1.0}
