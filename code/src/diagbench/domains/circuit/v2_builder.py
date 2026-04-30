"""Harder v2 circuit pilot with tighter boundaries and coupled constraints."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from diagbench.domains.circuit.builder import (
    DOMAIN,
    CircuitPilotBuilder,
    _divider_task_spec,
    _led_task_spec,
    _lower,
    _opamp_task_spec,
    _rc_task_spec,
    _regulator_task_spec,
    _target_log,
    _target_rel,
    _task_hash,
    _upper,
    _write_json,
    _write_jsonl,
)


CIRCUIT_PILOT_V2_VERSION = "circuit_pilot_v2"
TASK_COUNTS_V2 = {"P1": 16, "P2": 16, "P3": 18, "P4": 24}


def _base_task_v2(
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
        "pilot_version": CIRCUIT_PILOT_V2_VERSION,
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
            "oracle_version": "circuit_oracle_v0.2",
            "unit_system": "SI",
        },
    }
    if extra:
        task.update(extra)
    task["task_sha256"] = _task_hash({key: value for key, value in task.items() if key != "task_sha256"})
    return task


def _manifest_v2(tasks: list[dict[str, Any]], *, seed: int, probe: str, artifact_path: Path) -> dict[str, Any]:
    return {
        "domain": DOMAIN,
        "pilot_version": CIRCUIT_PILOT_V2_VERSION,
        "probe": probe,
        "n_tasks": len(tasks),
        "seed": seed,
        "artifact_path": str(artifact_path),
        "artifact_sha256": _task_hash({"tasks": tasks}),
        "task_ids": [task["task_id"] for task in tasks],
    }


def _set_tolerance(constraints: list[dict[str, Any]], *, tolerance_rel: float) -> list[dict[str, Any]]:
    updated = copy.deepcopy(constraints)
    for constraint in updated:
        if "tolerance_rel" in constraint:
            constraint["tolerance_rel"] = tolerance_rel
    return updated


class CircuitPilotV2Builder(CircuitPilotBuilder):
    """Build circuit_pilot_v2 with higher discrimination pressure."""

    def build(self) -> dict[str, list[dict[str, Any]]]:
        tasks = {
            "P1": self.build_p1_tasks(),
            "P2": self.build_p2_tasks(),
            "P3": self.build_p3_tasks(),
            "P4": self.build_p4_tasks(),
        }
        for probe, expected in TASK_COUNTS_V2.items():
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
            _write_json(out_dir / f"{probe.lower()}_manifest.json", _manifest_v2(tasks, seed=self.seed, probe=probe, artifact_path=task_path))
            for task in tasks:
                self.write_audit_bundle(task=task, audit_root=audit_dir)
        _write_json(out_dir / "dataset_summary.json", self.dataset_summary(tasks_by_probe))
        return tasks_by_probe

    def dataset_summary(self, tasks_by_probe: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return {
            "domain": DOMAIN,
            "pilot_version": CIRCUIT_PILOT_V2_VERSION,
            "seed": self.seed,
            "task_counts": {probe: len(tasks) for probe, tasks in tasks_by_probe.items()},
            "families": sorted({task["family"] for tasks in tasks_by_probe.values() for task in tasks}),
            "hardening_mechanisms": [
                "near-boundary P1 infeasibility",
                "P2 dual-constraint coupling",
                "P3 progressive dual traps",
                "P4 reweighted near-tie ranking variants",
            ],
        }

    def build_p1_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        propose_specs = [
            ("rc_filter", _rc_task_spec(1000.0), {"R_ohm": 15915.494309, "C_f": 1e-8}),
            ("loaded_divider", _divider_task_spec(2.5, load_ohm=100000.0), {"R1_ohm": 10000.0, "R2_ohm": 11111.111111}),
            ("led_current_limit", _led_task_spec(0.01), {"R_ohm": 300.0}),
            ("op_amp_amplifier", _opamp_task_spec(20.0, mode="inverting"), {"Rf_ohm": 200000.0, "Rin_ohm": 10000.0, "Rload_ohm": 20000.0}),
        ]
        for idx, (family, parts, ref) in enumerate(propose_specs):
            spec, variables, bounds, constraints, objective = parts
            tasks.append(
                _base_task_v2(
                    task_id=f"{CIRCUIT_PILOT_V2_VERSION}::P1::propose_design::{idx:02d}",
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
                            "oracle_version": "circuit_oracle_v0.2",
                            "unit_system": "SI",
                            "proof": {"proof_type": "reference_design_feasible", "reference_design": ref},
                        },
                    },
                )
            )

        infeasible_cases = [
            self._p1_led_near_infeasible(),
            self._p1_rc_near_infeasible(),
            self._p1_opamp_gbw_infeasible(),
            self._p1_loaded_divider_load_infeasible(),
        ]
        for idx, case in enumerate(infeasible_cases):
            tasks.append(
                _base_task_v2(
                    task_id=f"{CIRCUIT_PILOT_V2_VERSION}::P1::declare_infeasible::{idx:02d}",
                    probe="P1",
                    family=case["family"],
                    subtype=case["subtype"],
                    spec=case["spec"],
                    design_variables=case["variables"],
                    variable_bounds=case["bounds"],
                    constraints=case["constraints"],
                    objective=case["objective"],
                    query_budget=1,
                    best_known_feasible=None,
                    extra={
                        "gold_label": {"action_type": "declare_infeasible", "reason": case["proof"]},
                        "oracle_metadata": {
                            "oracle": "closed_form_circuit_oracle",
                            "oracle_version": "circuit_oracle_v0.2",
                            "unit_system": "SI",
                            "proof": {
                                "proof_type": "near_boundary_infeasibility",
                                "argument": case["proof"],
                                "margin_ratio": case.get("margin_ratio"),
                            },
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
            spec = copy.deepcopy(spec)
            for field in missing:
                _, key = field.split(".", 1)
                spec.pop(key, None)
            tasks.append(
                _base_task_v2(
                    task_id=f"{CIRCUIT_PILOT_V2_VERSION}::P1::request_missing_info::{idx:02d}",
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
                            "oracle_version": "circuit_oracle_v0.2",
                            "unit_system": "SI",
                            "proof": {"proof_type": "missing_blocker", "missing_fields": missing, "argument": reason},
                        },
                    },
                )
            )

        narrow_cases = [
            ("rc_filter", _rc_task_spec(1000.0), {"R_ohm": 15915.494309, "C_f": 1e-8}),
            ("op_amp_amplifier", _opamp_task_spec(20.0, mode="inverting"), {"Rf_ohm": 200000.0, "Rin_ohm": 10000.0, "Rload_ohm": 20000.0}),
            ("loaded_divider", _divider_task_spec(1.25, load_ohm=10000.0), {"R1_ohm": 15000.0, "R2_ohm": 10000.0}),
            ("linear_regulator", _regulator_task_spec(4.2), {"vout_v": 4.2, "dropout_v": 0.3, "thermal_resistance_c_per_w": 35.0}),
        ]
        for idx, (family, parts, ref) in enumerate(narrow_cases):
            spec, variables, bounds, constraints, objective = parts
            constraints = _set_tolerance(constraints, tolerance_rel=0.01)
            tasks.append(
                _base_task_v2(
                    task_id=f"{CIRCUIT_PILOT_V2_VERSION}::P1::feasible_narrow::{idx:02d}",
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
                        "hardening": {"tolerance_rel": 0.01, "risk": "conservative refusal on narrow feasible boundary"},
                        "oracle_metadata": {
                            "oracle": "closed_form_circuit_oracle",
                            "oracle_version": "circuit_oracle_v0.2",
                            "unit_system": "SI",
                            "proof": {"proof_type": "narrow_reference_design_feasible", "reference_design": ref},
                        },
                    },
                )
            )
        return tasks

    def _p1_led_near_infeasible(self) -> dict[str, Any]:
        spec, variables, bounds, _, objective = _led_task_spec(0.1)
        spec = {**spec, "supply_v": 5.0, "led_vf_v": 2.0, "target_current_a": 0.1}
        bounds = {"R_ohm": {"min": 33.0, "max": 2000.0, "unit": "ohm"}}
        constraints = [_target_rel("led_current", "led_current_a", 0.1, 0.05, "A"), _upper("resistor_power", "resistor_power_w", 0.25, "W", scale=0.25)]
        return {
            "family": "led_current_limit",
            "subtype": "declare_infeasible_near_margin",
            "spec": spec,
            "variables": variables,
            "bounds": bounds,
            "constraints": constraints,
            "objective": objective,
            "proof": "At R_min=33 ohm, I=(5-2)/33=90.9mA, below the 95mA lower edge of a 100mA +/-5% target.",
            "margin_ratio": 0.1 / ((5.0 - 2.0) / 33.0),
        }

    def _p1_rc_near_infeasible(self) -> dict[str, Any]:
        spec, variables, bounds, constraints, objective = _rc_task_spec(1000.0)
        bounds = {"R_ohm": {"min": 1000.0, "max": 10000.0, "unit": "ohm"}, "C_f": {"min": 1e-9, "max": 1e-8, "unit": "F"}}
        constraints = [_target_log("cutoff_frequency", "fc_hz", 1000.0, 0.05, "Hz"), _upper("source_current", "source_current_a", 0.005, "A", scale=0.005)]
        return {
            "family": "rc_filter",
            "subtype": "declare_infeasible_near_margin",
            "spec": spec,
            "variables": variables,
            "bounds": bounds,
            "constraints": constraints,
            "objective": objective,
            "proof": "With R<=10k and C<=10nF, fc_min=1/(2*pi*10k*10nF)=1.59kHz, outside a 1kHz +/-5% target.",
            "margin_ratio": 1591.5494309 / 1050.0,
        }

    def _p1_opamp_gbw_infeasible(self) -> dict[str, Any]:
        spec, variables, bounds, constraints, objective = _opamp_task_spec(50.0, mode="inverting")
        spec = {**spec, "gbw_hz": 5_000_000.0, "min_bandwidth_hz": 200_000.0}
        constraints = [_target_log("closed_loop_gain", "gain_v_per_v", 50.0, 0.05, "V/V"), _lower("minimum_bandwidth", "bandwidth_hz", 200_000.0, "Hz", scale=200_000.0), _lower("input_impedance", "input_impedance_ohm", 8000.0, "ohm", scale=8000.0)]
        return {
            "family": "op_amp_amplifier",
            "subtype": "declare_infeasible_impossible_objective",
            "spec": spec,
            "variables": variables,
            "bounds": bounds,
            "constraints": constraints,
            "objective": objective,
            "proof": "GBW/gain at gain 50 is 100kHz, but the minimum bandwidth requirement is 200kHz.",
            "margin_ratio": 2.0,
        }

    def _p1_loaded_divider_load_infeasible(self) -> dict[str, Any]:
        spec, variables, _, _, objective = _divider_task_spec(1.65, load_ohm=50.0)
        spec = {**spec, "vin_v": 3.3, "target_vout_v": 1.65, "load_ohm": 50.0}
        bounds = {"R1_ohm": {"min": 100.0, "max": 1000.0, "unit": "ohm"}, "R2_ohm": {"min": 100.0, "max": 1000.0, "unit": "ohm"}}
        constraints = [_target_rel("output_voltage", "vout_v", 1.65, 0.05, "V"), _upper("divider_current", "divider_current_a", 0.05, "A", scale=0.05)]
        return {
            "family": "loaded_divider",
            "subtype": "declare_infeasible_impossible_objective",
            "spec": spec,
            "variables": variables,
            "bounds": bounds,
            "constraints": constraints,
            "objective": objective,
            "proof": "With R1>=100 ohm and R2||RL <= 47.6 ohm, Vout_max=3.3*47.6/(100+47.6)=1.06V, below 1.65V +/-5%.",
            "margin_ratio": 1.65 / 1.0645,
        }

    def build_p2_tasks(self) -> list[dict[str, Any]]:
        variants = self._p2_rc_variants() + self._p2_divider_variants() + self._p2_opamp_variants() + self._p2_power_variants()
        tasks: list[dict[str, Any]] = []
        for idx, item in enumerate(variants):
            spec, variables, bounds, constraints, objective = item["parts"]
            task = _base_task_v2(
                task_id=f"{CIRCUIT_PILOT_V2_VERSION}::P2::{idx:02d}",
                probe="P2",
                family=item["family"],
                subtype=item.get("subtype", "dual_constraint_repair"),
                spec=spec,
                design_variables=variables,
                variable_bounds=bounds,
                constraints=constraints,
                objective=objective,
                query_budget=4,
                best_known_feasible=item["best"],
                extra={
                    "initial_design": item["initial"],
                    "allowed_edits": variables,
                    "hardening": item["hardening"],
                },
            )
            task["initial_oracle_result"] = self.oracle.evaluate(task, item["initial"]).to_dict()
            task["oracle_reference_result"] = self.oracle.evaluate(task, item["best"]).to_dict()
            if task["oracle_reference_result"]["feasible"] is not True:
                raise AssertionError(f"P2 best design is not feasible: {task['task_id']}")
            tasks.append(task)
        return tasks

    def _p2_rc_variants(self) -> list[dict[str, Any]]:
        configs = [
            (1000.0, 0.0005, {"R_ohm": 5000.0, "C_f": 1e-8}, {"R_ohm": 10000.0, "C_f": 1.5915494309e-8}),
            (1500.0, 0.0004, {"R_ohm": 5000.0, "C_f": 5e-9}, {"R_ohm": 15000.0, "C_f": 7.0735530263e-9}),
            (400.0, 0.00025, {"R_ohm": 5000.0, "C_f": 1e-8}, {"R_ohm": 33000.0, "C_f": 1.2056813870e-8}),
            (2500.0, 0.0008, {"R_ohm": 2000.0, "C_f": 1e-8}, {"R_ohm": 12000.0, "C_f": 5.3051647697e-9}),
        ]
        out: list[dict[str, Any]] = []
        for target, current_limit, initial, best in configs:
            spec, variables, bounds, _, objective = _rc_task_spec(target)
            constraints = [_target_log("cutoff_frequency", "fc_hz", target, 0.02, "Hz"), _upper("source_current", "source_current_a", current_limit, "A", scale=current_limit)]
            out.append({"family": "rc_filter", "parts": (spec, variables, bounds, constraints, objective), "initial": initial, "best": best, "hardening": {"mechanism": "dual_constraint_coupling", "coupled_metrics": ["fc_hz", "source_current_a"]}})
        return out

    def _p2_divider_variants(self) -> list[dict[str, Any]]:
        configs = [
            (2.5, 100000.0, 0.0005, {"R1_ohm": 500.0, "R2_ohm": 500.0}, {"R1_ohm": 20000.0, "R2_ohm": 25000.0}),
            (1.8, 47000.0, 0.0004, {"R1_ohm": 1000.0, "R2_ohm": 1000.0}, {"R1_ohm": 30000.0, "R2_ohm": 26315.789474}),
            (3.3, 100000.0, 0.0004, {"R1_ohm": 20000.0, "R2_ohm": 10000.0}, {"R1_ohm": 10000.0, "R2_ohm": 24096.385542}),
            (1.2, 10000.0, 0.0003, {"R1_ohm": 10000.0, "R2_ohm": 10000.0}, {"R1_ohm": 20000.0, "R2_ohm": 17142.857143}),
        ]
        out: list[dict[str, Any]] = []
        for target, load, current_limit, initial, best in configs:
            spec, variables, bounds, _, objective = _divider_task_spec(target, load_ohm=load)
            constraints = [_target_rel("output_voltage", "vout_v", target, 0.02, "V"), _upper("divider_current", "divider_current_a", current_limit, "A", scale=current_limit)]
            out.append({"family": "loaded_divider", "parts": (spec, variables, bounds, constraints, objective), "initial": initial, "best": best, "hardening": {"mechanism": "loaded_divider_current_coupling", "coupled_metrics": ["vout_v", "divider_current_a"]}})
        return out

    def _p2_opamp_variants(self) -> list[dict[str, Any]]:
        configs = [
            (20.0, 5_000_000.0, 200_000.0, 8000.0, {"Rf_ohm": 100000.0, "Rin_ohm": 2000.0, "Rload_ohm": 1000.0}, {"Rf_ohm": 160000.0, "Rin_ohm": 8000.0, "Rload_ohm": 20000.0}),
            (10.0, 2_500_000.0, 200_000.0, 10000.0, {"Rf_ohm": 100000.0, "Rin_ohm": 2000.0, "Rload_ohm": 1000.0}, {"Rf_ohm": 100000.0, "Rin_ohm": 10000.0, "Rload_ohm": 15000.0}),
            (15.0, 4_000_000.0, 220_000.0, 9000.0, {"Rf_ohm": 180000.0, "Rin_ohm": 3000.0, "Rload_ohm": 1000.0}, {"Rf_ohm": 150000.0, "Rin_ohm": 10000.0, "Rload_ohm": 20000.0}),
            (8.0, 2_000_000.0, 220_000.0, 12000.0, {"Rf_ohm": 160000.0, "Rin_ohm": 4000.0, "Rload_ohm": 1000.0}, {"Rf_ohm": 96000.0, "Rin_ohm": 12000.0, "Rload_ohm": 15000.0}),
        ]
        out: list[dict[str, Any]] = []
        for gain, gbw, min_bw, min_zin, initial, best in configs:
            spec, variables, bounds, _, objective = _opamp_task_spec(gain, mode="inverting")
            spec = {**spec, "gbw_hz": gbw, "min_bandwidth_hz": min_bw}
            constraints = [_target_log("closed_loop_gain", "gain_v_per_v", gain, 0.03, "V/V"), _lower("minimum_bandwidth", "bandwidth_hz", min_bw, "Hz", scale=min_bw), _lower("input_impedance", "input_impedance_ohm", min_zin, "ohm", scale=min_zin)]
            out.append({"family": "op_amp_amplifier", "parts": (spec, variables, bounds, constraints, objective), "initial": initial, "best": best, "hardening": {"mechanism": "gain_bandwidth_input_impedance_coupling", "coupled_metrics": ["gain_v_per_v", "bandwidth_hz", "input_impedance_ohm"]}})
        return out

    def _p2_power_variants(self) -> list[dict[str, Any]]:
        configs = [
            ("linear_regulator", _regulator_task_spec(3.3), {"vin_v": 5.0, "target_vout_v": 3.3, "load_current_a": 0.45, "ambient_c": 25.0}, {"vout_v": 4.5, "dropout_v": 0.8, "thermal_resistance_c_per_w": 120.0}, {"vout_v": 3.3, "dropout_v": 0.3, "thermal_resistance_c_per_w": 40.0}, ["vout_v", "dropout_margin_v", "junction_temp_c"]),
            ("linear_regulator", _regulator_task_spec(2.5), {"vin_v": 5.0, "target_vout_v": 2.5, "load_current_a": 0.8, "ambient_c": 25.0}, {"vout_v": 4.2, "dropout_v": 0.8, "thermal_resistance_c_per_w": 120.0}, {"vout_v": 2.5, "dropout_v": 0.2, "thermal_resistance_c_per_w": 25.0}, ["vout_v", "dropout_margin_v", "junction_temp_c"]),
            ("led_current_limit", _led_task_spec(0.02), {"supply_v": 5.0, "led_vf_v": 2.0, "target_current_a": 0.02, "resistor_power_rating_w": 0.125}, {"R_ohm": 50.0}, {"R_ohm": 150.0}, ["led_current_a", "resistor_power_w"]),
            ("led_current_limit", _led_task_spec(0.012), {"supply_v": 5.0, "led_vf_v": 2.1, "target_current_a": 0.012, "resistor_power_rating_w": 0.1}, {"R_ohm": 1000.0}, {"R_ohm": 241.666667}, ["led_current_a", "resistor_power_w"]),
        ]
        out: list[dict[str, Any]] = []
        for family, parts, spec_override, initial, best, metrics in configs:
            spec, variables, bounds, constraints, objective = parts
            spec = {**spec, **spec_override}
            if family == "led_current_limit":
                constraints = [_target_rel("led_current", "led_current_a", spec["target_current_a"], 0.04, "A"), _upper("resistor_power", "resistor_power_w", spec["resistor_power_rating_w"], "W", scale=spec["resistor_power_rating_w"])]
            else:
                constraints = [_target_rel("output_voltage", "vout_v", spec["target_vout_v"], 0.02, "V"), _lower("dropout_margin", "dropout_margin_v", 0.3, "V", scale=1.0), _upper("junction_temp", "junction_temp_c", 85.0, "C", scale=85.0)]
            out.append({"family": family, "parts": (spec, variables, bounds, constraints, objective), "initial": initial, "best": best, "hardening": {"mechanism": "power_bias_coupling", "coupled_metrics": metrics}})
        return out

    def build_p3_tasks(self) -> list[dict[str, Any]]:
        p2_sources = self.build_p2_tasks()
        base_types = ["unit_flip", "wrong_formula_direction", "false_feasibility", "topology_trap", "verifier_ignored", "progressive_contamination"]
        tasks: list[dict[str, Any]] = []
        for idx, corruption in enumerate(base_types * 2):
            source = p2_sources[(idx * 3) % len(p2_sources)]
            tasks.append(self._p3_from_source(source=source, corruption=corruption, idx=idx))
        dual_sources = []
        for family in ("rc_filter", "loaded_divider", "op_amp_amplifier"):
            dual_sources.extend([task for task in p2_sources if task["family"] == family][:2])
        for jdx, source in enumerate(dual_sources):
            tasks.append(self._p3_from_source(source=source, corruption="progressive_dual_trap", idx=jdx + 12))
        return tasks

    def _p3_from_source(self, *, source: dict[str, Any], corruption: str, idx: int) -> dict[str, Any]:
        bad = dict(source["initial_design"])
        best = dict(source["best_known_feasible"])
        history = self._corrupted_history_v2(corruption=corruption, task=source, bad=bad)
        extra = {
            "corruption_type": corruption,
            "corrupted_design": bad,
            "corrupted_history": history,
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
        }
        if corruption == "progressive_dual_trap":
            extra["dual_trap_metadata"] = self._dual_trap_metadata(source)
        return _base_task_v2(
            task_id=f"{CIRCUIT_PILOT_V2_VERSION}::P3::{corruption}::{idx:02d}",
            probe="P3",
            family=source["family"],
            subtype=corruption,
            spec=source["spec"],
            design_variables=source["design_variables"],
            variable_bounds=source["variable_bounds"],
            constraints=source["constraints"],
            objective=source["objective"],
            query_budget=5,
            best_known_feasible=best,
            extra=extra,
        )

    def _corrupted_history_v2(self, *, corruption: str, task: dict[str, Any], bad: dict[str, float]) -> list[dict[str, Any]]:
        result = self.oracle.evaluate(task, bad).to_dict()
        text = {
            "unit_flip": "Previous analysis treated nF as uF and trusted the wrong magnitude.",
            "wrong_formula_direction": "Previous analysis used the inverse repair direction for the dominant formula.",
            "false_feasibility": "Previous analysis declared feasible without recomputing all coupled constraints.",
            "topology_trap": "Previous analysis stayed inside the wrong topology/parameter regime despite verifier feedback.",
            "verifier_ignored": "Previous analysis treated verifier failure as advisory and kept optimizing the same direction.",
            "progressive_contamination": "A plausible early step introduced a small unit/sign error and later steps trusted it.",
            "progressive_dual_trap": "The history escapes the first violation by moving in a direction that creates or preserves a second coupled violation.",
        }[corruption]
        history = [
            {"step": 0, "action_type": "propose_design", "candidate": bad, "analysis_summary": text, "verifier_response": result},
        ]
        if corruption == "progressive_dual_trap":
            trap = self._trap_candidate(task, bad)
            history.append(
                {
                    "step": 1,
                    "action_type": "propose_design",
                    "candidate": trap,
                    "analysis_summary": "Partial escape: reduce the most visible violation while leaving the coupled constraint unresolved.",
                    "verifier_response": self.oracle.evaluate(task, trap).to_dict(),
                }
            )
        else:
            history.append({"step": 1, "action_type": "propose_design", "candidate": bad, "analysis_summary": "Continue the same repair direction based on the prior trajectory.", "verifier_response": result})
        return history

    @staticmethod
    def _trap_candidate(task: dict[str, Any], bad: dict[str, float]) -> dict[str, float]:
        family = task["family"]
        best = dict(task["best_known_feasible"])
        if family == "rc_filter":
            return {**best, "R_ohm": bad["R_ohm"]}
        if family == "loaded_divider":
            return {**best, "R1_ohm": bad["R1_ohm"]}
        if family == "op_amp_amplifier":
            return {**best, "Rin_ohm": bad.get("Rin_ohm", best.get("Rin_ohm", 1000.0))}
        return dict(bad)

    @staticmethod
    def _dual_trap_metadata(source: dict[str, Any]) -> dict[str, Any]:
        return {
            "mechanism": "escape_then_cascade",
            "phase1": "repair the visible dominant violation from corrupted history",
            "phase2": "the easy repair direction preserves or introduces a coupled violation",
            "coupled_constraints": [constraint["name"] for constraint in source["constraints"]],
        }

    def build_p4_tasks(self) -> list[dict[str, Any]]:
        base_tasks = CircuitPilotBuilder(seed=self.seed).build_p4_tasks()
        converted = [self._convert_p4_task(task, hard=False, idx=idx) for idx, task in enumerate(base_tasks)]
        harder = [self._convert_p4_task(task, hard=True, idx=idx) for idx, task in enumerate(base_tasks)]
        return converted + harder

    def _convert_p4_task(self, task: dict[str, Any], *, hard: bool, idx: int) -> dict[str, Any]:
        out = copy.deepcopy(task)
        suffix = "::harder" if hard else ""
        out["task_id"] = task["task_id"].replace("circuit_pilot_v1", CIRCUIT_PILOT_V2_VERSION) + suffix
        out["pilot_version"] = CIRCUIT_PILOT_V2_VERSION
        out["oracle_metadata"]["oracle_version"] = "circuit_oracle_v0.2"
        if hard:
            out["subtype"] = "policy_conditioned_ranking_harder_near_tie"
            out["policy"] = self._harder_policy(
                out["policy"],
                candidates=out["candidate_pool"],
                paired_matrix=out["paired_policy_pairwise_matrix"],
            )
            out["objective"] = {"name": out["policy"]["display_name"], "direction": "maximize", "metric": "policy_score"}
            ranking = self._rank_candidates(out["candidate_pool"], out["policy"])
            out["oracle_reference_ranking"] = ranking
            out["pairwise_matrix"] = self._pairwise_matrix(ranking)
            out["policy_flip_pairs"] = self._policy_flip_pairs(out["pairwise_matrix"], out["paired_policy_pairwise_matrix"])
            scores = self._policy_scores(out["candidate_pool"], out["policy"])
            ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            out["hardening"] = {
                "mechanism": "reweighted_policy_near_tie",
                "top2_score_gap": round(ordered[0][1] - ordered[1][1], 6) if len(ordered) > 1 else None,
                "top2_pair": [ordered[0][0], ordered[1][0]] if len(ordered) > 1 else [],
            }
        out["task_sha256"] = _task_hash({key: value for key, value in out.items() if key != "task_sha256"})
        return out

    @staticmethod
    def _harder_policy(
        policy: dict[str, Any],
        *,
        candidates: list[dict[str, Any]] | None = None,
        paired_matrix: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        hard = copy.deepcopy(policy)
        hard["name"] = f"{hard.get('name', 'policy')}_harder"
        hard["display_name"] = f"{hard.get('display_name', 'Policy')} harder near-tie"
        hard["description"] = f"{hard.get('description', '')} Reweighted to reduce dominant-term shortcuts."
        terms = hard.get("terms", [])
        if candidates and paired_matrix:
            candidate = CircuitPilotV2Builder._select_near_tie_weights(hard, candidates, paired_matrix)
            if candidate is not None:
                for term, weight in zip(terms, candidate["weights"]):
                    term["weight"] = weight
                hard["near_tie_search"] = {
                    "top2_score_gap": candidate["gap"],
                    "policy_flip_pairs": candidate["flip_pairs"],
                }
                return hard
        if len(terms) >= 3:
            terms[0]["weight"] = 0.45
            terms[1]["weight"] = 0.35
            terms[2]["weight"] = 0.20
        elif len(terms) == 2:
            terms[0]["weight"] = 0.55
            terms[1]["weight"] = 0.45
        return hard

    @staticmethod
    def _select_near_tie_weights(
        policy: dict[str, Any],
        candidates: list[dict[str, Any]],
        paired_matrix: dict[str, str],
    ) -> dict[str, Any] | None:
        terms = policy.get("terms", [])
        if not terms:
            return None
        weight_sets: list[tuple[float, ...]] = []
        if len(terms) == 2:
            weight_sets = [(round(w / 20.0, 2), round(1.0 - w / 20.0, 2)) for w in range(3, 18)]
        elif len(terms) >= 3:
            for a in range(2, 17):
                for b in range(2, 19 - a):
                    c = 20 - a - b
                    if c >= 2:
                        weight_sets.append((round(a / 20.0, 2), round(b / 20.0, 2), round(c / 20.0, 2)))
        best: dict[str, Any] | None = None
        for weights in weight_sets:
            trial = copy.deepcopy(policy)
            for term, weight in zip(trial["terms"], weights):
                term["weight"] = weight
            ranking = CircuitPilotBuilder._rank_candidates(candidates, trial)
            current_matrix = CircuitPilotBuilder._pairwise_matrix(ranking)
            flip_pairs = CircuitPilotBuilder._policy_flip_pairs(current_matrix, paired_matrix)
            if len(flip_pairs) < 3:
                continue
            scores = CircuitPilotBuilder._policy_scores(candidates, trial)
            ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            if len(ordered) < 2:
                continue
            gap = round(float(ordered[0][1] - ordered[1][1]), 12)
            candidate = {"weights": weights, "gap": gap, "flip_pairs": len(flip_pairs)}
            if best is None or (candidate["gap"], -candidate["flip_pairs"]) < (best["gap"], -best["flip_pairs"]):
                best = candidate
        return best

    def write_scripted_oracle_results(self, *, tasks_by_probe: dict[str, list[dict[str, Any]]], out_dir: Path, overwrite: bool = False) -> None:
        if out_dir.exists() and not overwrite:
            raise FileExistsError(f"Scripted output directory exists: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_dir / "p1_results.jsonl", [self._scripted_p1_row(task) for task in tasks_by_probe["P1"]])
        _write_jsonl(out_dir / "p2_results.jsonl", [self._scripted_repair_row(task) for task in tasks_by_probe["P2"]])
        _write_jsonl(out_dir / "p3_results.jsonl", [self._scripted_p3_row(task) for task in tasks_by_probe["P3"]])
        _write_jsonl(out_dir / "p4_results.jsonl", [self._scripted_p4_row(task) for task in tasks_by_probe["P4"]])
        _write_json(out_dir / "run_manifest.json", {"runner_name": "scripted_oracle", "domain": DOMAIN, "pilot_version": CIRCUIT_PILOT_V2_VERSION})

    def write_scripted_noop_results(self, *, tasks_by_probe: dict[str, list[dict[str, Any]]], out_dir: Path, overwrite: bool = False) -> None:
        if out_dir.exists() and not overwrite:
            raise FileExistsError(f"Scripted no-op output directory exists: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        p1_rows = [{**self._scripted_p1_row(task), "runner_name": "scripted_noop"} for task in tasks_by_probe["P1"]]
        p2_rows = [
            {
                "task_id": task["task_id"],
                "runner_name": "scripted_noop",
                "steps": [{"action_type": "propose_design", "candidate": task["initial_design"]}],
            }
            for task in tasks_by_probe["P2"]
        ]
        p3_rows = [
            {
                "task_id": task["task_id"],
                "runner_name": "scripted_noop",
                "steps": [{"action_type": "propose_design", "candidate": task["corrupted_design"]}],
            }
            for task in tasks_by_probe["P3"]
        ]
        p4_rows = [{**self._scripted_p4_row(task), "runner_name": "scripted_noop"} for task in tasks_by_probe["P4"]]
        _write_jsonl(out_dir / "p1_results.jsonl", p1_rows)
        _write_jsonl(out_dir / "p2_results.jsonl", p2_rows)
        _write_jsonl(out_dir / "p3_results.jsonl", p3_rows)
        _write_jsonl(out_dir / "p4_results.jsonl", p4_rows)
        _write_json(out_dir / "run_manifest.json", {"runner_name": "scripted_noop", "domain": DOMAIN, "pilot_version": CIRCUIT_PILOT_V2_VERSION})
