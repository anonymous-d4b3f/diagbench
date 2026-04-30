"""Circuit pilot v3 with harder P1/P2/P3 discrimination gates."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from diagbench.domains.circuit.builder import (
    DOMAIN,
    CircuitPilotBuilder,
    _lower,
    _opamp_task_spec,
    _regulator_task_spec,
    _target_log,
    _target_rel,
    _task_hash,
    _upper,
    _write_json,
    _write_jsonl,
)
from diagbench.domains.circuit.v2_builder import CircuitPilotV2Builder


CIRCUIT_PILOT_V3_VERSION = "circuit_pilot_v3"
TASK_COUNTS_V3 = {"P1": 24, "P2": 24, "P3": 30, "P4": 24}


def _base_task_v3(
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
        "pilot_version": CIRCUIT_PILOT_V3_VERSION,
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
            "oracle_version": "circuit_oracle_v0.3",
            "unit_system": "SI",
        },
    }
    if extra:
        task.update(extra)
    task["task_sha256"] = _task_hash({key: value for key, value in task.items() if key != "task_sha256"})
    return task


def _manifest_v3(tasks: list[dict[str, Any]], *, seed: int, probe: str, artifact_path: Path) -> dict[str, Any]:
    return {
        "domain": DOMAIN,
        "pilot_version": CIRCUIT_PILOT_V3_VERSION,
        "probe": probe,
        "n_tasks": len(tasks),
        "seed": seed,
        "artifact_path": str(artifact_path),
        "artifact_sha256": _task_hash({"tasks": tasks}),
        "task_ids": [task["task_id"] for task in tasks],
    }


class CircuitPilotV3Builder(CircuitPilotV2Builder):
    """Build circuit_pilot_v3.

    v3 keeps v2's 74-task comparability core and adds hard P1/P2/P3 items. The
    additions target the observed v2 ceilings: model_D at 100% on P1/P2 and
    model_E at 100% on P3.
    """

    def build(self) -> dict[str, list[dict[str, Any]]]:
        tasks = {
            "P1": self.build_p1_tasks(),
            "P2": self.build_p2_tasks(),
            "P3": self.build_p3_tasks(),
            "P4": self.build_p4_tasks(),
        }
        for probe, expected in TASK_COUNTS_V3.items():
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
            _write_json(out_dir / f"{probe.lower()}_manifest.json", _manifest_v3(tasks, seed=self.seed, probe=probe, artifact_path=task_path))
            for task in tasks:
                self.write_audit_bundle(task=task, audit_root=audit_dir)
        _write_json(out_dir / "dataset_summary.json", self.dataset_summary(tasks_by_probe))
        return tasks_by_probe

    def dataset_summary(self, tasks_by_probe: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return {
            "domain": DOMAIN,
            "pilot_version": CIRCUIT_PILOT_V3_VERSION,
            "seed": self.seed,
            "task_counts": {probe: len(tasks) for probe, tasks in tasks_by_probe.items()},
            "families": sorted({task["family"] for tasks in tasks_by_probe.values() for task in tasks}),
            "hardening_mechanisms": [
                "v2 comparability core",
                "P1 missing-source and near-impossible multi-constraint triage",
                "P2 RLC and op-amp load-coupled three-variable repair",
                "P3 multi-constraint trap states with shorter recovery budgets",
            ],
            "quality_gate": {
                "P1_P2": "model_D should no longer be ceiling-saturated",
                "P3": "model_E should no longer be ceiling-saturated",
            },
        }

    def build_p1_tasks(self) -> list[dict[str, Any]]:
        return [self._retag_v3(task) for task in super().build_p1_tasks()] + self._p1_hard_additions()

    def build_p2_tasks(self) -> list[dict[str, Any]]:
        return [self._retag_v3(task) for task in super().build_p2_tasks()] + self._p2_hard_additions()

    def build_p3_tasks(self) -> list[dict[str, Any]]:
        base = [self._retag_v3(task) for task in super().build_p3_tasks()]
        return base + self._p3_hard_additions()

    def build_p4_tasks(self) -> list[dict[str, Any]]:
        return [self._retag_v3(task) for task in super().build_p4_tasks()]

    def _retag_v3(self, task: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(task)
        out["task_id"] = out["task_id"].replace("circuit_pilot_v2", CIRCUIT_PILOT_V3_VERSION).replace("circuit_pilot_v1", CIRCUIT_PILOT_V3_VERSION)
        out["pilot_version"] = CIRCUIT_PILOT_V3_VERSION
        out.setdefault("oracle_metadata", {})["oracle_version"] = "circuit_oracle_v0.3"
        out["task_sha256"] = _task_hash({key: value for key, value in out.items() if key != "task_sha256"})
        return out

    def _p1_hard_additions(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []

        # Missing-source cases: the target is numerically specified, but a safety
        # or loading constraint depends on a missing operating condition.
        missing_specs = [
            {
                "family": "rc_filter",
                "spec": {"filter_type": "lowpass", "target_fc_hz": 1800.0},
                "variables": ["R_ohm", "C_f"],
                "bounds": {"R_ohm": {"min": 1000.0, "max": 100000.0, "unit": "ohm"}, "C_f": {"min": 1e-9, "max": 1e-6, "unit": "F"}},
                "constraints": [_target_log("cutoff_frequency", "fc_hz", 1800.0, 0.015, "Hz"), _upper("source_current", "source_current_a", 0.00035, "A", scale=0.00035)],
                "objective": {"name": "low_source_current", "metric": "source_current_a", "direction": "minimize", "best": 0.00005, "worst": 0.003},
                "missing": ["spec.vin_v"],
                "reason": "source-current safety cannot be checked without the input voltage",
            },
            {
                "family": "op_amp_amplifier",
                "parts": _opamp_task_spec(16.0, mode="inverting"),
                "missing": ["spec.input_vpp_v"],
                "reason": "output swing cannot be checked without input amplitude",
            },
        ]
        for idx, item in enumerate(missing_specs):
            if "parts" in item:
                spec, variables, bounds, constraints, objective = item["parts"]
                spec = copy.deepcopy(spec)
                for field in item["missing"]:
                    spec.pop(field.split(".", 1)[1], None)
            else:
                spec, variables, bounds, constraints, objective = item["spec"], item["variables"], item["bounds"], item["constraints"], item["objective"]
            tasks.append(
                _base_task_v3(
                    task_id=f"{CIRCUIT_PILOT_V3_VERSION}::P1::request_missing_hard::{idx:02d}",
                    probe="P1",
                    family=item["family"],
                    subtype="request_missing_info_hard",
                    spec=spec,
                    design_variables=variables,
                    variable_bounds=bounds,
                    constraints=constraints,
                    objective=objective,
                    query_budget=1,
                    extra={
                        "gold_label": {"action_type": "request_missing_info", "missing_fields": item["missing"], "reason": item["reason"]},
                        "missing_fields_ground_truth": item["missing"],
                        "hardening": {"mechanism": "missing_operating_condition_for_safety_constraint"},
                        "oracle_metadata": {
                            "oracle": "closed_form_circuit_oracle",
                            "oracle_version": "circuit_oracle_v0.3",
                            "unit_system": "SI",
                            "proof": {"proof_type": "missing_blocker", "missing_fields": item["missing"], "argument": item["reason"]},
                        },
                    },
                )
            )

        infeasible = [
            self._p1_loaded_divider_hard_infeasible(),
            self._p1_led_power_conflict(),
            self._p1_opamp_gbw_swing_conflict(),
        ]
        for idx, case in enumerate(infeasible):
            tasks.append(
                _base_task_v3(
                    task_id=f"{CIRCUIT_PILOT_V3_VERSION}::P1::declare_infeasible_hard::{idx:02d}",
                    probe="P1",
                    family=case["family"],
                    subtype=case["subtype"],
                    spec=case["spec"],
                    design_variables=case["variables"],
                    variable_bounds=case["bounds"],
                    constraints=case["constraints"],
                    objective=case["objective"],
                    query_budget=1,
                    extra={
                        "gold_label": {"action_type": "declare_infeasible", "reason": case["proof"]},
                        "hardening": {"mechanism": "near_boundary_or_multi_constraint_infeasibility", "margin_ratio": case.get("margin_ratio")},
                        "oracle_metadata": {
                            "oracle": "closed_form_circuit_oracle",
                            "oracle_version": "circuit_oracle_v0.3",
                            "unit_system": "SI",
                            "proof": {"proof_type": "closed_form_infeasibility", "argument": case["proof"], "margin_ratio": case.get("margin_ratio")},
                        },
                    },
                )
            )

        narrow = [
            self._p1_rc_ultra_narrow(),
            self._p1_divider_ultra_narrow(),
            self._p1_regulator_near_dropout(),
        ]
        for idx, case in enumerate(narrow):
            task = _base_task_v3(
                task_id=f"{CIRCUIT_PILOT_V3_VERSION}::P1::feasible_ultra_narrow::{idx:02d}",
                probe="P1",
                family=case["family"],
                subtype="feasible_ultra_narrow",
                spec=case["spec"],
                design_variables=case["variables"],
                variable_bounds=case["bounds"],
                constraints=case["constraints"],
                objective=case["objective"],
                query_budget=1,
                best_known_feasible=case["reference"],
                extra={
                    "gold_label": {"action_type": "propose_design"},
                    "reference_design": case["reference"],
                    "hardening": {"mechanism": "sub_percent_feasible_boundary", "tolerance_rel": case.get("tolerance_rel")},
                    "oracle_metadata": {
                        "oracle": "closed_form_circuit_oracle",
                        "oracle_version": "circuit_oracle_v0.3",
                        "unit_system": "SI",
                        "proof": {"proof_type": "ultra_narrow_reference_design_feasible", "reference_design": case["reference"]},
                    },
                },
            )
            if not self.oracle.evaluate(task, case["reference"]).feasible:
                raise AssertionError(f"P1 hard reference is not feasible: {task['task_id']}")
            tasks.append(task)
        return tasks

    @staticmethod
    def _p1_loaded_divider_hard_infeasible() -> dict[str, Any]:
        spec = {"vin_v": 3.3, "target_vout_v": 2.3, "load_ohm": 100.0}
        bounds = {"R1_ohm": {"min": 47.0, "max": 1000.0, "unit": "ohm"}, "R2_ohm": {"min": 47.0, "max": 1000.0, "unit": "ohm"}}
        constraints = [_target_rel("output_voltage", "vout_v", 2.3, 0.02, "V"), _upper("divider_current", "divider_current_a", 0.05, "A", scale=0.05)]
        return {
            "family": "loaded_divider",
            "subtype": "declare_infeasible_loaded_boundary",
            "spec": spec,
            "variables": ["R1_ohm", "R2_ohm"],
            "bounds": bounds,
            "constraints": constraints,
            "objective": {"name": "low_power", "metric": "power_w", "direction": "minimize", "best": 0.00002, "worst": 0.02},
            "proof": "Even at R1=47 ohm and R2=1k, R2||100=90.9 ohm and Vout_max=2.174V, below the 2.254V lower edge.",
            "margin_ratio": 2.254 / 2.174,
        }

    @staticmethod
    def _p1_led_power_conflict() -> dict[str, Any]:
        spec = {"supply_v": 5.0, "led_vf_v": 2.0, "target_current_a": 0.05, "resistor_power_rating_w": 0.125}
        bounds = {"R_ohm": {"min": 50.0, "max": 2000.0, "unit": "ohm"}}
        constraints = [_target_rel("led_current", "led_current_a", 0.05, 0.02, "A"), _upper("resistor_power", "resistor_power_w", 0.125, "W", scale=0.125)]
        return {
            "family": "led_current_limit",
            "subtype": "declare_infeasible_power_conflict",
            "spec": spec,
            "variables": ["R_ohm"],
            "bounds": bounds,
            "constraints": constraints,
            "objective": {"name": "power_margin", "metric": "safety_margin_w", "direction": "maximize", "best": 0.25, "worst": 0.0},
            "proof": "The lowest allowed current in a 50mA +/-2% band is 49mA, which dissipates at least 0.144W in the resistor, above the 0.125W rating.",
            "margin_ratio": 0.144 / 0.125,
        }

    @staticmethod
    def _p1_opamp_gbw_swing_conflict() -> dict[str, Any]:
        spec, variables, bounds, _, objective = _opamp_task_spec(20.0, mode="inverting")
        spec = {**spec, "gbw_hz": 1_000_000.0, "min_bandwidth_hz": 55_000.0, "input_vpp_v": 0.2, "output_current_limit_a": 0.001}
        constraints = [_target_log("closed_loop_gain", "gain_v_per_v", 20.0, 0.02, "V/V"), _lower("minimum_bandwidth", "bandwidth_hz", 55_000.0, "Hz", scale=55_000.0), _lower("swing_margin", "swing_margin_v", 0.2, "V", scale=1.0), _lower("input_impedance", "input_impedance_ohm", 8000.0, "ohm", scale=8000.0)]
        return {
            "family": "op_amp_amplifier",
            "subtype": "declare_infeasible_gbw_boundary",
            "spec": spec,
            "variables": variables,
            "bounds": bounds,
            "constraints": constraints,
            "objective": objective,
            "proof": "The gain band requires gain at least 19.6, but GBW/gain is then at most 51.0kHz, below the 55kHz bandwidth limit.",
            "margin_ratio": 55_000.0 / (1_000_000.0 / 19.6),
        }

    @staticmethod
    def _p1_rc_ultra_narrow() -> dict[str, Any]:
        target = 1800.0
        ref = {"R_ohm": 15000.0, "C_f": 1.0 / (2.0 * 3.141592653589793 * 15000.0 * target)}
        spec = {"filter_type": "lowpass", "target_fc_hz": target, "vin_v": 5.0}
        constraints = [_target_log("cutoff_frequency", "fc_hz", target, 0.005, "Hz"), _upper("source_current", "source_current_a", 0.00035, "A", scale=0.00035)]
        return {
            "family": "rc_filter",
            "spec": spec,
            "variables": ["R_ohm", "C_f"],
            "bounds": {"R_ohm": {"min": 1000.0, "max": 100000.0, "unit": "ohm"}, "C_f": {"min": 1e-9, "max": 1e-6, "unit": "F"}},
            "constraints": constraints,
            "objective": {"name": "low_source_current", "metric": "source_current_a", "direction": "minimize", "best": 0.00005, "worst": 0.003},
            "reference": ref,
            "tolerance_rel": 0.005,
        }

    @staticmethod
    def _p1_divider_ultra_narrow() -> dict[str, Any]:
        spec = {"vin_v": 5.0, "target_vout_v": 3.0, "load_ohm": 47000.0}
        ref = {"R1_ohm": 10000.0, "R2_ohm": 22031.25}
        return {
            "family": "loaded_divider",
            "spec": spec,
            "variables": ["R1_ohm", "R2_ohm"],
            "bounds": {"R1_ohm": {"min": 1000.0, "max": 200000.0, "unit": "ohm"}, "R2_ohm": {"min": 1000.0, "max": 200000.0, "unit": "ohm"}},
            "constraints": [_target_rel("output_voltage", "vout_v", 3.0, 0.005, "V"), _upper("divider_current", "divider_current_a", 0.0004, "A", scale=0.0004)],
            "objective": {"name": "low_power", "metric": "power_w", "direction": "minimize", "best": 0.00002, "worst": 0.005},
            "reference": ref,
            "tolerance_rel": 0.005,
        }

    @staticmethod
    def _p1_regulator_near_dropout() -> dict[str, Any]:
        spec, variables, bounds, _, objective = _regulator_task_spec(4.4)
        spec = {**spec, "vin_v": 5.0, "target_vout_v": 4.4, "load_current_a": 0.2, "ambient_c": 25.0}
        return {
            "family": "linear_regulator",
            "spec": spec,
            "variables": variables,
            "bounds": bounds,
            "constraints": [_target_rel("output_voltage", "vout_v", 4.4, 0.005, "V"), _lower("dropout_margin", "dropout_margin_v", 0.3, "V", scale=1.0), _upper("junction_temp", "junction_temp_c", 85.0, "C", scale=85.0)],
            "objective": objective,
            "reference": {"vout_v": 4.4, "dropout_v": 0.29, "thermal_resistance_c_per_w": 30.0},
            "tolerance_rel": 0.005,
        }

    def _p2_hard_additions(self) -> list[dict[str, Any]]:
        additions = self._p2_rlc_variants() + self._p2_opamp_load_variants()
        tasks: list[dict[str, Any]] = []
        for idx, item in enumerate(additions):
            task = _base_task_v3(
                task_id=f"{CIRCUIT_PILOT_V3_VERSION}::P2::hard::{idx:02d}",
                probe="P2",
                family=item["family"],
                subtype=item["subtype"],
                spec=item["spec"],
                design_variables=item["variables"],
                variable_bounds=item["bounds"],
                constraints=item["constraints"],
                objective=item["objective"],
                query_budget=3,
                best_known_feasible=item["best"],
                extra={"initial_design": item["initial"], "allowed_edits": item["variables"], "hardening": item["hardening"]},
            )
            task["initial_oracle_result"] = self.oracle.evaluate(task, item["initial"]).to_dict()
            task["oracle_reference_result"] = self.oracle.evaluate(task, item["best"]).to_dict()
            if task["oracle_reference_result"]["feasible"] is not True:
                raise AssertionError(f"P2 hard best design is not feasible: {task['task_id']}")
            if task["initial_oracle_result"]["feasible"] is True:
                raise AssertionError(f"P2 hard initial design should be infeasible: {task['task_id']}")
            tasks.append(task)
        return tasks

    @staticmethod
    def _rlc_spec(target_hz: float) -> tuple[dict[str, Any], list[str], dict[str, dict[str, float | str]], dict[str, Any]]:
        return (
            {"filter_type": "series_rlc", "target_resonant_hz": target_hz, "vin_v": 5.0},
            ["R_ohm", "C_f", "L_h"],
            {
                "R_ohm": {"min": 100.0, "max": 100000.0, "unit": "ohm"},
                "C_f": {"min": 1e-9, "max": 1e-6, "unit": "F"},
                "L_h": {"min": 1e-6, "max": 0.1, "unit": "H"},
            },
            {"name": "low_source_current", "metric": "source_current_a", "direction": "minimize", "best": 0.00005, "worst": 0.02},
        )

    def _p2_rlc_variants(self) -> list[dict[str, Any]]:
        configs = [
            (10_000.0, 0.30, 0.0030, {"R_ohm": 500.0, "C_f": 1e-8, "L_h": 0.002533029591}, {"R_ohm": 6800.0, "C_f": 1e-8, "L_h": 0.02533029591}),
            (18_000.0, 0.24, 0.0025, {"R_ohm": 600.0, "C_f": 6.8e-9, "L_h": 0.0010}, {"R_ohm": 6800.0, "C_f": 6.8e-9, "L_h": 0.0114982068}),
            (25_000.0, 0.20, 0.0020, {"R_ohm": 800.0, "C_f": 4.7e-9, "L_h": 0.0008}, {"R_ohm": 8200.0, "C_f": 4.7e-9, "L_h": 0.008625464}),
            (40_000.0, 0.16, 0.0015, {"R_ohm": 1000.0, "C_f": 2.2e-9, "L_h": 0.0002}, {"R_ohm": 15000.0, "C_f": 2.2e-9, "L_h": 0.007196758}),
        ]
        out: list[dict[str, Any]] = []
        for target, q_limit, current_limit, initial, best in configs:
            spec, variables, bounds, objective = self._rlc_spec(target)
            out.append(
                {
                    "family": "rlc_filter",
                    "subtype": "rlc_three_variable_repair",
                    "spec": spec,
                    "variables": variables,
                    "bounds": bounds,
                    "constraints": [
                        _target_log("resonant_frequency", "resonant_hz", target, 0.015, "Hz"),
                        _upper("q_ceiling", "q_factor", q_limit, "", scale=q_limit),
                        _upper("source_current", "source_current_a", current_limit, "A", scale=current_limit),
                    ],
                    "objective": objective,
                    "initial": initial,
                    "best": best,
                    "hardening": {"mechanism": "three_variable_rlc_frequency_q_current_coupling", "coupled_metrics": ["resonant_hz", "q_factor", "source_current_a"]},
                }
            )
        return out

    def _p2_opamp_load_variants(self) -> list[dict[str, Any]]:
        configs = [
            (18.0, 1_200_000.0, 60_000.0, 0.30, 0.0010, {"Rf_ohm": 180000.0, "Rin_ohm": 2000.0, "Rload_ohm": 1000.0}, {"Rf_ohm": 180000.0, "Rin_ohm": 10000.0, "Rload_ohm": 5000.0}),
            (8.0, 600_000.0, 70_000.0, 0.80, 0.0010, {"Rf_ohm": 160000.0, "Rin_ohm": 2000.0, "Rload_ohm": 1000.0}, {"Rf_ohm": 80000.0, "Rin_ohm": 10000.0, "Rload_ohm": 6000.0}),
            (12.0, 1_000_000.0, 80_000.0, 0.30, 0.0010, {"Rf_ohm": 180000.0, "Rg_ohm": 2000.0, "Rload_ohm": 1000.0}, {"Rf_ohm": 110000.0, "Rg_ohm": 10000.0, "Rload_ohm": 5000.0}, "non_inverting"),
            (25.0, 2_000_000.0, 75_000.0, 0.20, 0.0010, {"Rf_ohm": 100000.0, "Rin_ohm": 2000.0, "Rload_ohm": 1000.0}, {"Rf_ohm": 200000.0, "Rin_ohm": 8000.0, "Rload_ohm": 5000.0}),
        ]
        out: list[dict[str, Any]] = []
        for entry in configs:
            gain, gbw, min_bw, input_vpp, i_max, initial, best, *mode_part = entry
            mode = mode_part[0] if mode_part else "inverting"
            spec, variables, bounds, _, objective = _opamp_task_spec(gain, mode=mode)
            spec = {**spec, "gbw_hz": gbw, "min_bandwidth_hz": min_bw, "input_vpp_v": input_vpp, "output_current_limit_a": i_max}
            constraints = [_target_log("closed_loop_gain", "gain_v_per_v", gain, 0.02, "V/V"), _lower("minimum_bandwidth", "bandwidth_hz", min_bw, "Hz", scale=min_bw), _lower("swing_margin", "swing_margin_v", 0.2, "V", scale=1.0)]
            if mode == "inverting":
                constraints.append(_lower("input_impedance", "input_impedance_ohm", 8000.0, "ohm", scale=8000.0))
            out.append(
                {
                    "family": "op_amp_amplifier",
                    "subtype": "opamp_gain_bandwidth_swing_repair",
                    "spec": spec,
                    "variables": variables,
                    "bounds": bounds,
                    "constraints": constraints,
                    "objective": objective,
                    "initial": initial,
                    "best": best,
                    "hardening": {"mechanism": "gain_bandwidth_load_swing_coupling", "coupled_metrics": ["gain_v_per_v", "bandwidth_hz", "swing_margin_v"]},
                }
            )
        return out

    def _p3_hard_additions(self) -> list[dict[str, Any]]:
        sources = self._p2_hard_additions() + [self._retag_v3(task) for task in super().build_p2_tasks()[:4]]
        corruptions = [
            "multi_constraint_continuity_trap",
            "false_safe_margin",
            "escape_then_second_violation",
        ]
        tasks: list[dict[str, Any]] = []
        for idx, source in enumerate(sources[:12]):
            corruption = corruptions[idx % len(corruptions)]
            tasks.append(self._p3_hard_from_source(source=source, corruption=corruption, idx=idx))
        return tasks

    def _p3_hard_from_source(self, *, source: dict[str, Any], corruption: str, idx: int) -> dict[str, Any]:
        bad = dict(source["initial_design"])
        trap = self._hard_trap_candidate(source)
        best = dict(source["best_known_feasible"])
        trap_eval = self.oracle.evaluate(source, trap).to_dict()
        history = [
            {
                "step": 0,
                "action_type": "propose_design",
                "candidate": bad,
                "analysis_summary": "Earlier trajectory trusted a local formula shortcut and targeted only the dominant visible violation.",
                "verifier_response": self.oracle.evaluate(source, bad).to_dict(),
            },
            {
                "step": 1,
                "action_type": "propose_design",
                "candidate": trap,
                "analysis_summary": self._hard_corruption_text(corruption),
                "verifier_response": trap_eval,
            },
        ]
        task = _base_task_v3(
            task_id=f"{CIRCUIT_PILOT_V3_VERSION}::P3::hard::{corruption}::{idx:02d}",
            probe="P3",
            family=source["family"],
            subtype=corruption,
            spec=source["spec"],
            design_variables=source["design_variables"],
            variable_bounds=source["variable_bounds"],
            constraints=source["constraints"],
            objective=source["objective"],
            query_budget=3,
            best_known_feasible=best,
            extra={
                "corruption_type": corruption,
                "corrupted_design": trap,
                "corrupted_history": history,
                "hidden_truth": {
                    "corrupted_oracle_result": trap_eval,
                    "recovery_design": best,
                    "recovery_oracle_result": self.oracle.evaluate(source, best).to_dict(),
                },
                "escape_criterion": {
                    "type": "violation_reduction",
                    "relative_total_violation": 0.4,
                    "description": "Escape requires cutting total violation by 60% from the corrupted trap state.",
                },
                "hardening": {"mechanism": "short_budget_multi_constraint_corrupted_state", "source_task_id": source["task_id"]},
            },
        )
        if self.oracle.evaluate(task, trap).feasible:
            raise AssertionError(f"P3 hard trap should be infeasible: {task['task_id']}")
        if not self.oracle.evaluate(task, best).feasible:
            raise AssertionError(f"P3 hard recovery should be feasible: {task['task_id']}")
        return task

    @staticmethod
    def _hard_corruption_text(corruption: str) -> str:
        return {
            "multi_constraint_continuity_trap": "The previous step appears to fix the visible target metric, so continue from it rather than resetting.",
            "false_safe_margin": "The previous notes claim the safety/load margin is adequate, but that conclusion used the old load or unit scale.",
            "escape_then_second_violation": "The trajectory escapes the first violation but the chosen direction creates a second coupled violation.",
        }[corruption]

    @staticmethod
    def _hard_trap_candidate(source: dict[str, Any]) -> dict[str, float]:
        best = dict(source["best_known_feasible"])
        family = source["family"]
        if family == "rlc_filter":
            trap = dict(best)
            trap["R_ohm"] = max(float(source["variable_bounds"]["R_ohm"]["min"]), min(float(best["R_ohm"]) / 4.0, float(best["R_ohm"]) - 1.0))
            return trap
        if family == "op_amp_amplifier":
            trap = dict(best)
            if "Rload_ohm" in trap:
                trap["Rload_ohm"] = 1000.0
            if "Rin_ohm" in trap:
                trap["Rin_ohm"] = max(float(source["variable_bounds"]["Rin_ohm"]["min"]), float(trap["Rin_ohm"]) / 2.0)
            elif "Rg_ohm" in trap:
                trap["Rg_ohm"] = max(float(source["variable_bounds"]["Rg_ohm"]["min"]), float(trap["Rg_ohm"]) / 2.0)
            return trap
        trap = dict(best)
        first = source["design_variables"][0]
        trap[first] = source["initial_design"][first]
        return trap
