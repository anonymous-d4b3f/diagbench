"""Circuit pilot v3.1 with VEH-style P1/P2 hardening."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from diagbench.domains.circuit.builder import (
    DOMAIN,
    _lower,
    _target_log,
    _target_rel,
    _task_hash,
    _upper,
    _write_json,
    _write_jsonl,
)
from diagbench.domains.circuit.v3_builder import CircuitPilotV3Builder


CIRCUIT_PILOT_V3_1_VERSION = "circuit_pilot_v3_1"
TASK_COUNTS_V31 = {"P1": 32, "P2": 32, "P3": 30, "P4": 24}


def _base_task_v31(
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
        "pilot_version": CIRCUIT_PILOT_V3_1_VERSION,
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


def _manifest_v31(tasks: list[dict[str, Any]], *, seed: int, probe: str, artifact_path: Path) -> dict[str, Any]:
    return {
        "domain": DOMAIN,
        "pilot_version": CIRCUIT_PILOT_V3_1_VERSION,
        "probe": probe,
        "n_tasks": len(tasks),
        "seed": seed,
        "artifact_path": str(artifact_path),
        "artifact_sha256": _task_hash({"tasks": tasks}),
        "task_ids": [task["task_id"] for task in tasks],
    }


class CircuitPilotV31Builder(CircuitPilotV3Builder):
    """Build circuit_pilot_v3_1.

    v3.1 applies the mature VEH P1 lesson to the circuit pilot: P1 must score
    credible triage rather than raw action accuracy, and P2 objectives must be
    calibrated on the feasible frontier rather than on unreachable global
    values.
    """

    def build(self) -> dict[str, list[dict[str, Any]]]:
        tasks = {
            "P1": self.build_p1_tasks(),
            "P2": self.build_p2_tasks(),
            "P3": self.build_p3_tasks(),
            "P4": self.build_p4_tasks(),
        }
        for probe, expected in TASK_COUNTS_V31.items():
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
            _write_json(out_dir / f"{probe.lower()}_manifest.json", _manifest_v31(tasks, seed=self.seed, probe=probe, artifact_path=task_path))
            for task in tasks:
                self.write_audit_bundle(task=task, audit_root=audit_dir)
        _write_json(out_dir / "dataset_summary.json", self.dataset_summary(tasks_by_probe))
        return tasks_by_probe

    def dataset_summary(self, tasks_by_probe: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return {
            "domain": DOMAIN,
            "pilot_version": CIRCUIT_PILOT_V3_1_VERSION,
            "seed": self.seed,
            "task_counts": {probe: len(tasks) for probe, tasks in tasks_by_probe.items()},
            "families": sorted({task["family"] for tasks in tasks_by_probe.values() for task in tasks}),
            "hardening_mechanisms": [
                "P1 VEH-style credible triage subtypes",
                "P1 exact missing-field and infeasibility-proof scoring metadata",
                "P2 feasible-frontier objective calibration",
                "P2 op-amp gain-bandwidth-swing tasks with robustness objective",
            ],
        }

    def build_p1_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for subtype, cases in self._p1_v31_cases().items():
            for idx, case in enumerate(cases):
                task = self._make_p1_task(subtype=subtype, idx=idx, case=case)
                if task["gold_label"]["action_type"] == "propose_design":
                    reference = task.get("reference_design") or task.get("best_known_feasible")
                    if not self.oracle.evaluate(task, reference).feasible:
                        raise AssertionError(f"P1 v3.1 reference is not feasible: {task['task_id']}")
                tasks.append(task)
        return tasks

    def build_p2_tasks(self) -> list[dict[str, Any]]:
        base = [self._retag_v31(task) for task in super().build_p2_tasks()[:16]]
        return base + self._p2_v31_calibrated_hard_tasks()

    def build_p3_tasks(self) -> list[dict[str, Any]]:
        return [self._retag_v31(task) for task in super().build_p3_tasks()]

    def build_p4_tasks(self) -> list[dict[str, Any]]:
        return [self._retag_v31(task) for task in super().build_p4_tasks()]

    def write_scripted_oracle_results(self, *, tasks_by_probe: dict[str, list[dict[str, Any]]], out_dir: Path, overwrite: bool = False) -> None:
        super().write_scripted_oracle_results(tasks_by_probe=tasks_by_probe, out_dir=out_dir, overwrite=overwrite)
        _write_json(out_dir / "run_manifest.json", {"runner_name": "scripted_oracle", "domain": DOMAIN, "pilot_version": CIRCUIT_PILOT_V3_1_VERSION})

    def write_scripted_noop_results(self, *, tasks_by_probe: dict[str, list[dict[str, Any]]], out_dir: Path, overwrite: bool = False) -> None:
        super().write_scripted_noop_results(tasks_by_probe=tasks_by_probe, out_dir=out_dir, overwrite=overwrite)
        _write_json(out_dir / "run_manifest.json", {"runner_name": "scripted_noop", "domain": DOMAIN, "pilot_version": CIRCUIT_PILOT_V3_1_VERSION})

    @staticmethod
    def _scripted_p1_row(task: dict[str, Any]) -> dict[str, Any]:
        action = dict(task["gold_label"])
        if action["action_type"] == "propose_design":
            action["candidate"] = task.get("reference_design") or task.get("best_known_feasible")
        if action["action_type"] == "declare_infeasible" and task.get("proof_requirements"):
            action["proof"] = dict(task["proof_requirements"])
            action["reason"] = f"blocking constraint {action['proof'].get('blocking_constraint')} violates metric {action['proof'].get('metric')}"
        return {"task_id": task["task_id"], "runner_name": "scripted_oracle", "parsed_action": action}

    def _retag_v31(self, task: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(task)
        out["task_id"] = (
            out["task_id"]
            .replace("circuit_pilot_v3", CIRCUIT_PILOT_V3_1_VERSION)
            .replace("circuit_pilot_v2", CIRCUIT_PILOT_V3_1_VERSION)
            .replace("circuit_pilot_v1", CIRCUIT_PILOT_V3_1_VERSION)
        )
        out["pilot_version"] = CIRCUIT_PILOT_V3_1_VERSION
        out.setdefault("oracle_metadata", {})["oracle_version"] = "circuit_oracle_v0.3"
        out["task_sha256"] = _task_hash({key: value for key, value in out.items() if key != "task_sha256"})
        return out

    @staticmethod
    def _rc_parts(target: float, *, vin: float = 5.0, tol: float = 0.02) -> tuple[dict[str, Any], list[str], dict[str, dict[str, float | str]], list[dict[str, Any]], dict[str, Any]]:
        spec = {"filter_type": "lowpass", "target_fc_hz": target, "vin_v": vin}
        variables = ["R_ohm", "C_f"]
        bounds = {"R_ohm": {"min": 1000.0, "max": 200000.0, "unit": "ohm"}, "C_f": {"min": 5e-10, "max": 2e-6, "unit": "F"}}
        constraints = [_target_log("cutoff_frequency", "fc_hz", target, tol, "Hz"), _upper("source_current", "source_current_a", 0.00045, "A", scale=0.00045)]
        objective = {"name": "low_source_current", "metric": "source_current_a", "direction": "minimize", "best": 0.000025, "worst": 0.003}
        return spec, variables, bounds, constraints, objective

    @staticmethod
    def _divider_parts(target: float, *, load: float = 100000.0, vin: float = 5.0, tol: float = 0.02) -> tuple[dict[str, Any], list[str], dict[str, dict[str, float | str]], list[dict[str, Any]], dict[str, Any]]:
        spec = {"vin_v": vin, "target_vout_v": target, "load_ohm": load}
        variables = ["R1_ohm", "R2_ohm"]
        bounds = {"R1_ohm": {"min": 47.0, "max": 200000.0, "unit": "ohm"}, "R2_ohm": {"min": 47.0, "max": 200000.0, "unit": "ohm"}}
        constraints = [_target_rel("output_voltage", "vout_v", target, tol, "V"), _upper("divider_current", "divider_current_a", 0.02, "A", scale=0.02)]
        objective = {"name": "low_power", "metric": "power_w", "direction": "minimize", "best": 2e-5, "worst": 0.02}
        return spec, variables, bounds, constraints, objective

    @staticmethod
    def _led_parts(current: float, *, tol: float = 0.02, rating: float = 0.25) -> tuple[dict[str, Any], list[str], dict[str, dict[str, float | str]], list[dict[str, Any]], dict[str, Any]]:
        spec = {"supply_v": 5.0, "led_vf_v": 2.0, "target_current_a": current, "resistor_power_rating_w": rating}
        variables = ["R_ohm"]
        bounds = {"R_ohm": {"min": 20.0, "max": 5000.0, "unit": "ohm"}}
        constraints = [_target_rel("led_current", "led_current_a", current, tol, "A"), _upper("resistor_power", "resistor_power_w", rating, "W", scale=rating)]
        objective = {"name": "low_resistor_power", "metric": "resistor_power_w", "direction": "minimize", "best": 0.005, "worst": rating}
        return spec, variables, bounds, constraints, objective

    @staticmethod
    def _opamp_parts(gain: float, *, mode: str = "inverting", gbw: float = 2_000_000.0, min_bw: float = 100_000.0, vin_pp: float = 0.2, tol: float = 0.05) -> tuple[dict[str, Any], list[str], dict[str, dict[str, float | str]], list[dict[str, Any]], dict[str, Any]]:
        spec = {"mode": mode, "target_gain": gain, "gbw_hz": gbw, "min_bandwidth_hz": min_bw, "vcc_v": 5.0, "vsat_v": 0.7, "output_current_limit_a": 0.001, "load_ohm": 10000.0, "input_vpp_v": vin_pp}
        if mode == "inverting":
            variables = ["Rf_ohm", "Rin_ohm", "Rload_ohm"]
            bounds = {"Rf_ohm": {"min": 1000.0, "max": 300000.0, "unit": "ohm"}, "Rin_ohm": {"min": 1000.0, "max": 50000.0, "unit": "ohm"}, "Rload_ohm": {"min": 500.0, "max": 20000.0, "unit": "ohm"}}
        else:
            variables = ["Rf_ohm", "Rg_ohm", "Rload_ohm"]
            bounds = {"Rf_ohm": {"min": 1000.0, "max": 300000.0, "unit": "ohm"}, "Rg_ohm": {"min": 1000.0, "max": 50000.0, "unit": "ohm"}, "Rload_ohm": {"min": 500.0, "max": 20000.0, "unit": "ohm"}}
        constraints = [
            _target_log("closed_loop_gain", "gain_v_per_v", gain, tol, "V/V"),
            _lower("minimum_bandwidth", "bandwidth_hz", min_bw, "Hz", scale=min_bw),
            _lower("swing_margin", "swing_margin_v", 0.2, "V", scale=1.0),
        ]
        if mode == "inverting":
            constraints.append(_lower("input_impedance", "input_impedance_ohm", 8000.0, "ohm", scale=8000.0))
        objective = {"name": "bandwidth_robustness", "metric": "robustness_margin", "direction": "maximize", "best": 0.25, "worst": 0.0}
        return spec, variables, bounds, constraints, objective

    def _make_p1_task(self, *, subtype: str, idx: int, case: dict[str, Any]) -> dict[str, Any]:
        spec, variables, bounds, constraints, objective = case["parts"]
        extra = {
            "gold_label": case["gold_label"],
            "reference_design": case.get("reference_design"),
            "spec_context": case.get("spec_context", {}),
            "p1_response_requirements": {
                "score_candidate_feasibility": True,
                "score_missing_field_exactness": True,
                "score_infeasibility_proof": True,
            },
            "hardening": case.get("hardening", {"subtype": subtype}),
        }
        if case.get("missing_fields"):
            extra["missing_fields_ground_truth"] = case["missing_fields"]
        if case.get("proof_requirements"):
            extra["proof_requirements"] = case["proof_requirements"]
        extra["oracle_metadata"] = {
            "oracle": "closed_form_circuit_oracle",
            "oracle_version": "circuit_oracle_v0.3",
            "unit_system": "SI",
            "proof": case.get("proof_requirements", {"proof_type": "reference_or_label_metadata"}),
        }
        return _base_task_v31(
            task_id=f"{CIRCUIT_PILOT_V3_1_VERSION}::P1::{subtype}::{idx:02d}",
            probe="P1",
            family=case["family"],
            subtype=subtype,
            spec=copy.deepcopy(spec),
            design_variables=list(variables),
            variable_bounds=copy.deepcopy(bounds),
            constraints=copy.deepcopy(constraints),
            objective=copy.deepcopy(objective),
            query_budget=1,
            best_known_feasible=case.get("reference_design"),
            extra=extra,
        )

    def _p1_v31_cases(self) -> dict[str, list[dict[str, Any]]]:
        rc_1k = self._rc_parts(1000.0, tol=0.02)
        rc_18 = self._rc_parts(1800.0, tol=0.005)
        div_25 = self._divider_parts(2.5, load=1_000_000.0, tol=0.02)
        div_33 = self._divider_parts(3.3, load=100_000.0, tol=0.015)
        led_10 = self._led_parts(0.01, tol=0.03, rating=0.25)
        led_20 = self._led_parts(0.02, tol=0.01, rating=0.063)
        op_10 = self._opamp_parts(10.0, gbw=2_000_000.0, min_bw=150_000.0, vin_pp=0.25)
        op_18 = self._opamp_parts(18.0, gbw=1_200_000.0, min_bw=60_000.0, vin_pp=0.3)

        missing_rc = self._rc_parts(1800.0, tol=0.015)
        missing_rc[0].pop("vin_v", None)
        missing_op = self._opamp_parts(16.0, gbw=10_000_000.0, min_bw=100_000.0, vin_pp=0.2)
        missing_op[0].pop("input_vpp_v", None)
        missing_div = self._divider_parts(2.5, load=100_000.0)
        missing_div[0].pop("load_ohm", None)
        missing_led = self._led_parts(0.01)
        missing_led[0].pop("led_vf_v", None)

        impossible_div = self._divider_parts(2.3, load=100.0, vin=3.3, tol=0.02)
        impossible_div[1].clear()
        impossible_div[1].extend(["R1_ohm", "R2_ohm"])
        impossible_div[2]["R1_ohm"] = {"min": 47.0, "max": 1000.0, "unit": "ohm"}
        impossible_div[2]["R2_ohm"] = {"min": 47.0, "max": 1000.0, "unit": "ohm"}
        impossible_led = self._led_parts(0.05, tol=0.02, rating=0.125)
        impossible_op = self._opamp_parts(20.0, gbw=1_000_000.0, min_bw=55_000.0, vin_pp=0.2, tol=0.02)
        impossible_rc = self._rc_parts(1000.0, vin=5.0, tol=0.02)
        impossible_rc[2]["R_ohm"] = {"min": 1000.0, "max": 100000.0, "unit": "ohm"}
        impossible_rc[2]["C_f"] = {"min": 1e-9, "max": 10e-9, "unit": "F"}

        return {
            "feasible_base": [
                {"family": "rc_filter", "parts": rc_1k, "reference_design": {"R_ohm": 20000.0, "C_f": 7.957747154594767e-9}, "gold_label": {"action_type": "propose_design"}},
                {"family": "loaded_divider", "parts": div_25, "reference_design": {"R1_ohm": 10000.0, "R2_ohm": 10200.0}, "gold_label": {"action_type": "propose_design"}},
                {"family": "led_current_limit", "parts": led_10, "reference_design": {"R_ohm": 300.0}, "gold_label": {"action_type": "propose_design"}},
                {"family": "op_amp_amplifier", "parts": op_10, "reference_design": {"Rf_ohm": 80000.0, "Rin_ohm": 8000.0, "Rload_ohm": 5000.0}, "gold_label": {"action_type": "propose_design"}},
            ],
            "feasible_boundary": [
                {"family": "rc_filter", "parts": rc_18, "reference_design": {"R_ohm": 88420.0, "C_f": 1e-9}, "gold_label": {"action_type": "propose_design"}},
                {"family": "loaded_divider", "parts": div_33, "reference_design": {"R1_ohm": 20000.0, "R2_ohm": 63500.0}, "gold_label": {"action_type": "propose_design"}},
                {"family": "led_current_limit", "parts": led_20, "reference_design": {"R_ohm": 150.0}, "gold_label": {"action_type": "propose_design"}},
                {"family": "op_amp_amplifier", "parts": op_18, "reference_design": {"Rf_ohm": 172000.0, "Rin_ohm": 10000.0, "Rload_ohm": 5000.0}, "gold_label": {"action_type": "propose_design"}},
            ],
            "feasible_red_herring": [
                {"family": "rc_filter", "parts": rc_1k, "reference_design": {"R_ohm": 20000.0, "C_f": 7.957747154594767e-9}, "gold_label": {"action_type": "propose_design"}, "spec_context": {"red_herring": "Preferred capacitor dielectric is unspecified; this does not block closed-form triage."}},
                {"family": "loaded_divider", "parts": div_25, "reference_design": {"R1_ohm": 20000.0, "R2_ohm": 20500.0}, "gold_label": {"action_type": "propose_design"}, "spec_context": {"red_herring": "PCB resistor package is not specified; assume standard package for triage."}},
                {"family": "led_current_limit", "parts": led_10, "reference_design": {"R_ohm": 300.0}, "gold_label": {"action_type": "propose_design"}, "spec_context": {"red_herring": "LED color bin is unspecified but forward voltage is already provided."}},
                {"family": "op_amp_amplifier", "parts": op_10, "reference_design": {"Rf_ohm": 82000.0, "Rin_ohm": 8200.0, "Rload_ohm": 8000.0}, "gold_label": {"action_type": "propose_design"}, "spec_context": {"red_herring": "Op-amp package and vendor are not specified; GBW and swing parameters are provided."}},
            ],
            "infeasible_margin": [
                {"family": "loaded_divider", "parts": impossible_div, "gold_label": {"action_type": "declare_infeasible"}, "proof_requirements": {"blocking_constraint": "output_voltage", "metric": "vout_v", "computed_bound": 2.174, "required_bound": 2.254, "margin_ratio": 1.037}},
                {"family": "led_current_limit", "parts": impossible_led, "gold_label": {"action_type": "declare_infeasible"}, "proof_requirements": {"blocking_constraint": "resistor_power", "metric": "resistor_power_w", "computed_bound": 0.144, "required_bound": 0.125, "margin_ratio": 1.152}},
                {"family": "op_amp_amplifier", "parts": impossible_op, "gold_label": {"action_type": "declare_infeasible"}, "proof_requirements": {"blocking_constraint": "minimum_bandwidth", "metric": "bandwidth_hz", "computed_bound": 51020.0, "required_bound": 55000.0, "margin_ratio": 1.078}},
                {"family": "rc_filter", "parts": impossible_rc, "gold_label": {"action_type": "declare_infeasible"}, "proof_requirements": {"blocking_constraint": "cutoff_frequency", "metric": "fc_hz", "computed_bound": 15915.0, "required_bound": 1000.0, "margin_ratio": 15.9}},
            ],
            "infeasible_topology_or_envelope": [
                {"family": "op_amp_amplifier", "parts": self._opamp_parts(50.0, gbw=5_000_000.0, min_bw=200_000.0, vin_pp=0.08, tol=0.02), "gold_label": {"action_type": "declare_infeasible"}, "proof_requirements": {"blocking_constraint": "minimum_bandwidth", "metric": "bandwidth_hz", "computed_bound": 102041.0, "required_bound": 200000.0, "margin_ratio": 1.96}},
                {"family": "loaded_divider", "parts": self._divider_parts(4.8, load=1000.0, vin=5.0, tol=0.01), "gold_label": {"action_type": "declare_infeasible"}, "proof_requirements": {"blocking_constraint": "output_voltage", "metric": "vout_v", "computed_bound": 4.17, "required_bound": 4.752, "margin_ratio": 1.14}},
                {"family": "led_current_limit", "parts": self._led_parts(0.08, tol=0.02, rating=0.125), "gold_label": {"action_type": "declare_infeasible"}, "proof_requirements": {"blocking_constraint": "resistor_power", "metric": "resistor_power_w", "computed_bound": 0.184, "required_bound": 0.125, "margin_ratio": 1.47}},
                {"family": "rc_filter", "parts": self._rc_parts(80.0, vin=5.0, tol=0.02), "gold_label": {"action_type": "declare_infeasible"}, "proof_requirements": {"blocking_constraint": "cutoff_frequency", "metric": "fc_hz", "computed_bound": 398.0, "required_bound": 80.0, "margin_ratio": 4.98}},
            ],
            "missing_blocker_obvious": [
                {"family": "rc_filter", "parts": missing_rc, "gold_label": {"action_type": "request_missing_info", "missing_fields": ["spec.vin_v"]}, "missing_fields": ["spec.vin_v"]},
                {"family": "op_amp_amplifier", "parts": missing_op, "gold_label": {"action_type": "request_missing_info", "missing_fields": ["spec.input_vpp_v"]}, "missing_fields": ["spec.input_vpp_v"]},
                {"family": "loaded_divider", "parts": missing_div, "gold_label": {"action_type": "request_missing_info", "missing_fields": ["spec.load_ohm"]}, "missing_fields": ["spec.load_ohm"]},
                {"family": "led_current_limit", "parts": missing_led, "gold_label": {"action_type": "request_missing_info", "missing_fields": ["spec.led_vf_v"]}, "missing_fields": ["spec.led_vf_v"]},
            ],
            "missing_blocker_ambiguous": [
                {"family": "rc_filter", "parts": missing_rc, "gold_label": {"action_type": "request_missing_info", "missing_fields": ["spec.vin_v"]}, "missing_fields": ["spec.vin_v"], "spec_context": {"ambiguous_note": "The operating source is described only as a small-signal sensor node."}},
                {"family": "loaded_divider", "parts": missing_div, "gold_label": {"action_type": "request_missing_info", "missing_fields": ["spec.load_ohm"]}, "missing_fields": ["spec.load_ohm"], "spec_context": {"ambiguous_note": "The downstream ADC input impedance is not specified."}},
                {"family": "rc_filter", "parts": rc_18, "reference_design": {"R_ohm": 88420.0, "C_f": 1e-9}, "gold_label": {"action_type": "propose_design"}, "spec_context": {"ambiguous_note": "The preferred capacitor series is unspecified; use continuous values for this triage step."}},
                {"family": "op_amp_amplifier", "parts": op_10, "reference_design": {"Rf_ohm": 80000.0, "Rin_ohm": 8000.0, "Rload_ohm": 5000.0}, "gold_label": {"action_type": "propose_design"}, "spec_context": {"ambiguous_note": "Noise density is not specified; it is not part of this closed-form task."}},
            ],
            "missing_nonblocker": [
                {"family": "rc_filter", "parts": rc_1k, "reference_design": {"R_ohm": 20000.0, "C_f": 7.957747154594767e-9}, "gold_label": {"action_type": "propose_design"}, "spec_context": {"missing_nonblocker": "Board area is not specified."}},
                {"family": "loaded_divider", "parts": div_25, "reference_design": {"R1_ohm": 10000.0, "R2_ohm": 10200.0}, "gold_label": {"action_type": "propose_design"}, "spec_context": {"missing_nonblocker": "Resistor package size is not specified."}},
                {"family": "led_current_limit", "parts": led_10, "reference_design": {"R_ohm": 300.0}, "gold_label": {"action_type": "propose_design"}, "spec_context": {"missing_nonblocker": "LED color bin name is not specified."}},
                {"family": "op_amp_amplifier", "parts": op_18, "reference_design": {"Rf_ohm": 172000.0, "Rin_ohm": 10000.0, "Rload_ohm": 5000.0}, "gold_label": {"action_type": "propose_design"}, "spec_context": {"missing_nonblocker": "Vendor part number is not specified."}},
            ],
        }

    def _p2_v31_calibrated_hard_tasks(self) -> list[dict[str, Any]]:
        specs = [
            self._opamp_parts(18.0, gbw=1_200_000.0, min_bw=60_000.0, vin_pp=0.30),
            self._opamp_parts(8.0, gbw=600_000.0, min_bw=70_000.0, vin_pp=0.80),
            self._opamp_parts(12.0, mode="non_inverting", gbw=1_000_000.0, min_bw=80_000.0, vin_pp=0.30),
            self._opamp_parts(25.0, gbw=2_000_000.0, min_bw=75_000.0, vin_pp=0.20),
            self._opamp_parts(16.0, gbw=1_500_000.0, min_bw=85_000.0, vin_pp=0.25),
            self._opamp_parts(6.0, mode="non_inverting", gbw=500_000.0, min_bw=75_000.0, vin_pp=0.80),
            self._opamp_parts(30.0, gbw=3_000_000.0, min_bw=90_000.0, vin_pp=0.15),
            self._opamp_parts(10.0, gbw=900_000.0, min_bw=80_000.0, vin_pp=0.55),
        ]
        tasks: list[dict[str, Any]] = []
        for idx, parts in enumerate(specs):
            spec, variables, bounds, constraints, objective = parts
            initial = self._opamp_initial(spec, variables)
            candidates = self._opamp_candidate_grid(spec, variables)
            best, worst = self._calibrate_objective(parts, candidates, metric="robustness_margin", direction="maximize")
            objective = {**objective, "best": best["score_metric"], "worst": worst["score_metric"]}
            task = _base_task_v31(
                task_id=f"{CIRCUIT_PILOT_V3_1_VERSION}::P2::frontier::{idx:02d}",
                probe="P2",
                family="op_amp_amplifier",
                subtype="calibrated_gain_bandwidth_swing_repair",
                spec=copy.deepcopy(spec),
                design_variables=list(variables),
                variable_bounds=copy.deepcopy(bounds),
                constraints=copy.deepcopy(constraints),
                objective=copy.deepcopy(objective),
                query_budget=3,
                best_known_feasible=best["design"],
                extra={
                    "initial_design": initial,
                    "initial_oracle_result": self.oracle.evaluate(_base_task_v31(
                        task_id="tmp", probe="P2", family="op_amp_amplifier", subtype="tmp",
                        spec=spec, design_variables=variables, variable_bounds=bounds,
                        constraints=constraints, objective=objective, query_budget=3, best_known_feasible=best["design"],
                    ), initial).to_dict(),
                    "oracle_reference_result": None,
                    "hardening": {
                        "mechanism": "calibrated_feasible_frontier",
                        "objective_metric": "robustness_margin",
                        "frontier_best": best["score_metric"],
                        "frontier_worst": worst["score_metric"],
                    },
                },
            )
            task["oracle_reference_result"] = self.oracle.evaluate(task, best["design"]).to_dict()
            tasks.append(task)
        # Duplicate with stricter local-edit starts: the same feasible frontier but
        # starts closer to the boundary, which makes over-edit visible.
        clones: list[dict[str, Any]] = []
        for task in tasks:
            clone = copy.deepcopy(task)
            clone["task_id"] = clone["task_id"].replace("::frontier::", "::local_frontier::")
            clone["subtype"] = "calibrated_local_repair"
            clone["query_budget"] = 2
            clone["initial_design"] = self._near_boundary_initial(clone)
            clone["initial_oracle_result"] = self.oracle.evaluate(clone, clone["initial_design"]).to_dict()
            clone["task_sha256"] = _task_hash({key: value for key, value in clone.items() if key != "task_sha256"})
            clones.append(clone)
        return tasks + clones

    def _calibrate_objective(self, parts: tuple[Any, ...], candidates: list[dict[str, float]], *, metric: str, direction: str) -> tuple[dict[str, Any], dict[str, Any]]:
        spec, variables, bounds, constraints, objective = parts
        task = _base_task_v31(
            task_id="calibration",
            probe="P2",
            family="op_amp_amplifier",
            subtype="calibration",
            spec=spec,
            design_variables=variables,
            variable_bounds=bounds,
            constraints=constraints,
            objective={**objective, "metric": metric, "direction": direction, "best": 1.0, "worst": 0.0},
            query_budget=3,
            best_known_feasible=None,
        )
        feasible = []
        for design in candidates:
            result = self.oracle.evaluate(task, design)
            if result.feasible:
                feasible.append({"design": design, "score_metric": float(result.metrics[metric])})
        if not feasible:
            raise AssertionError("No feasible candidates for P2 v3.1 calibration")
        reverse = direction == "maximize"
        ordered = sorted(feasible, key=lambda item: item["score_metric"], reverse=reverse)
        return ordered[0], ordered[-1]

    @staticmethod
    def _opamp_initial(spec: dict[str, Any], variables: list[str]) -> dict[str, float]:
        if "Rin_ohm" in variables:
            return {"Rf_ohm": min(250000.0, max(100000.0, spec["target_gain"] * 10000.0)), "Rin_ohm": 2000.0, "Rload_ohm": 1000.0}
        return {"Rf_ohm": min(250000.0, max(100000.0, (spec["target_gain"] - 1.0) * 10000.0)), "Rg_ohm": 2000.0, "Rload_ohm": 1000.0}

    @staticmethod
    def _opamp_candidate_grid(spec: dict[str, Any], variables: list[str]) -> list[dict[str, float]]:
        target = float(spec["target_gain"])
        gains = [target * ratio for ratio in (0.95, 0.975, 1.0, 1.025, 1.05)]
        bases = [8000.0, 10000.0, 15000.0, 22000.0]
        loads = [3000.0, 5000.0, 8000.0, 12000.0, 18000.0]
        designs: list[dict[str, float]] = []
        for gain in gains:
            for base in bases:
                for load in loads:
                    if "Rin_ohm" in variables:
                        designs.append({"Rf_ohm": gain * base, "Rin_ohm": base, "Rload_ohm": load})
                    else:
                        designs.append({"Rf_ohm": max(gain - 1.0, 0.1) * base, "Rg_ohm": base, "Rload_ohm": load})
        return designs

    @staticmethod
    def _near_boundary_initial(task: dict[str, Any]) -> dict[str, float]:
        spec = task["spec"]
        variables = task["design_variables"]
        target = float(spec["target_gain"])
        if "Rin_ohm" in variables:
            return {"Rf_ohm": target * 6000.0, "Rin_ohm": 6000.0, "Rload_ohm": 2000.0}
        return {"Rf_ohm": max(target - 1.0, 0.1) * 6000.0, "Rg_ohm": 6000.0, "Rload_ohm": 2000.0}
