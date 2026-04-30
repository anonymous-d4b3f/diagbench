"""Deterministic closed-form oracle for the circuit pilot.

The pilot intentionally avoids SPICE and uses only simple analytical formulas.
The goal is construct-validity evidence for P1--P4 decision regimes, not broad
circuit-design coverage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


EPS = 1e-12


@dataclass(frozen=True)
class CircuitOracleResult:
    feasible: bool
    metrics: dict[str, float]
    violations: list[dict[str, Any]]
    total_violation: float
    objective_score: float
    oracle_feedback: str
    formulas_used: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "metrics": {key: round(value, 12) for key, value in self.metrics.items()},
            "violations": self.violations,
            "total_violation": round(self.total_violation, 12),
            "objective_score": round(self.objective_score, 12),
            "oracle_feedback": self.oracle_feedback,
            "formulas_used": self.formulas_used,
        }


def safe_target_violation(value: float, target: float, tolerance_rel: float) -> float:
    scale = max(abs(float(target)), EPS)
    tolerance_rel = max(float(tolerance_rel), 1e-6)
    return max(0.0, abs(float(value) - float(target)) / (tolerance_rel * scale) - 1.0)


def safe_log_violation(value: float, target: float, tolerance_rel: float) -> float:
    value = max(abs(float(value)), EPS)
    target = max(abs(float(target)), EPS)
    tolerance_rel = max(float(tolerance_rel), 0.01)
    return max(0.0, abs(math.log(value / target)) / math.log(1.0 + tolerance_rel) - 1.0)


def upper_bound_violation(value: float, upper: float, scale: float | None = None) -> float:
    scale = max(abs(float(scale if scale is not None else upper)), EPS)
    return max(0.0, (float(value) - float(upper)) / scale)


def lower_bound_violation(value: float, lower: float, scale: float | None = None) -> float:
    scale = max(abs(float(scale if scale is not None else lower)), EPS)
    return max(0.0, (float(lower) - float(value)) / scale)


def _parallel(*resistances: float) -> float:
    inv = sum(1.0 / max(float(r), EPS) for r in resistances)
    return 1.0 / max(inv, EPS)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class CircuitOracle:
    """Closed-form verifier for the circuit pilot families."""

    def evaluate(self, task: dict[str, Any], design: dict[str, Any]) -> CircuitOracleResult:
        clean_design = self._coerce_design(task, design)
        family = task["family"]
        if family in {"rc_filter", "rlc_filter"}:
            metrics, formulas = self._metrics_rc_filter(task, clean_design)
        elif family == "loaded_divider":
            metrics, formulas = self._metrics_loaded_divider(task, clean_design)
        elif family == "led_current_limit":
            metrics, formulas = self._metrics_led_current_limit(task, clean_design)
        elif family == "op_amp_amplifier":
            metrics, formulas = self._metrics_op_amp(task, clean_design)
        elif family == "linear_regulator":
            metrics, formulas = self._metrics_linear_regulator(task, clean_design)
        else:
            raise ValueError(f"Unsupported circuit family: {family}")

        violations = self._evaluate_bounds(task, clean_design)
        violations.extend(self._evaluate_constraints(task, metrics))
        total_violation = sum(float(item["normalized_violation"]) for item in violations)
        feasible = total_violation <= 1e-9
        objective_score = self._objective_score(task, metrics, feasible=feasible)
        feedback = self._feedback(violations)
        return CircuitOracleResult(
            feasible=feasible,
            metrics=metrics,
            violations=violations,
            total_violation=total_violation,
            objective_score=objective_score,
            oracle_feedback=feedback,
            formulas_used=formulas,
        )

    def proof_for_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Return deterministic construction proof for P1/audit tasks."""
        proof = dict(task.get("oracle_metadata", {}).get("proof", {}))
        if proof:
            return proof
        reference = task.get("best_known_feasible") or task.get("reference_design")
        if isinstance(reference, dict):
            result = self.evaluate(task, reference)
            return {
                "proof_type": "reference_design_evaluation",
                "reference_design": reference,
                "oracle_result": result.to_dict(),
            }
        return {"proof_type": "metadata_only", "note": "No reference design supplied."}

    def _coerce_design(self, task: dict[str, Any], design: dict[str, Any]) -> dict[str, float]:
        if not isinstance(design, dict):
            raise ValueError("design must be a JSON object")
        cleaned: dict[str, float] = {}
        for variable in task.get("design_variables", []):
            if variable not in design:
                raise ValueError(f"missing design variable: {variable}")
            value = float(design[variable])
            bounds = task.get("variable_bounds", {}).get(variable)
            if isinstance(bounds, dict):
                lo = float(bounds["min"])
                hi = float(bounds["max"])
                if value < lo or value > hi:
                    # Keep the value for physical evaluation but expose the bound
                    # violation through the generic constraint path below.
                    pass
            cleaned[variable] = value
        return cleaned

    @staticmethod
    def _metrics_rc_filter(task: dict[str, Any], design: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        spec = task["spec"]
        r = max(float(design.get("R_ohm", design.get("R_series_ohm", 0.0))), EPS)
        c = max(float(design["C_f"]), EPS)
        fc = 1.0 / (2.0 * math.pi * r * c)
        vin = float(spec.get("vin_v", 5.0))
        source_current = vin / r
        power_w = vin * source_current
        metrics = {
            "fc_hz": fc,
            "fc_error_rel": abs(fc - float(spec.get("target_fc_hz", fc))) / max(abs(float(spec.get("target_fc_hz", fc))), EPS),
            "source_current_a": source_current,
            "power_w": power_w,
            "component_cost": 0.5 + 0.08 * math.log10(r) + 0.08 * max(0.0, -math.log10(c)),
            "robustness_margin": 1.0 / (1.0 + abs(math.log(fc / max(float(spec.get("target_fc_hz", fc)), EPS)))),
        }
        if "L_h" in design:
            l = max(float(design["L_h"]), EPS)
            metrics["resonant_hz"] = 1.0 / (2.0 * math.pi * math.sqrt(l * c))
            metrics["q_factor"] = math.sqrt(l / c) / r
        return metrics, ["fc = 1/(2*pi*R*C)", "I_source = Vin/R"]

    @staticmethod
    def _metrics_loaded_divider(task: dict[str, Any], design: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        spec = task["spec"]
        vin = float(spec["vin_v"])
        r1 = max(float(design["R1_ohm"]), EPS)
        r2 = max(float(design["R2_ohm"]), EPS)
        rload = max(float(spec["load_ohm"]), EPS)
        r2_eff = _parallel(r2, rload)
        vout = vin * r2_eff / (r1 + r2_eff)
        divider_current = vin / (r1 + r2_eff)
        load_current = vout / rload
        metrics = {
            "vout_v": vout,
            "vout_error_rel": abs(vout - float(spec.get("target_vout_v", vout))) / max(abs(float(spec.get("target_vout_v", vout))), EPS),
            "divider_current_a": divider_current,
            "load_current_a": load_current,
            "power_w": vin * divider_current,
            "load_regulation_error_v": abs(vout - float(spec.get("target_vout_v", vout))),
            "component_cost": 0.4 + 0.05 * (math.log10(r1) + math.log10(r2)),
            "robustness_margin": min(r1, r2) / max(r1, r2),
        }
        return metrics, ["R2_eff = 1/(1/R2 + 1/Rload)", "Vout = Vin*R2_eff/(R1+R2_eff)"]

    @staticmethod
    def _metrics_led_current_limit(task: dict[str, Any], design: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        spec = task["spec"]
        vs = float(spec["supply_v"])
        vf = float(spec["led_vf_v"])
        r = max(float(design["R_ohm"]), EPS)
        current = max(0.0, (vs - vf) / r)
        power = current * current * r
        metrics = {
            "led_current_a": current,
            "led_current_error_rel": abs(current - float(spec.get("target_current_a", current))) / max(abs(float(spec.get("target_current_a", current))), EPS),
            "resistor_power_w": power,
            "safety_margin_w": float(spec.get("resistor_power_rating_w", 0.25)) - power,
            "component_cost": 0.15 + 0.02 * math.log10(r),
            "robustness_margin": max(0.0, float(spec.get("resistor_power_rating_w", 0.25)) - power),
        }
        return metrics, ["I_led = max(0, (Vs - Vf)/R)", "P_R = I_led^2 * R"]

    @staticmethod
    def _metrics_op_amp(task: dict[str, Any], design: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        spec = task["spec"]
        mode = spec.get("mode", "non_inverting")
        rf = max(float(design["Rf_ohm"]), EPS)
        if mode == "inverting":
            rin = max(float(design["Rin_ohm"]), EPS)
            gain = abs(rf / rin)
            input_impedance = rin
            formulas = ["gain = abs(Rf/Rin)", "BW = GBW/gain", "Vout_peak = Vin_peak*gain"]
        else:
            rg = max(float(design["Rg_ohm"]), EPS)
            gain = 1.0 + rf / rg
            input_impedance = float(spec.get("noninv_input_impedance_ohm", 1e9))
            formulas = ["gain = 1 + Rf/Rg", "BW = GBW/gain", "Vout_peak = Vin_peak*gain"]
        gbw = float(spec["gbw_hz"])
        bandwidth = gbw / max(gain, EPS)
        vin_peak = float(spec.get("input_vpp_v", 0.1)) / 2.0
        vout_peak = vin_peak * gain
        vcc = float(spec["vcc_v"])
        vsat = float(spec.get("vsat_v", 1.0))
        i_max = float(spec.get("output_current_limit_a", 0.02))
        rload = max(float(design.get("Rload_ohm", spec.get("load_ohm", 10000.0))), EPS)
        vout_max = min(max(vcc - vsat, 0.0), i_max * rload)
        metrics = {
            "gain_v_per_v": gain,
            "gain_error_rel": abs(gain - float(spec.get("target_gain", gain))) / max(abs(float(spec.get("target_gain", gain))), EPS),
            "bandwidth_hz": bandwidth,
            "input_impedance_ohm": input_impedance,
            "vout_peak_required_v": vout_peak,
            "vout_max_v": vout_max,
            "swing_margin_v": vout_max - vout_peak,
            "component_cost": 0.8 + 0.04 * math.log10(rf),
            "robustness_margin": min(max((bandwidth / max(float(spec.get("min_bandwidth_hz", bandwidth)), EPS)) - 1.0, 0.0), 1.0),
        }
        return metrics, formulas + ["Vout_max = min(Vcc - Vsat, Imax*Rload)"]

    @staticmethod
    def _metrics_linear_regulator(task: dict[str, Any], design: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        spec = task["spec"]
        vin = float(spec["vin_v"])
        vout = float(design["vout_v"])
        dropout = float(design["dropout_v"])
        theta = float(design["thermal_resistance_c_per_w"])
        iload = float(spec["load_current_a"])
        ambient = float(spec.get("ambient_c", 25.0))
        power_loss = max(0.0, (vin - vout) * iload)
        junction_temp = ambient + power_loss * theta
        metrics = {
            "vout_v": vout,
            "vout_error_rel": abs(vout - float(spec.get("target_vout_v", vout))) / max(abs(float(spec.get("target_vout_v", vout))), EPS),
            "dropout_margin_v": vin - vout - dropout,
            "power_loss_w": power_loss,
            "junction_temp_c": junction_temp,
            "efficiency": vout / max(vin, EPS),
            "component_cost": 1.0 + 0.02 * max(0.0, 120.0 - theta),
            "robustness_margin": max(0.0, vin - vout - dropout),
        }
        return metrics, ["P_loss = (Vin - Vout)*Iload", "Tj = Tamb + P_loss*theta", "dropout_margin = Vin - Vout - dropout"]

    def _evaluate_constraints(self, task: dict[str, Any], metrics: dict[str, float]) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []

        for constraint in task.get("constraints", []):
            metric_name = constraint.get("metric", constraint["name"])
            if metric_name not in metrics:
                continue
            value = float(metrics[metric_name])
            ctype = constraint["type"]
            if ctype == "target_rel":
                violation = safe_target_violation(value, float(constraint["target"]), float(constraint["tolerance_rel"]))
                target = constraint["target"]
            elif ctype == "target_log":
                violation = safe_log_violation(value, float(constraint["target"]), float(constraint["tolerance_rel"]))
                target = constraint["target"]
            elif ctype == "upper_bound":
                violation = upper_bound_violation(value, float(constraint["limit"]), constraint.get("scale"))
                target = constraint["limit"]
            elif ctype == "lower_bound":
                violation = lower_bound_violation(value, float(constraint["limit"]), constraint.get("scale"))
                target = constraint["limit"]
            else:
                raise ValueError(f"Unsupported constraint type: {ctype}")
            if violation > 0.0:
                violations.append(
                    {
                        "name": constraint["name"],
                        "metric": metric_name,
                        "type": ctype,
                        "value": value,
                        "target": target,
                        "normalized_violation": violation,
                        "unit": constraint.get("unit", ""),
                    }
                )
        return violations

    @staticmethod
    def _evaluate_bounds(task: dict[str, Any], design: dict[str, float]) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for variable, bounds in task.get("variable_bounds", {}).items():
            if variable not in design:
                continue
            value = float(design[variable])
            lo = float(bounds["min"])
            hi = float(bounds["max"])
            scale = max(abs(hi - lo), EPS)
            if value < lo:
                violations.append(
                    {
                        "name": f"bounds.{variable}",
                        "metric": variable,
                        "type": "lower_bound",
                        "value": value,
                        "target": lo,
                        "normalized_violation": (lo - value) / scale,
                        "unit": bounds.get("unit", ""),
                    }
                )
            if value > hi:
                violations.append(
                    {
                        "name": f"bounds.{variable}",
                        "metric": variable,
                        "type": "upper_bound",
                        "value": value,
                        "target": hi,
                        "normalized_violation": (value - hi) / scale,
                        "unit": bounds.get("unit", ""),
                    }
                )
        return violations

    def _objective_score(self, task: dict[str, Any], metrics: dict[str, float], *, feasible: bool) -> float:
        objective = task.get("objective", {})
        metric = objective.get("metric")
        if not metric or metric not in metrics:
            return 1.0 if feasible else 0.0
        value = float(metrics[metric])
        best = float(objective.get("best", value if value else 1.0))
        worst = float(objective.get("worst", 0.0))
        if abs(best - worst) < EPS:
            score = 1.0
        elif objective.get("direction", "maximize") == "minimize":
            score = (worst - value) / (worst - best)
        else:
            score = (value - worst) / (best - worst)
        score = _clip01(score)
        return score if feasible else 0.0

    @staticmethod
    def _feedback(violations: list[dict[str, Any]]) -> str:
        if not violations:
            return "All checked circuit constraints are satisfied."
        worst = max(violations, key=lambda item: float(item["normalized_violation"]))
        return (
            f"Constraint {worst['name']} is violated: metric {worst['metric']}="
            f"{worst['value']:.6g} vs target/limit {worst['target']:.6g}."
        )
