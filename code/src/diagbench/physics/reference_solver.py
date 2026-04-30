"""
Reference solver portfolio for RBKF (Reference Best-Known Feasible) generation.

Replaces v1 fixture_midpoint_reference with a multi-strategy optimizer portfolio.

Portfolio strategy:
  1. Multi-start L-BFGS-B (20 restarts, penalized objective)
  2. Nelder-Mead (5 restarts)
  3. Differential Evolution (scipy, global search proxy)
  4. Latin Hypercube Sampling (1000 random points, dense coverage)

The best feasible result across all strategies is the RBKF.
This is NOT guaranteed to be the global optimum — it is the best-known feasible
found under this portfolio, hence "reference best-known feasible solution."

BKF record will include:
  source_solver = "reference_solver_portfolio"
  search_budget = total oracle calls across all strategies
  oracle_tier   = "analytical" (or "matlab"/"comsol" if high-fidelity oracle used)
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Optional

from diagbench.physics.oracle import PiezoelectricOracle, OracleResult


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioResult:
    """Result from ReferenceSolverPortfolio."""
    task_id: str
    candidate: dict[str, float]
    objective_value: float          # load_power_uw of best feasible
    oracle_result: OracleResult
    source_solver: str              # "reference_solver_portfolio"
    search_budget: int              # total oracle calls
    oracle_tier: str                # "analytical" / "matlab" / "comsol"
    is_feasible: bool               # False if no feasible solution found
    solver_breakdown: dict[str, int]  # {solver_name: calls} for provenance

    def to_bkf_dict(self) -> dict:
        """Convert to format compatible with BKFRecord.build()."""
        return {
            "candidate": self.candidate,
            "objective_value": round(self.objective_value, 6),
            "objective_name": "load_power_uw",
            "source_solver": self.source_solver,
            "search_budget": self.search_budget,
            "constraint_slack": self.oracle_result.constraint_slack,
            "oracle_tier": self.oracle_tier,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio solver
# ──────────────────────────────────────────────────────────────────────────────

class ReferenceSolverPortfolio:
    """
    Multi-strategy RBKF generator for VEHBench-DiagBench.

    Usage:
        oracle = PiezoelectricOracle()
        portfolio = ReferenceSolverPortfolio(oracle)
        result = portfolio.compute(task)
        if result.is_feasible:
            bkf_dict = result.to_bkf_dict()
    """

    def __init__(
        self,
        oracle: PiezoelectricOracle,
        n_lbfgsb_restarts: int = 20,
        n_nelder_restarts: int = 5,
        lhs_samples: int = 1000,
        de_maxiter: int = 300,
        random_seed: int = 42,
        penalty_scale: float = 1e6,
    ) -> None:
        self._oracle = oracle
        self._n_lbfgsb = n_lbfgsb_restarts
        self._n_nelder = n_nelder_restarts
        self._lhs_samples = lhs_samples
        self._de_maxiter = de_maxiter
        self._seed = random_seed
        self._penalty = penalty_scale

    def compute(self, task: dict) -> PortfolioResult:
        """
        Run reference solver portfolio on a task.

        Args:
            task: Task dict with design_variables, variable_bounds,
                  excitation_context, environment_context, constraints.

        Returns:
            PortfolioResult — check .is_feasible before using .candidate.
        """
        vars_list = task["design_variables"]
        bounds_list = [(task["variable_bounds"][v]["min"],
                        task["variable_bounds"][v]["max"])
                       for v in vars_list]
        excitation = task["excitation_context"]
        environment = task.get("environment_context", {})

        # Build constraint limits dict from task
        constraint_limits: dict[str, float] = {}
        for c in task.get("constraints", []):
            constraint_limits[c["name"]] = c["limit"]

        task_id = task.get("task_id", "unknown")

        if os.getenv("DIAGBENCH_DISABLE_SCIPY") == "1":
            return self._compute_without_scipy(
                task_id=task_id,
                vars_list=vars_list,
                bounds_list=bounds_list,
                excitation=excitation,
                environment=environment,
                constraint_limits=constraint_limits,
            )

        try:
            import numpy as np
            from scipy.optimize import differential_evolution, minimize
            from scipy.stats import qmc
        except Exception:
            return self._compute_without_scipy(
                task_id=task_id,
                vars_list=vars_list,
                bounds_list=bounds_list,
                excitation=excitation,
                environment=environment,
                constraint_limits=constraint_limits,
            )

        all_candidates: list[tuple[object, str]] = []

        def _oracle_call(x_arr: object) -> OracleResult:
            if hasattr(x_arr, "tolist"):
                values = x_arr.tolist()
            else:
                values = list(x_arr)
            params = dict(zip(vars_list, values))
            return self._oracle.evaluate(
                params, excitation,
                constraints=constraint_limits,
                environment=environment,
            )

        def _penalized_neg_obj(x_arr: np.ndarray) -> float:
            """Penalized objective: -power + penalty * total_violation."""
            result = _oracle_call(x_arr)
            power = result.load_power_uw
            if result.is_feasible:
                return -power
            # Sum of constraint violations (positive = violated)
            total_violation = sum(max(0.0, -s) for s in result.constraint_slack.values())
            return -power + self._penalty * total_violation

        rng = np.random.default_rng(self._seed)

        # ── Solver 1: Multi-start L-BFGS-B ────────────────────────────────
        nfev_lbfgsb = 0
        for _ in range(self._n_lbfgsb):
            x0 = rng.uniform(
                [b[0] for b in bounds_list],
                [b[1] for b in bounds_list],
            )
            res = minimize(
                _penalized_neg_obj, x0,
                method="L-BFGS-B",
                bounds=bounds_list,
                options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6},
            )
            nfev_lbfgsb += res.nfev
            all_candidates.append((res.x, "lbfgsb"))

        # ── Solver 2: Nelder-Mead ─────────────────────────────────────────
        nfev_nelder = 0
        for _ in range(self._n_nelder):
            x0 = rng.uniform(
                [b[0] for b in bounds_list],
                [b[1] for b in bounds_list],
            )
            res = minimize(
                _penalized_neg_obj, x0,
                method="Nelder-Mead",
                bounds=bounds_list,
                options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-5},
            )
            nfev_nelder += res.nfev
            all_candidates.append((res.x, "nelder_mead"))

        # ── Solver 3: Differential Evolution (global search) ──────────────
        de_result = differential_evolution(
            _penalized_neg_obj,
            bounds_list,
            seed=self._seed,
            maxiter=self._de_maxiter,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            popsize=10,
            workers=1,
            updating="immediate",
        )
        all_candidates.append((de_result.x, "differential_evolution"))

        # ── Solver 4: Latin Hypercube Sampling ────────────────────────────
        sampler = qmc.LatinHypercube(d=len(bounds_list), seed=self._seed)
        lhs_unit = sampler.random(n=self._lhs_samples)
        lo = np.array([b[0] for b in bounds_list])
        hi = np.array([b[1] for b in bounds_list])
        lhs_scaled = qmc.scale(lhs_unit, lo, hi)
        for x in lhs_scaled:
            all_candidates.append((x, "latin_hypercube"))

        # ── Select best feasible candidate ────────────────────────────────
        best_result: Optional[OracleResult] = None
        best_x: Optional[np.ndarray] = None
        best_power = -float("inf")
        selection_counts = {
            "lbfgsb": 0,
            "nelder_mead": 0,
            "differential_evolution": 0,
            "latin_hypercube": 0,
        }

        for x, source in all_candidates:
            # Clip to bounds (numerical solvers may slightly exceed)
            x_clipped = np.clip(x, lo, hi)
            result = _oracle_call(x_clipped)
            selection_counts[source] += 1
            if result.is_feasible and result.load_power_uw > best_power:
                best_power = result.load_power_uw
                best_result = result
                best_x = x_clipped

        budget_per_solver = {
            "lbfgsb": nfev_lbfgsb + selection_counts["lbfgsb"],
            "nelder_mead": nfev_nelder + selection_counts["nelder_mead"],
            "differential_evolution": de_result.nfev + selection_counts["differential_evolution"],
            "latin_hypercube": selection_counts["latin_hypercube"],
        }
        total_calls = sum(budget_per_solver.values())

        if best_x is None or best_result is None:
            # No feasible solution found — task may be infeasible
            return PortfolioResult(
                task_id=task_id,
                candidate={},
                objective_value=0.0,
                oracle_result=OracleResult(
                    resonant_freq_hz=0.0,
                    load_power_uw=0.0,
                    tip_stress_mpa=0.0,
                    tip_disp_mm=0.0,
                    freq_error_pct=0.0,
                    is_feasible=False,
                    constraint_slack={},
                ),
                source_solver="reference_solver_portfolio",
                search_budget=total_calls,
                oracle_tier="analytical",
                is_feasible=False,
                solver_breakdown=budget_per_solver,
            )

        candidate = {
            var: round(float(val), 8)
            for var, val in zip(vars_list, best_x.tolist())
        }

        return PortfolioResult(
            task_id=task_id,
            candidate=candidate,
            objective_value=round(best_power, 6),
            oracle_result=best_result,
            source_solver="reference_solver_portfolio",
            search_budget=total_calls,
            oracle_tier="analytical",
            is_feasible=True,
            solver_breakdown=budget_per_solver,
        )

    def _compute_without_scipy(
        self,
        *,
        task_id: str,
        vars_list: list[str],
        bounds_list: list[tuple[float, float]],
        excitation: dict,
        environment: dict,
        constraint_limits: dict[str, float],
    ) -> PortfolioResult:
        rng = random.Random(self._seed)
        lo = [float(bound[0]) for bound in bounds_list]
        hi = [float(bound[1]) for bound in bounds_list]

        def _clip(values: list[float]) -> list[float]:
            return [
                min(max(float(value), lo_i), hi_i)
                for value, lo_i, hi_i in zip(values, lo, hi)
            ]

        def _oracle_call(values: list[float]) -> OracleResult:
            params = dict(zip(vars_list, values))
            return self._oracle.evaluate(
                params,
                excitation,
                constraints=constraint_limits,
                environment=environment,
            )

        def _score(result: OracleResult) -> float:
            if result.is_feasible:
                return float(result.load_power_uw)
            total_violation = sum(max(0.0, -float(slack)) for slack in result.constraint_slack.values())
            return float(result.load_power_uw) - self._penalty * total_violation

        def _lhs_points(n: int) -> list[list[float]]:
            points = [[0.0] * len(bounds_list) for _ in range(n)]
            for dim, (lo_i, hi_i) in enumerate(bounds_list):
                bins = list(range(n))
                rng.shuffle(bins)
                for idx, bin_id in enumerate(bins):
                    u = (bin_id + rng.random()) / n
                    points[idx][dim] = lo_i + u * (hi_i - lo_i)
            return points

        midpoint = [(lo_i + hi_i) / 2.0 for lo_i, hi_i in bounds_list]
        evaluated: list[tuple[list[float], OracleResult, str]] = []
        solver_breakdown = {
            "lbfgsb": 0,
            "nelder_mead": 0,
            "differential_evolution": 0,
            "latin_hypercube": 0,
            "fallback_local_search": 0,
        }

        def _record(values: list[float], source: str) -> tuple[list[float], OracleResult]:
            clipped = _clip(values)
            result = _oracle_call(clipped)
            evaluated.append((clipped, result, source))
            solver_breakdown[source] = solver_breakdown.get(source, 0) + 1
            return clipped, result

        _record(midpoint, "latin_hypercube")
        for point in _lhs_points(max(4, self._lhs_samples)):
            _record(point, "latin_hypercube")

        ranked = sorted(evaluated, key=lambda item: _score(item[1]), reverse=True)
        local_seed_count = min(8, len(ranked))
        step_fractions = [0.20, 0.10, 0.05, 0.02]

        for seed_values, seed_result, _source in ranked[:local_seed_count]:
            current = list(seed_values)
            best_result = seed_result
            best_score = _score(seed_result)
            for step_fraction in step_fractions:
                improved = True
                passes = 0
                while improved and passes < 2:
                    improved = False
                    passes += 1
                    for dim, (lo_i, hi_i) in enumerate(bounds_list):
                        step = max((hi_i - lo_i) * step_fraction, 1e-9)
                        for direction in (-1.0, 1.0):
                            proposal = list(current)
                            proposal[dim] = proposal[dim] + direction * step
                            proposal, proposal_result = _record(proposal, "fallback_local_search")
                            proposal_score = _score(proposal_result)
                            if proposal_score > best_score:
                                current = proposal
                                best_result = proposal_result
                                best_score = proposal_score
                                improved = True

        best_entry: tuple[list[float], OracleResult, str] | None = None
        best_power = -float("inf")
        for values, result, source in evaluated:
            if result.is_feasible and float(result.load_power_uw) > best_power:
                best_power = float(result.load_power_uw)
                best_entry = (values, result, source)

        total_calls = sum(solver_breakdown.values())
        if best_entry is None:
            return PortfolioResult(
                task_id=task_id,
                candidate={},
                objective_value=0.0,
                oracle_result=OracleResult(
                    resonant_freq_hz=0.0,
                    load_power_uw=0.0,
                    tip_stress_mpa=0.0,
                    tip_disp_mm=0.0,
                    freq_error_pct=0.0,
                    is_feasible=False,
                    constraint_slack={},
                ),
                source_solver="reference_solver_portfolio_fallback",
                search_budget=total_calls,
                oracle_tier="analytical",
                is_feasible=False,
                solver_breakdown=solver_breakdown,
            )

        values, best_result, _source = best_entry
        candidate = {
            variable: round(float(value), 8)
            for variable, value in zip(vars_list, values)
        }
        return PortfolioResult(
            task_id=task_id,
            candidate=candidate,
            objective_value=round(best_power, 6),
            oracle_result=best_result,
            source_solver="reference_solver_portfolio_fallback",
            search_budget=total_calls,
            oracle_tier="analytical",
            is_feasible=True,
            solver_breakdown=solver_breakdown,
        )
