#!/usr/bin/env python3
"""CMA-ES baseline for DiagBench VEH P2 constrained repair/search.

Runs CMA-ES under the same analytical oracle and query budget as the LLM agents.
Produces P2-compatible evaluation output for direct comparison.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

# ── Add project root ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

try:
    import cma
except ImportError:
    print("cma not installed. Run: pip install cma --break-system-packages")
    sys.exit(1)

from diagbench.physics.oracle import PiezoelectricOracle, OracleResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_tasks(task_files: list[str]) -> list[dict]:
    """Load all P2 tasks from one or more JSONL files."""
    tasks = []
    for fpath in task_files:
        with open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    return tasks


def _task_to_bounds(task: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract variable bounds and names from a P2 task."""
    var_names = list(task["variable_bounds"].keys())
    lb = np.array([task["variable_bounds"][v]["min"] for v in var_names])
    ub = np.array([task["variable_bounds"][v]["max"] for v in var_names])
    return lb, ub, var_names


def _dict_to_array(d: dict, var_names: list[str]) -> np.ndarray:
    return np.array([d[v] for v in var_names])


def _array_to_dict(x: np.ndarray, var_names: list[str]) -> dict:
    return {v: float(x[i]) for i, v in enumerate(var_names)}


def _target_fitness(
    x: np.ndarray,
    var_names: list[str],
    oracle: PiezoelectricOracle,
    excitation: dict,
    constraints: Optional[dict],
    environment: Optional[dict],
    bkf_power_uw: float,
) -> float:
    """CMA-ES fitness: maximize feasible power, penalize infeasibility.

    Returns negative fitness (CMA-ES minimizes). Higher is better.
    """
    params = _array_to_dict(x, var_names)
    result: OracleResult = oracle.evaluate(params, excitation, constraints, environment)

    if result.is_feasible:
        power_uw = result.load_power_uw
        # Return high reward for feasible, normalized by BKF
        return -(power_uw / max(bkf_power_uw, 0.01))
    else:
        # Penalize by sum of negative constraint slacks
        total_violation = sum(max(0, -v) for v in result.constraint_slack.values())
        return total_violation + 1.0  # shift so feasible always > infeasible


def run_cmaes_on_task(
    task: dict,
    oracle: PiezoelectricOracle,
    bkf_power_uw: float,
    query_budget: int = 6,
    population_size: int = 8,
    sigma0: float = 0.15,
    max_iter: int = 50,
) -> dict:
    """Run CMA-ES on a single P2 task.

    Args:
        task: P2 task dict.
        oracle: PiezoelectricOracle instance.
        bkf_power_uw: BKF reference power (µW) for normalizing objective.
        query_budget: Max oracle calls (matching LLM query budget).
        population_size: CMA-ES population size per generation.
        sigma0: Initial step size (relative to bounds range).
        max_iter: Maximum CMA-ES iterations.

    Returns:
        dict with trajectory, final_result, metadata.
    """
    lb, ub, var_names = _task_to_bounds(task)
    # Initialize from BKF if available, otherwise center of bounds
    bkf_design = task.get("best_known_feasible")
    if bkf_design and all(v in bkf_design for v in var_names):
        x0 = _dict_to_array(bkf_design, var_names)
    else:
        x0 = (lb + ub) / 2.0  # fallback: center of bounds

    excitation = task.get("excitation_context", {})
    constraints = task.get("constraints")
    environment = task.get("physics_metadata") or task.get("environment_context")

    # Build constraint dict from task format
    constraint_dict = None
    if constraints:
        constraint_dict = {}
        for c in constraints:
            if "limit" in c:
                constraint_dict[c["name"]] = c["limit"]
            elif "target" in c:
                # Convert target-based to limit-based for oracle
                pass

    # Pass constraints through as-is; oracle handles list or dict
    try:
        constraint_arg = constraint_dict if constraint_dict else constraints
    except Exception:
        constraint_arg = constraints

    # Scale sigma0 to bounds range
    bounds_range = np.mean(ub - lb)
    sigma = sigma0 * bounds_range

    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        sigma,
        {
            "bounds": [lb.tolist(), ub.tolist()],
            "maxfevals": query_budget,
            "popsize": min(population_size, query_budget),
            "verbose": -9,
            "CMA_diagonal": False,
        },
    )

    trajectory = []
    best_feasible_power = 0.0
    best_feasible_design = None
    oracle_calls = 0

    while not es.stop() and oracle_calls < query_budget:
        solutions = es.ask()
        fitnesses = []
        for x in solutions:
            if oracle_calls >= query_budget:
                break
            oracle_calls += 1
            f = _target_fitness(
                np.array(x), var_names, oracle,
                excitation, constraint_arg, environment,
                bkf_power_uw,
            )
            fitnesses.append(f)
            result: OracleResult = oracle.evaluate(
                _array_to_dict(np.array(x), var_names),
                excitation, constraint_arg, environment,
            )

            total_viol = sum(max(0, -v) for v in result.constraint_slack.values())
            step_record = {
                "query": oracle_calls,
                "design": _array_to_dict(np.array(x), var_names),
                "feasible": result.is_feasible,
                "total_violation": total_viol,
                "power_uw": result.load_power_uw,
            }
            if result.is_feasible:
                pw = result.load_power_uw
                step_record["power_ratio"] = pw / max(bkf_power_uw, 0.01)
                if pw > best_feasible_power:
                    best_feasible_power = pw
                    best_feasible_design = _array_to_dict(np.array(x), var_names)
            trajectory.append(step_record)

        if fitnesses:
            es.tell(solutions[:len(fitnesses)], fitnesses)

        # Early stop if we found a good feasible design
        if best_feasible_power / max(bkf_power_uw, 0.01) > 0.95:
            break

    # Final evaluation of best design
    final_feasible = best_feasible_power > 0
    final_power_ratio = best_feasible_power / max(bkf_power_uw, 0.01) if final_feasible else 0.0

    return {
        "task_id": task["task_id"],
        "final_feasible": final_feasible,
        "final_power_uw": best_feasible_power,
        "final_power_ratio": final_power_ratio,
        "bkf_power_uw": bkf_power_uw,
        "oracle_calls": oracle_calls,
        "trajectory": trajectory,
        "best_design": best_feasible_design,
        "split": task.get("split", "unknown"),
        "source_group": task.get("source_group", "unknown"),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-files", nargs="+", required=True,
                    help="P2 task JSONL files")
    ap.add_argument("--bkf-file", required=True,
                    help="BKF reference JSONL file")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for results")
    ap.add_argument("--query-budget", type=int, default=6,
                    help="Oracle query budget (default: 6, matching LLM)")
    ap.add_argument("--max-tasks", type=int, default=0,
                    help="Max tasks to run (0 = all)")
    args = ap.parse_args()

    # Load tasks
    tasks = _load_tasks(args.task_files)
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
    print(f"Loaded {len(tasks)} P2 tasks")

    # Load BKF references
    bkf_map = {}
    with open(args.bkf_file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                bkf_map[r["task_id"]] = r

    # Init oracle
    oracle = PiezoelectricOracle()

    # Run
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.monotonic()

    for i, task in enumerate(tasks):
        tid = task["task_id"]
        bkf_entry = bkf_map.get(tid, {})
        bkf_power = bkf_entry.get("objective_value", 15.0)  # default if missing

        r = run_cmaes_on_task(
            task, oracle, bkf_power,
            query_budget=args.query_budget,
        )
        results.append(r)

        if (i + 1) % 20 == 0:
            elapsed = time.monotonic() - t0
            feasible_n = sum(1 for r_ in results if r_["final_feasible"])
            print(f"  [{i+1}/{len(tasks)}] feasible={feasible_n}/{i+1}  "
                  f"({elapsed:.1f}s)")

    elapsed = time.monotonic() - t0
    feasible_n = sum(1 for r_ in results if r_["final_feasible"])

    # Summary
    power_ratios = [r["final_power_ratio"] for r in results]
    mean_ratio = np.mean(power_ratios) if power_ratios else 0.0
    median_ratio = np.median(power_ratios) if power_ratios else 0.0
    mean_calls = np.mean([r["oracle_calls"] for r in results])

    summary = {
        "method": "CMA-ES",
        "n_tasks": len(tasks),
        "query_budget": args.query_budget,
        "final_feasible_rate": feasible_n / len(tasks),
        "mean_power_ratio": float(mean_ratio),
        "median_power_ratio": float(median_ratio),
        "mean_oracle_calls": float(mean_calls),
        "total_time_s": elapsed,
        "split_breakdown": {},
    }
    # Per-split breakdown
    for r in results:
        sp = r["split"]
        if sp not in summary["split_breakdown"]:
            summary["split_breakdown"][sp] = {"n": 0, "feasible": 0, "ratios": []}
        summary["split_breakdown"][sp]["n"] += 1
        if r["final_feasible"]:
            summary["split_breakdown"][sp]["feasible"] += 1
        summary["split_breakdown"][sp]["ratios"].append(r["final_power_ratio"])

    for sp in summary["split_breakdown"]:
        sb = summary["split_breakdown"][sp]
        sb["feasible_rate"] = sb["feasible"] / sb["n"]
        sb["mean_ratio"] = float(np.mean(sb["ratios"]))
        del sb["ratios"]

    # Save
    with open(out_dir / "cmaes_results.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    with open(out_dir / "cmaes_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n=== CMA-ES P2 Summary ===")
    print(f"Tasks: {len(tasks)}")
    print(f"Final feasible rate: {summary['final_feasible_rate']:.4f}")
    print(f"Mean power ratio:    {summary['mean_power_ratio']:.4f}")
    print(f"Median power ratio:  {summary['median_power_ratio']:.4f}")
    print(f"Mean oracle calls:   {summary['mean_oracle_calls']:.1f}")
    print(f"Total time:          {elapsed:.1f}s")
    for sp, sb in summary["split_breakdown"].items():
        print(f"  {sp}: feasible={sb['feasible_rate']:.4f}  "
              f"mean_ratio={sb['mean_ratio']:.4f}")
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
