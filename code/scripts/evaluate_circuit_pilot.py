#!/usr/bin/env python3
"""Evaluate model/scripted outputs for the circuit pilot."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagbench.domains.circuit import CIRCUIT_PILOT_VERSION, CircuitPilotEvaluator


DEFAULT_RESULTS = ROOT / "artifacts" / "runs" / CIRCUIT_PILOT_VERSION / "scripted_oracle"
DEFAULT_TASKS = ROOT / "data" / CIRCUIT_PILOT_VERSION
DEFAULT_OUT = ROOT / "results" / "analysis" / CIRCUIT_PILOT_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate circuit pilot outputs.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)

    evaluator = CircuitPilotEvaluator()
    summary = evaluator.evaluate_directory(results_dir=args.results, tasks_dir=args.tasks)
    evaluator.write_outputs(summary=summary, out_dir=args.out_dir, overwrite=args.overwrite)

    print(f"Runner:  {summary['runner_name']}")
    print(f"Results: {args.results}")
    print(f"Summary: {args.out_dir / 'summary.json'}")
    print(f"Table:   {args.out_dir / 'pilot_table.md'}")


if __name__ == "__main__":
    main()
