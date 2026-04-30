#!/usr/bin/env python3
"""Build the harder v2 circuit-design pilot task bank and audit bundle."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagbench.domains.circuit import CIRCUIT_PILOT_V2_VERSION, CircuitPilotV2Builder


DEFAULT_OUT_DIR = ROOT / "data" / CIRCUIT_PILOT_V2_VERSION
DEFAULT_AUDIT_DIR = ROOT / "artifacts" / "audit" / CIRCUIT_PILOT_V2_VERSION
DEFAULT_SCRIPTED_DIR = ROOT / "artifacts" / "runs" / CIRCUIT_PILOT_V2_VERSION / "scripted_oracle"
DEFAULT_NOOP_DIR = ROOT / "artifacts" / "runs" / CIRCUIT_PILOT_V2_VERSION / "scripted_noop"
SCHEMA_PATH = ROOT / "schemas" / "circuit_pilot_v2_task_schema.json"


def _load_schema() -> dict:
    with SCHEMA_PATH.open() as fh:
        return json.load(fh)


def _validate_tasks(tasks_by_probe: dict[str, list[dict]]) -> None:
    schema = _load_schema()
    validator = jsonschema.Draft7Validator(schema)
    errors: list[str] = []
    for probe, tasks in tasks_by_probe.items():
        for task in tasks:
            task_errors = sorted(validator.iter_errors(task), key=lambda item: list(item.path))
            for error in task_errors:
                path = ".".join(str(part) for part in error.path) or "<root>"
                errors.append(f"{probe}:{task.get('task_id')}:{path}: {error.message}")
    if errors:
        raise SystemExit("Circuit pilot v2 schema validation failed:\n" + "\n".join(errors[:50]))


def _remove_if_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build harder circuit pilot v2 tasks and audit bundle.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--scripted-run-dir", type=Path, default=DEFAULT_SCRIPTED_DIR)
    parser.add_argument("--scripted-noop-dir", type=Path, default=DEFAULT_NOOP_DIR)
    parser.add_argument("--seed", type=int, default=1702)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-scripted-run", action="store_true")
    parser.add_argument("--no-noop-run", action="store_true")
    args = parser.parse_args()

    _remove_if_overwrite(args.out_dir, args.overwrite)
    _remove_if_overwrite(args.audit_dir, args.overwrite)
    if not args.no_scripted_run:
        _remove_if_overwrite(args.scripted_run_dir, args.overwrite)
    if not args.no_noop_run:
        _remove_if_overwrite(args.scripted_noop_dir, args.overwrite)

    builder = CircuitPilotV2Builder(seed=args.seed)
    tasks_by_probe = builder.write(out_dir=args.out_dir, audit_dir=args.audit_dir, overwrite=args.overwrite)
    _validate_tasks(tasks_by_probe)
    if not args.no_scripted_run:
        builder.write_scripted_oracle_results(
            tasks_by_probe=tasks_by_probe,
            out_dir=args.scripted_run_dir,
            overwrite=args.overwrite,
        )
    if not args.no_noop_run:
        builder.write_scripted_noop_results(
            tasks_by_probe=tasks_by_probe,
            out_dir=args.scripted_noop_dir,
            overwrite=args.overwrite,
        )

    print(f"Built {sum(len(tasks) for tasks in tasks_by_probe.values())} circuit pilot v2 tasks")
    print(f"Tasks:   {args.out_dir}")
    print(f"Audit:   {args.audit_dir}")
    if not args.no_scripted_run:
        print(f"Scripted oracle results: {args.scripted_run_dir}")
    if not args.no_noop_run:
        print(f"Scripted no-op results:  {args.scripted_noop_dir}")


if __name__ == "__main__":
    main()
