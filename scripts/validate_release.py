#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "data/veh/p1_tasks.jsonl": 240,
    "data/veh/p2_tasks.jsonl": 208,
    "data/veh/p3_tasks.jsonl": 156,
    "data/veh/p4_tasks.jsonl": 159,
    "data/circuit/p1_tasks.jsonl": 32,
    "data/circuit/p2_tasks.jsonl": 32,
    "data/circuit/p3_tasks.jsonl": 18,
    "data/circuit/p4_tasks.jsonl": 24,
}

def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            json.loads(line)
            count += 1
    return count

for rel, expected in EXPECTED.items():
    actual = count_jsonl(ROOT / rel)
    if actual != expected:
        raise SystemExit(f"{rel}: expected {expected}, found {actual}")

from diagbench.physics.oracle import PiezoelectricOracle
from diagbench.domains.circuit.oracle import CircuitOracle

veh_oracle = PiezoelectricOracle()
circuit_oracle = CircuitOracle()
print("OK: release validates", veh_oracle.__class__.__name__, circuit_oracle.__class__.__name__)
