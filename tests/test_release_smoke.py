from __future__ import annotations

import json
from pathlib import Path

from diagbench.domains.circuit.oracle import CircuitOracle
from diagbench.physics.oracle import PiezoelectricOracle


ROOT = Path(__file__).resolve().parents[1]


def test_oracles_importable():
    assert PiezoelectricOracle().__class__.__name__ == "PiezoelectricOracle"
    assert CircuitOracle().__class__.__name__ == "CircuitOracle"


def test_task_counts_and_jsonl():
    expected = {
        "data/veh/p1_tasks.jsonl": 240,
        "data/veh/p2_tasks.jsonl": 208,
        "data/veh/p3_tasks.jsonl": 156,
        "data/veh/p4_tasks.jsonl": 159,
        "data/circuit/p1_tasks.jsonl": 32,
        "data/circuit/p2_tasks.jsonl": 32,
        "data/circuit/p3_tasks.jsonl": 18,
        "data/circuit/p4_tasks.jsonl": 24,
    }
    for rel, count in expected.items():
        rows = [json.loads(line) for line in (ROOT / rel).read_text().splitlines() if line.strip()]
        assert len(rows) == count
        assert all("task_id" in row for row in rows)
