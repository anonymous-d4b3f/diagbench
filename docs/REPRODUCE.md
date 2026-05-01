# Reproducing DiagBench

This document describes the review-period artifact workflow. It reproduces task loading, oracle checks, public evaluator smoke tests, and paper-facing analysis inputs without using provider API keys.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The package is installed in editable mode through `requirements.txt`, which points at `code/src`.

## 2. Validate Released Data

```bash
python scripts/validate_release.py
```

Expected task counts:

| Domain | P1 | P2 | P3 | P4 | Total |
|---|---:|---:|---:|---:|---:|
| VEH | 240 | 208 | 156 | 159 | 763 |
| Circuit | 32 | 32 | 18 | 24 | 106 |

The validator checks JSONL parseability, required task fields, P4 candidate/policy fields, and SHA256 checksums in `data/manifests/release_manifest.json`.

## 3. Run Smoke Tests

```bash
python -m pytest -q tests
```

These tests import `PiezoelectricOracle` and `CircuitOracle`, load representative released tasks, and run lightweight oracle/evaluator checks. They are intentionally small so reviewers can run them quickly.

## 4. Inspect Task Banks

Primary task files:

```text
data/veh/p1_tasks.jsonl
data/veh/p2_tasks.jsonl
data/veh/p3_tasks.jsonl
data/veh/p4_tasks.jsonl
data/circuit/p1_tasks.jsonl
data/circuit/p2_tasks.jsonl
data/circuit/p3_tasks.jsonl
data/circuit/p4_tasks.jsonl
```

Split manifests and hashes:

```text
data/veh/splits/
data/manifests/release_manifest.json
```

Some `task_id` values and raw-output directory names retain historical runner tokens for hash stability. Treat them as opaque stable identifiers; the released files are the final paper-facing task banks.

## 5. Recompute Public Analysis Components

The artifact includes the public scripts used for scoring and audit calculations:

```bash
PYTHONPATH=code/src python code/scripts/quantify_response_control_profiles.py --help
PYTHONPATH=code/src python code/scripts/evaluate_circuit_pilot.py --help
PYTHONPATH=code/src python code/scripts/baselines/cmaes_baseline.py --help
```

The paper-facing generated tables are stored in:

```text
paper/generated/
results/analysis/paper_generated/
```

The second directory mirrors the generated paper tables for reviewers who inspect results without opening the LaTeX source.

## 6. Reproduce Model Calls

Provider-specific API runners and keys are excluded from the anonymous artifact. To reproduce a run:

1. Use the prompt templates in `prompts/`.
2. Submit prompts to a model with temperature 0.
3. Save one JSON object per task in the released JSONL response format.
4. Score the JSONL outputs with the public evaluators in `code/src/diagbench/evaluation/`.

Closed API models may change after the paper snapshot. The released raw JSONL logs, run manifests, prompt templates, and task hashes preserve the exact reported snapshot.

## 7. Artifact Scope

Included:

- final task banks,
- oracle and evaluator code,
- prompt templates,
- anonymized raw outputs,
- analysis summaries,
- generated paper tables,
- construction audit documentation.

Excluded:

- API keys,
- provider-specific live runners,
- local extraction scratch state,
- local filesystem paths,
- git history from the working repository.
