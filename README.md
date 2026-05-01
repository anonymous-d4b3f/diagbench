# DiagBench

DiagBench is a workflow-structured benchmark for evaluating engineering-design agents under verifier-gated physical constraints. This anonymous review artifact contains the paper-facing release: 763 vibration energy harvester (VEH) tasks and a 106-task circuit construct-validity audit across four probes.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/validate_release.py
python -m pytest -q tests
```

The validation script checks task counts, JSONL parseability, required task fields, SHA256 manifest consistency, and importability of the released oracle code. The tests run lightweight oracle and evaluator smoke checks without provider API keys.

## Dataset Summary

| Probe | VEH Tasks | Circuit Tasks | Headline Metric | What It Tests |
|---|---:|---:|---|---|
| P1 | 240 | 32 | P1-Composite | Entry triage: propose, request missing information, or declare infeasible |
| P2 | 208 | 32 | Final feasible power ratio | Iterative repair under verifier feedback |
| P3 | 156 | 18 | P3-Success | Recovery from corrupted trajectory state |
| P4 | 159 | 24 | Full Kendall Tau | Policy-conditioned ranking among feasible candidates |

All public files are the final `release_v1` task banks. Some internal `task_id` strings and raw-output directory names preserve historical runner tokens so that raw JSONL logs remain hash-stable; these tokens are opaque identifiers and are not version-selection instructions.

## File Map

- `data/veh/p1_tasks.jsonl` through `p4_tasks.jsonl`: final VEH task banks.
- `data/veh/splits/`: dev, in-distribution test, and OOD test split manifests.
- `data/circuit/p1_tasks.jsonl` through `p4_tasks.jsonl`: final circuit audit task banks.
- `data/manifests/release_manifest.json`: counts, byte sizes, SHA256 hashes, and release provenance.
- `code/src/diagbench/physics/`: closed-form VEH oracle and reference solver.
- `code/src/diagbench/domains/circuit/`: closed-form circuit oracle and public circuit builders/evaluator.
- `code/src/diagbench/evaluation/`: P1-P4 scorers used by the paper.
- `code/scripts/`: public build, evaluation, profile-extraction, and CMA-ES calibration scripts.
- `prompts/`: P1-P4 prompts and controlled-prompt audit templates.
- `results/model_outputs/`: anonymized raw JSONL outputs for the paper snapshot.
- `results/analysis/`: paper-facing summaries, generated tables, and profile scores.
- `docs/`: datasheet, artifact statement, construction audit, and reproduction guide.

VEH tasks are generated from a curated VEH bundle and scored by `diagbench.physics.oracle`. There is no separate `domains/veh` package: VEH is the main physics domain of the benchmark, while `domains/circuit` is the secondary cross-domain audit.

## Reproducing Scores

```bash
python scripts/validate_release.py
python -m pytest -q tests
PYTHONPATH=code/src python code/scripts/quantify_response_control_profiles.py --help
PYTHONPATH=code/src python code/scripts/evaluate_circuit_pilot.py --help
```

Provider API runners and API keys are intentionally excluded. To reproduce model calls, use the prompt templates in `prompts/`, save model responses as JSONL in the released output format, and score them with the public evaluators. The included raw JSONL logs preserve the reported paper snapshot because closed API model behavior may drift over time.

## Metadata and Licenses

- Code: MIT License. See `LICENSE-CODE`.
- Data, prompts, logs, and documentation: CC-BY-4.0. See `LICENSE-DATA`.
- Citation metadata: `CITATION.cff`.
- MLCommons Croissant metadata: `croissant.json`.

## Intended Use

DiagBench is intended for research on engineering-agent diagnostics, verifier-gated workflow evaluation, and response-control profile analysis. It is not intended for production safety certification or unsupervised engineering deployment.

## Contact

Anonymous during review.
