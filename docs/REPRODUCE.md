# Reproducing DiagBench

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Validate the Artifact

```bash
python scripts/validate_release.py
pytest -q tests
```

## Data

Task banks are under `data/veh` and `data/circuit`. Split manifests and SHA256 checksums are under `data/manifests` and `data/veh/splits`.

## Scoring

The released package includes the analytical VEH oracle, circuit oracle, P1-P4 evaluators, response-control profile extraction code, and the CMA-ES baseline script used for calibration.

## API Runs

Provider API scripts and API keys are not included. To reproduce model calls, use the prompts in `prompts/` and score generated JSONL outputs with the public evaluators.
