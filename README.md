# DiagBench

A workflow-structured benchmark for evaluating engineering design agents under verifier-gated physical constraints.

This anonymous review artifact contains 763 VEH tasks and a paper-matched 106-task circuit pilot across four probes.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/validate_release.py
```

## Dataset Summary

| Probe | VEH Tasks | Circuit Tasks | Headline Metric |
|---|---:|---:|---|
| P1 | 240 | 32 | P1-Composite |
| P2 | 208 | 32 | Final Feasible Power Ratio |
| P3 | 156 | 18 | P3-Success |
| P4 | 159 | 24 | Full Kendall Tau |

The circuit pilot follows the current paper configuration: P1/P2 use the v3.1 hardened task bank; P3/P4 use the original circuit pilot task banks.

## Layout

- `data/`: released task banks and split manifests.
- `code/`: minimal Python package source and public evaluation/build scripts.
- `results/`: anonymized raw model outputs and paper-facing summaries.
- `prompts/`: prompt templates used by the evaluation harness.
- `docs/`: datasheet, reproduction guide, audit notes, and artifact statement.
- `tests/`: lightweight oracle/evaluator smoke tests.

## License

- Data: CC-BY-4.0
- Code: MIT

## Citation

See `CITATION.cff`.

## Contact

Anonymous during review.
