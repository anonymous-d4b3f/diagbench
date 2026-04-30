# Circuit Pilot v2 Detailed P1-P4 Scores

## Detailed P1-P4 Score Tables

The following tables report every probe-level metric emitted by `CircuitPilotEvaluator`, rather than only the headline metrics. `NA` means the metric is structurally undefined for that run, for example no feasible-to-feasible preservation transition occurred.

### P1 Action Triage

| Model | n | Acc | Macro-F1 | Action entropy | Spurious propose | Unsafe propose | Request recall | Infeasible recall | Feasible-narrow refusal | Parse error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_A | 16 | 1.000 | 1.000 | 0.750 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| model_B | 16 | 0.875 | 0.874 | 0.739 | 0.062 | 0.000 | 0.750 | 1.000 | 0.000 | 0.000 |
| model_C | 16 | 0.750 | 0.739 | 0.703 | 0.125 | 0.000 | 0.500 | 1.000 | 0.250 | 0.000 |
| model_D | 16 | 1.000 | 1.000 | 0.750 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| model_M | 16 | 1.000 | 1.000 | 0.750 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| model_E | 16 | 0.812 | 0.747 | 0.561 | 0.188 | 0.000 | 0.250 | 1.000 | 0.000 | 0.000 |

### P2 Iterative Repair

| Model | n | Final feasible | Objective | Violation reduction | Directed repair | Feasibility preservation | Mean log edit delta | No-op | Over-edit | Queries | Parse error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_A | 16 | 0.750 | 0.297 | 0.828 | 0.891 | NA | 0.865 | 0.000 | 0.417 | 2.250 | 0.000 |
| model_B | 16 | 1.000 | 0.553 | 1.000 | 1.000 | NA | 2.125 | 0.000 | 1.000 | 1.000 | 0.000 |
| model_C | 16 | 0.812 | 0.343 | 0.875 | 0.891 | NA | 0.898 | 0.069 | 0.345 | 1.812 | 0.000 |
| model_D | 16 | 1.000 | 0.482 | 0.891 | 0.859 | NA | 0.926 | 0.000 | 0.448 | 1.812 | 0.000 |
| model_M | 16 | 0.812 | 0.393 | 0.828 | 0.875 | 0.000 | 1.082 | 0.000 | 0.594 | 2.000 | 0.000 |
| model_E | 16 | 0.938 | 0.443 | 0.953 | 0.969 | NA | 1.413 | 0.000 | 0.727 | 1.375 | 0.000 |

### P3 Corrupted-State Recovery

| Model | n | Escape | Explicit replan | Reset history | Cascade | Dead budget | Final success | Recovery quality | Violation reduction | Raw-history delta | Parse error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_A | 18 | 0.944 | 0.000 | 0.000 | 0.471 | 0.000 | 0.778 | 0.366 | 0.944 | NA | 0.000 |
| model_B | 18 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.611 | 1.000 | NA | 0.000 |
| model_C | 18 | 0.944 | 0.111 | 0.111 | 0.176 | 0.056 | 0.778 | 0.401 | 0.944 | NA | 0.000 |
| model_D | 18 | 0.778 | 0.000 | 0.000 | 0.000 | 0.000 | 0.722 | 0.314 | 0.944 | NA | 0.000 |
| model_M | 18 | 0.889 | 0.000 | 0.000 | 0.062 | 0.000 | 0.778 | 0.364 | 0.944 | NA | 0.000 |
| model_E | 18 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.568 | 1.000 | NA | 0.000 |

### P4 Policy-Conditioned Ranking

| Model | n | Kendall tau | Tau scaled | Exact match | Top1 | Top2 set | Pairwise acc | Policy-flip acc | BARS | Parse error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_A | 24 | 0.608 | 0.804 | 0.375 | 0.583 | 0.583 | 0.804 | 0.769 | 0.710 | 0.000 |
| model_B | 24 | 0.475 | 0.738 | 0.250 | 0.500 | 0.500 | 0.738 | 0.679 | 0.625 | 0.000 |
| model_C | 24 | 0.625 | 0.812 | 0.375 | 0.583 | 0.625 | 0.812 | 0.801 | 0.722 | 0.000 |
| model_D | 24 | 0.542 | 0.771 | 0.292 | 0.583 | 0.500 | 0.771 | 0.716 | 0.661 | 0.000 |
| model_M | 24 | 0.492 | 0.746 | 0.208 | 0.458 | 0.417 | 0.746 | 0.647 | 0.614 | 0.000 |
| model_E | 24 | 0.608 | 0.804 | 0.375 | 0.542 | 0.500 | 0.804 | 0.759 | 0.707 | 0.000 |
