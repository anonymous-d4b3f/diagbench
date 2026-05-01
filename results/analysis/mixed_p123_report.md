# Circuit Audit P1-P3 Results

This paper-facing analysis reports the final circuit audit P1--P3 scores. P4 ranking scores are reported separately in the circuit headline tables and generated paper outputs.

## Headline Metrics

| Model | P1-Composite | P2b | P2 Feasible | P3-Success | P3 Cascade | Parse P1/P2/P3 |
|---|---|---|---|---|---|---|
| model_A | 0.971 | 0.494 | 0.906 | 0.833 | 0.310 | 0.000/0.000/0.000 |
| Hunyuan-Hy3 | 0.944 | 0.512 | 0.938 | 0.833 | 0.071 | 0.000/0.000/0.000 |
| model_D | 0.974 | 0.594 | 0.969 | 1.000 | 0.067 | 0.000/0.000/0.000 |
| Gemini-3.1 | 0.974 | 0.662 | 1.000 | 1.000 | 0.000 | 0.000/0.000/0.000 |
| model_C | 0.923 | 0.457 | 0.875 | 0.867 | 0.107 | 0.000/0.000/0.000 |
| model_E | 0.932 | 0.487 | 0.906 | 0.833 | 0.138 | 0.000/0.000/0.000 |

## Diagnostic Metrics

| Model | P1 Acc | P1 Macro-F1 | P1 Spurious | P1 Infeasible Recall | P2 Directed | P2 Over-edit | P2 Queries | P3 Escape | P3 Replan | P3 Dead |
|---|---|---|---|---|---|---|---|---|---|---|
| model_A | 0.969 | 0.965 | 0.000 | 1.000 | 0.938 | 0.400 | 1.719 | 0.967 | 0.000 | 0.000 |
| Hunyuan-Hy3 | 0.938 | 0.933 | 0.000 | 1.000 | 0.914 | 0.771 | 1.500 | 0.933 | 0.000 | 0.000 |
| model_D | 0.969 | 0.971 | 0.000 | 1.000 | 1.000 | 0.605 | 1.344 | 1.000 | 0.000 | 0.000 |
| Gemini-3.1 | 0.969 | 0.971 | 0.000 | 1.000 | 0.969 | 0.938 | 1.000 | 1.000 | 0.000 | 0.000 |
| model_C | 0.906 | 0.917 | 0.000 | 1.000 | 0.883 | 0.489 | 1.469 | 0.933 | 0.067 | 0.033 |
| model_E | 0.938 | 0.940 | 0.031 | 0.875 | 0.914 | 0.646 | 1.500 | 0.967 | 0.000 | 0.000 |

## Analysis

- P1 is no longer reported as raw accuracy only. `P1-Composite` preserves the VEH-style entry-discipline readout by combining classification quality with penalties for spurious/unsafe proposals and missed request/infeasible cases.
- Gemini-3.1 and model_D form the top P1/P3 group under this mixed audit. Both reach `P3-Success=1.000`; Gemini has zero cascade while DeepSeek still cascades on a small fraction of recovered traces.
- P2 remains the clearest differentiator. Gemini-3.1 leads on `P2b=0.662` with full feasibility, followed by model_D at `0.594`; model_C is lowest at `0.457`, indicating weaker objective-preserving repair despite parse-clean outputs.
- model_A stays strong at entry triage (`P1-Composite=0.971`) but has the largest P3 cascade rate (`0.310`), matching the state-trust interpretation: strong initial action discipline does not imply corrupted-state recovery.
- Hunyuan-Hy3 and model_E are middle-band in P2/P3. Hunyuan has lower P3 cascade than Qwen/Claude/GPT, while Claude's P2b and P3 success do not recover its earlier weak P1 tendency.
- All six runs are parse-clean on P1/P2/P3, so the observed spread is not a formatting artifact.
