# Tier3 Phase A Results

Run directory: `tier3_results_20260429_193009`

This Phase A run completed the four preregistered cells with parse-clean results for C1--C3. C4 is complete but model_E has nonzero parse errors, especially under the P1 targeted prompt.

## Primary Paired Bootstrap Results

| Cell | Target | n pairs | Targeted | Neutral | Targeted - Neutral | 95% CI | Readout |
|---|---|---:|---:|---:|---:|---:|---|
| C1 | model_C P2 feasible rate | 40 | 0.150 | 0.175 | -0.025 | [-0.150, 0.100] | no targeted gain |
| C2 | model_E P3 success | 36 | 0.194 | 0.389 | -0.194 | [-0.361, -0.028] | targeted is worse than neutral |
| C3 | Gemini-3.1 P4 Kendall tau | 50 | 0.708 | 0.692 | +0.016 | [0.000, 0.048] | tiny, not robust |
| C4 | model_C P1 accuracy as retained by script | 60 | 0.467 | 0.400 | +0.067 | [-0.050, 0.183] | not significant |

Important caveat for C4: the current `tier3_analysis.py` report overwrites the first C4 model entry and retains only the last model's paired statistics, so the JSON C4 effect is model_C only. Per-model inspection shows that Claude behaves differently and is worse under the targeted prompt.

## Per-Condition Summary

| Cell | Model | Stage | Cond | n | Main metric | Secondary metric | Third metric | Parse |
|---|---|---|---:|---:|---:|---:|---:|---:|
| C1 | model_C | P2 | default | 40 | 0.150 feasible | 61.031 mean objective | 4.05 queries | 0.000 |
| C1 | model_C | P2 | neutral | 40 | 0.175 feasible | 4.335 mean objective | 4.55 queries | 0.000 |
| C1 | model_C | P2 | targeted | 40 | 0.150 feasible | 461.817 mean objective | 4.83 queries | 0.000 |
| C1 | model_C | P2 | wrong | 40 | 0.075 feasible | 2.084 mean objective | 3.93 queries | 0.000 |
| C2 | model_E | P3 | default | 36 | 0.167 success | 0.694 cascade | 8.00 queries | 0.000 |
| C2 | model_E | P3 | neutral | 36 | 0.389 success | 0.528 cascade | 8.00 queries | 0.000 |
| C2 | model_E | P3 | targeted | 36 | 0.194 success | 0.778 cascade | 8.00 queries | 0.000 |
| C2 | model_E | P3 | wrong | 36 | 0.222 success | 0.694 cascade | 8.00 queries | 0.000 |
| C2 | model_E | P3 | state-summary-plus | 36 | 0.222 success | 0.750 cascade | 8.00 queries | 0.000 |
| C3 | Gemini-3.1 | P4 | default | 50 | 0.692 tau | 0.320 exact | 0.940 top1 | 0.000 |
| C3 | Gemini-3.1 | P4 | neutral | 50 | 0.692 tau | 0.320 exact | 0.960 top1 | 0.000 |
| C3 | Gemini-3.1 | P4 | targeted | 50 | 0.708 tau | 0.320 exact | 1.000 top1 | 0.000 |
| C3 | Gemini-3.1 | P4 | wrong | 50 | 0.700 tau | 0.300 exact | 0.880 top1 | 0.000 |
| C4 | model_E | P1 | default | 60 | 0.650 accuracy | 0.050 parse | -- | 0.050 |
| C4 | model_E | P1 | neutral | 60 | 0.617 accuracy | 0.050 parse | -- | 0.050 |
| C4 | model_E | P1 | targeted | 60 | 0.567 accuracy | 0.183 parse | -- | 0.183 |
| C4 | model_E | P1 | wrong | 60 | 0.633 accuracy | 0.050 parse | -- | 0.050 |
| C4 | model_C | P1 | default | 60 | 0.433 accuracy | 0.000 parse | -- | 0.000 |
| C4 | model_C | P1 | neutral | 60 | 0.400 accuracy | 0.000 parse | -- | 0.000 |
| C4 | model_C | P1 | targeted | 60 | 0.467 accuracy | 0.000 parse | -- | 0.000 |
| C4 | model_C | P1 | wrong | 60 | 0.400 accuracy | 0.000 parse | -- | 0.000 |

## Interpretation

1. The original targeted-prompt hypothesis is not supported by Phase A. Targeted prompts do not reliably beat neutral controls.
2. C2 is the strongest result, but in the opposite direction: Claude P3 neutral control improves over default (+0.222, CI [0.083, 0.361]), while the P3 targeted prompt is worse than neutral (-0.194, CI [-0.361, -0.028]). This suggests that Claude benefits from generic structure/verification, not from the current skeptical-reset prompt wording.
3. C3 shows a small Gemini P4 gain from the evaluator prompt (+0.016 tau), but the effect is too small to carry a main claim.
4. C4 should not be reported as a single positive effect. model_C improves under P1 targeted (+0.067 over neutral), while Claude degrades (-0.050) and targeted P1 increases Claude parse errors from 5.0% to 18.3%.
5. The safer paper framing is a negative/diagnostic result: boundary-aware prompting is not automatically sufficient; prompt wording can interact with model-specific output control and can even harm the intended boundary.

## Recommendation

Do not insert Tier3 as positive evidence in the main paper yet. If used, place it in an appendix as an exploratory controlled-prompt audit showing that generic structured prompting can improve Claude P3, while targeted boundary prompts require redesign. For a stronger main-paper result, rerun a v2 Tier3 prompt design with shorter JSON-safe control text and separate model-specific prompt constraints.
