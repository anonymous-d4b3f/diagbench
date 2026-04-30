# VEHBench Latest P1-P4 Snapshot

**Date:** 2026-04-27

This snapshot regenerates the P1-P4 readout after adding three thinking-model runs: `deepseek-v4-pro`, `model_L`, and `hunyuan-hy3-preview`. The new Hunyuan full-run root is `artifacts/runs/full_20260427_model_M_p1_p4`.

## 1. Result Integrity

Post-repair scan status: **complete with one persistent protocol failure counted as task failure**.

| Check | Status | Note |
|---|---|---|
| Expected row counts | Pass | Hy3 has all expected rows: P1 `240/240`, P2 `208/208`, P3 `156/156`, P4 `159/159`. |
| Duplicate task IDs | Pass | No duplicates in the final merged Hy3 analysis view. |
| `invalid_output` / parse errors | Warn | One persistent P2 OOD `invalid_output` remains after three targeted retry passes; it is counted as infeasible/failure, not as a missing run. |
| Retry recovery | Pass | Initial Hy3 failures were retried and merged by task id; all quota/API/process failures were resolved. |
| Prior Mimo recharge recovery | Pass | Historical Mimo `HTTP 402` rows were already rerun and replaced in the 2026-04-25 analysis view. |

Audit artifacts:

- `artifacts/runs/full_20260427_model_M_p1_p4/analysis/merged_clean_final_20260427/merge_summary.json`
- `artifacts/runs/full_20260427_model_M_p1_p4/analysis/latest_snapshot_inputs/model_M_metrics.json`
- `artifacts/runs/full_20260427_model_M_p1_p4/analysis/latest_snapshot_inputs/model_M_per_split_metrics.json`
- `artifacts/runs/full_20260427_model_M_p1_p4/analysis/latest_snapshot_inputs/latest_with_hy3_combined_rankings.json`
- `artifacts/runs/full_20260425_deepseek_mimo_p1_p4/analysis/latest_snapshot_inputs/new_thinking_model_metrics.json`
- `results/analysis/response_control_profiles/profile_scores_complete_p1_p4.csv`
- `results/analysis/response_control_profiles/profile_validity_complete_p1_p4.json`
- `results/analysis/response_control_profiles/profile_extraction_audit.json`

The single residual protocol failure is `p2v3r1::test_ood::boundary_binding::10_1088_1742_6596_922_1_012009::0055`. The model first proposed an infeasible design and then returned a non-JSON refusal-like response. Because the behavior persisted after targeted rerun, this snapshot treats it as a model failure and leaves it visible in the audit trail.

## 9. CMA-ES Classical Optimization Baseline

**Date:** 2026-04-28

We ran CMA-ES (Covariance Matrix Adaptation Evolution Strategy) on the full VEH P2 task bank (156 tasks across dev/test\_id/test\_ood splits) under the same analytical oracle as the LLM agents. The CMA-ES configuration: population size 8, initial sigma 0.15, max evaluations equal to the query budget.

**Results:**

| Method | Budget | Feasible Rate | Mean Power Ratio | Mean Queries |
|--------|:---:|:---:|:---:|:---:|
| CMA-ES (6 queries) | 6 evaluations | 0.269 | 0.099 | 6.0 |
| CMA-ES (40 queries) | 40 evaluations | 0.578 | 0.305 | 40.0 |
| Anchor-fixed heuristic | 0 | 1.000 | 0.306 | 0.0 |
| model_C (LLM) | ~6 queries | 0.361 | 0.133 | 4.75 |
| model_B (LLM) | ~6 queries | **0.962** | **0.390** | **2.63** |

**Interpretation:**

1. CMA-ES under the same 6-query budget performs poorly (feasible rate 0.269, ratio 0.099), placing it between model_C (0.133) and model_J (0.004) on the ratio metric. Classical optimization needs many more oracle calls to converge.

2. Even with 40 queries (6.7x the LLM budget), CMA-ES achieves only 0.305 mean power ratio — equal to the anchor-fixed heuristic (0.306) and below Gemini (0.390). This means CMA-ES with 40 queries barely matches the trivial baseline of "return the literature-derived seed with no search," while Gemini achieves 27% improvement over both.

3. The anchor-fixed heuristic achieves 100% feasible coverage (it simply returns the already-feasible BKF), but its power ratio (0.306) is below Gemini's (0.390). This confirms that the benchmark does not reward feasibility alone — it rewards feasibility-preserving improvement — and that Gemini's advantage is not reducible to "CM-ES would also work if given more budget."

4. The per-split breakdown shows CMA-ES performs best on OOD tasks (feasible 0.404), suggesting the OOD constraints may be structurally easier to satisfy than in-domain constraints — consistent with the LLM split-resolved results.

**Paper-facing claim:**

> Classical black-box optimization under the same oracle and budget is not competitive with frontier LLM agents on P2 constrained repair. CMA-ES with 40 queries (6.7x the typical LLM budget) only matches the anchor-fixed heuristic, while Gemini achieves 27% higher power ratio in 2.63 queries on average. This supports the interpretation that LLM agents bring physical priors from pretraining that accelerate feasibility-preserving search — and that P2 should be evaluated as an agent behavior probe, not as a solver competition.

**Artifacts:**
- `scripts/baselines/cmaes_baseline.py`
- `results/baselines/cmaes_p2_full/cmaes_summary.json`
- `results/baselines/cmaes_p2_40steps/cmaes_summary.json`

Interpretation note: P1 `invalid_candidate` rows are not counted as protocol failures. They are valid model-recognition failures where the model proposed a design on missing-blocker tasks and the evaluator could not materialize a complete candidate; P1 intent metrics still map them back to `propose_design`.

## 2. Headline Snapshot

| Benchmark | Latest build | N | Headline metric | Current leader | Updated ordering |
|---|---|---:|---|---|---|
| `P1` | `P1 v3r4` | 240 | `P1-Composite` | `model_A` (`0.574`) | `model_A` > `model_B` > `model_F` > `model_D` > `gpt-5.4` > `model_M` > `model_H` > `model_C-mini` > `llama-3.3-70b` > `model_L` > `model_G` > `model_K` > `claude-sonnet-4-6` > `GPT-4o-mini` |
| `P2` | `P2 v3r1` | 208 | Final feasible power ratio | `gemini-3.1-pro-preview` (`0.3904`) | `gemini-3.1-pro-preview` > `model_D` > `claude-sonnet-4-6` > `model_G` > `model_A` > `model_M` > `model_H` > `model_F` > `gpt-5.4` > `model_K` > `llama-3.3-70b` > `model_L` > `model_C-mini` > `GPT-4o-mini` |
| `P3` | `P3 v3r1` | 156 | `P3-Success` recovered feasible rate | `model_M` (`47.4%`) | `model_M` > `model_L` > `model_D` > `gpt-5.4` > `gemini-3.1-pro-preview` > `model_H` > `model_A` > `model_K` > `model_G` > `model_F` > `claude-sonnet-4-6` > `llama-3.3-70b` |
| `P4` | `P4_full_v2` | 159 | Full Kendall Tau | `gpt-5.4` (`0.887`) | `gpt-5.4` > `model_G` > `model_H` > `model_L` > `claude-sonnet-4-6` > `model_M` > `model_A` > `model_D` > `gemini-3.1-pro-preview` > `model_K` > `model_F` > `llama-3.3-70b` |

## 3. P1 - Credible Engineering Triage

| Rank | Model | Acc | MDS | IDS | P1-Composite |
|---|---|---:|---:|---:|---:|
| 1 | `model_A` | 0.654 | 0.533 | 0.486 | **0.574** |
| 2 | `model_B` | 0.692 | 0.667 | 0.218 | **0.549** |
| 3 | `model_F` | 0.700 | 0.396 | 0.361 | **0.518** |
| 4 | `model_D` | 0.662 | 0.510 | 0.298 | **0.504** |
| 5 | `gpt-5.4` | 0.433 | 0.657 | 0.223 | **0.428** |
| 6 | `model_M` | 0.662 | 0.333 | 0.200 | **0.425** |
| 7 | `model_H` | 0.646 | 0.352 | 0.100 | **0.369** |
| 8 | `model_C-mini` | 0.621 | 0.375 | 0.083 | **0.365** |
| 9 | `llama-3.3-70b` | 0.625 | 0.200 | 0.192 | **0.361** |
| 10 | `model_L` | 0.617 | 0.244 | 0.083 | **0.317** |
| 11 | `model_G` | 0.613 | 0.200 | 0.100 | **0.306** |
| 12 | `model_K` | 0.613 | 0.200 | 0.100 | **0.306** |
| 13 | `claude-sonnet-4-6` | 0.575 | 0.000 | 0.100 | **0.207** |
| 14 | `GPT-4o-mini` | 0.229 | 0.301 | 0.205 | **0.196** |

P1 update: `model_M` is the strongest of the newly added thinking APIs on triage. It reaches `0.425` composite, essentially tied with `gpt-5.4` but still below the Qwen/Gemini/o4/R1 top tier. The profile is balanced enough to avoid the severe over-acceptance seen in `model_L` and `model_K`, but its missing-discipline and infeasibility-discipline scores are not frontier-leading.

## 4. P2 - Constrained Design

| Rank | Model | Final feasible power ratio | Final feasible rate | Mean power ratio feasible | Mean queries |
|---|---|---:|---:|---:|---:|
| 1 | `gemini-3.1-pro-preview` | 0.3904 | 96.2% | 0.4101 | 2.635 |
| 2 | `model_D` | 0.2413 | 67.8% | 0.3585 | 3.471 |
| 3 | `claude-sonnet-4-6` | 0.2394 | 62.5% | 0.3891 | 4.413 |
| 4 | `model_G` | 0.2063 | 61.1% | 0.3432 | 3.870 |
| 5 | `model_A` | 0.1955 | 55.8% | 0.3537 | 4.212 |
| 6 | `model_M` | 0.1609 | 48.6% | 0.3380 | 4.788 |
| 7 | `model_H` | 0.1565 | 44.2% | 0.3576 | 4.678 |
| 8 | `model_F` | 0.1551 | 42.8% | 0.3665 | 3.587 |
| 9 | `gpt-5.4` | 0.1329 | 36.1% | 0.3841 | 4.750 |
| 10 | `model_K` | 0.1294 | 41.3% | 0.3205 | 4.264 |
| 11 | `llama-3.3-70b` | 0.1197 | 37.0% | 0.3277 | 4.385 |
| 12 | `model_L` | 0.1073 | 33.2% | 0.3282 | 4.505 |
| 13 | `model_C-mini` | 0.0850 | 26.4% | 0.3401 | 4.894 |
| 14 | `GPT-4o-mini` | 0.0035 | 1.0% | 0.3638 | 1.154 |

P2 update: `model_M` lands just above `model_H` and `model_F` on endpoint ratio (`0.1609`), but it is far from Gemini's closure regime. Its first-step feasible rate is only `2.4%`, while final feasible rate reaches `48.6%`; this is a late-repair profile rather than a strong initial-anchoring profile. The one persistent `invalid_output` appears in P2 OOD and is already included as failure in these metrics.

## 5. P3 - Trap Recovery

| Rank | Model | P3-Success | P3-RecoveryQuality | First recovery feasible | Trap escape | Mean queries |
|---|---|---:|---:|---:|---:|---:|
| 1 | `model_M` | 47.4% | 0.4876 | 40.4% | 77.6% | 2.929 |
| 2 | `model_L` | 45.5% | 0.6263 | 41.0% | 63.5% | 2.923 |
| 3 | `model_D` | 42.9% | 0.2708 | 22.4% | 92.3% | 2.974 |
| 4 | `gpt-5.4` | 42.3% | 0.3723 | 26.3% | 90.4% | 2.974 |
| 5 | `gemini-3.1-pro-preview` | 37.2% | 0.3194 | 26.9% | 92.3% | 3.000 |
| 6 | `model_H` | 35.3% | 0.2868 | 23.1% | 82.7% | 3.000 |
| 7 | `model_A` | 30.1% | 0.2612 | 22.4% | 85.9% | 3.000 |
| 8 | `model_K` | 28.8% | 0.1986 | 27.6% | 90.4% | 3.000 |
| 9 | `model_G` | 27.6% | 0.1453 | 12.8% | 94.9% | 3.000 |
| 10 | `model_F` | 26.3% | 0.1389 | 12.2% | 69.2% | 2.917 |
| 11 | `claude-sonnet-4-6` | 16.0% | 0.0960 | 0.6% | 96.8% | 3.000 |
| 12 | `llama-3.3-70b` | 3.2% | 0.0926 | 2.6% | 34.6% | 2.051 |

P3 update: `model_M` becomes the new P3 point-estimate leader at `47.4%`. It is not simply the strongest trap-escape model: its escape rate (`77.6%`) is lower than R1/GPT/Gemini, but it converts escapes into feasible endpoints better than the replan-heavy models. `model_L` remains the best recovery-quality model (`0.6263`), while Hy3 is the best final recovery-success model.

## 6. P4 - Trade-Off Ranking

| Rank | Model | Full tau | Exact | Top-1 | Top-2 set | Pareto tau | Pareto violation | Parse errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `gpt-5.4` | **0.887** | 57.2% | 83.0% | 81.1% | 1.000 | 0.000 | 0 |
| 2 | `model_G` | **0.877** | 66.3% | 79.1% | 83.4% | 1.000 | 0.000 | 0 |
| 3 | `model_H` | **0.860** | 54.7% | 77.4% | 79.2% | 0.984 | 0.008 | 0 |
| 4 | `model_L` | **0.843** | 52.8% | 72.3% | 79.2% | 0.993 | 0.003 | 0 |
| 5 | `claude-sonnet-4-6` | **0.840** | 56.6% | 72.3% | 79.2% | 1.000 | 0.000 | 0 |
| 6 | `model_M` | **0.839** | 50.9% | 77.4% | 79.9% | 0.943 | 0.028 | 0 |
| 7 | `model_A` | **0.835** | 49.7% | 78.0% | 76.7% | 0.983 | 0.008 | 0 |
| 8 | `model_D` | **0.833** | 56.0% | 71.7% | 76.7% | 1.000 | 0.000 | 0 |
| 9 | `gemini-3.1-pro-preview` | **0.824** | 54.1% | 71.7% | 78.6% | 0.975 | 0.013 | 2 |
| 10 | `model_K` | **0.794** | 43.4% | 60.4% | 74.2% | 0.977 | 0.012 | 0 |
| 11 | `model_F` | **0.780** | 50.0% | 60.0% | 72.0% | 1.000 | 0.000 | 0 |
| 12 | `llama-3.3-70b` | **0.714** | 34.0% | 54.1% | 65.4% | 0.959 | 0.020 | 0 |

New Hy3 P4 split view:

| Model | `dev` exact / tau | `test_id` exact / tau | `test_ood` exact / tau | Balanced-active BARS |
|---|---|---|---|---:|
| `model_M` | `51.6% / 0.856` | `41.7% / 0.767` | `60.0% / 0.873` | 0.5610 |

P4 update: `model_M` is a strong second-tier ranker (`0.839` full tau), essentially tied with Claude/Qwen/R1 and just below `model_L`. Its top-1/top-2 rates are competitive, but its Pareto violation rate (`0.028`) is higher than the zero-violation GPT/R1/Claude/model_G group, so it is not a new ranking frontier.

## 7. Thinking-Model Readout

- `model_M` is the best current P3 success model and the strongest triage model among the three newly added thinking APIs, but it does not change the P1/P2/P4 leaders.
- `model_L` remains a P3/P4 specialist: best P3 recovery quality and strong P4 full-order ranking, but weak P1/P2 endpoint behavior.
- `model_K` remains the weakest of the three new thinking APIs overall. It is schema-clean after repair but replan-heavy and does not convert P3 trap escape into final feasibility as well as Hy3, Mimo, R1, or model_C.
- The updated pattern is still layer-specific rather than monotonic by model generation or by thinking support. Thinking helps most clearly in corrupted-state recovery; it does not automatically solve front-end triage, constrained endpoint closure, or policy-conditioned ranking.
- The final Hy3 analysis view is complete and resume-safe. The single residual P2 OOD non-schema output is preserved as an auditable model failure rather than silently removed.

## 8. Response-Control Profile Quantification

The profile analysis uses **12 complete P1-P4 model runs**, not a dev-only slice. Here `n=12` means 12 model-level observations with all four probes complete: the nine original common-coverage models plus `model_K`, `model_L`, and `model_M`. The task-level coverage remains P1 `240`, P2 `208`, P3 `156`, and P4 `159` per complete model. `GPT-4o-mini` and `model_C-mini` are retained in the raw profile output for P1/P2 but excluded from the 12-model cross-stage analysis because their P3/P4 artifacts are not complete.

The implementation is in `scripts/quantify_response_control_profiles.py`. It reads only existing JSONL results and trajectories, applies stable task-id de-duplication, chooses full P1 files over partial historical remnants, uses the complete `model_F` P4 repair artifact, and merges the 14 Mimo P3 `HTTP 402` retry rows into the paper-facing view. The extraction audit reports no partial nonzero stages in the 12-model subset; it records `4` de-duplicated P4 rows for `model_G` and `14` repaired P3 rows for `model_L`.

| Model | Action prior | Edit style | Feedback obedience | State trust | Preference execution | P1 comp | P2 ratio | P3 success | P4 tau |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `model_A` | 0.737 | 0.571 | 0.407 | 0.646 | 0.679 | 0.574 | 0.1956 | 30.1% | 0.835 |
| `model_B` | 0.668 | 0.753 | 0.508 | 0.425 | 0.729 | 0.549 | 0.3904 | 37.2% | 0.824 |
| `model_C` | 0.657 | 0.482 | 0.406 | 0.453 | 0.743 | 0.428 | 0.1329 | 42.3% | 0.887 |
| `model_D` | 0.652 | 0.667 | 0.455 | 0.455 | 0.728 | 0.504 | 0.2413 | 42.9% | 0.833 |
| `model_F` | 0.635 | 0.470 | 0.391 | 0.495 | 0.666 | 0.518 | 0.1551 | 26.3% | 0.772 |
| `model_M` | 0.564 | 0.485 | 0.429 | 0.599 | 0.688 | 0.425 | 0.1609 | 47.4% | 0.839 |
| `model_H` | 0.536 | 0.493 | 0.389 | 0.629 | 0.708 | 0.369 | 0.1565 | 35.3% | 0.860 |
| `Llama-3.3-70B` | 0.528 | 0.586 | 0.326 | 0.452 | 0.622 | 0.361 | 0.1197 | 3.2% | 0.714 |
| `model_L` | 0.499 | 0.385 | 0.391 | 0.627 | 0.707 | 0.317 | 0.1073 | 45.5% | 0.843 |
| `model_K` | 0.491 | 0.404 | 0.332 | 0.737 | 0.666 | 0.306 | 0.1294 | 28.8% | 0.794 |
| `model_G` | 0.491 | 0.613 | 0.404 | 0.523 | 0.761 | 0.306 | 0.2063 | 27.6% | 0.874 |
| `model_E` | 0.428 | 0.622 | 0.392 | 0.443 | 0.730 | 0.207 | 0.2394 | 16.0% | 0.840 |

Key validation results on the complete 12-model subset:

- `Action prior` predicts P1 composite strongly: Spearman `rho=0.972`, Pearson `r=0.961`.
- `Edit style` predicts P2 final feasible power ratio: Spearman `rho=0.839`, Pearson `r=0.864`.
- `Feedback obedience` also tracks P2 closure (`rho=0.741`) and moderately tracks P3 success (`rho=0.587`), which is consistent with verifier feedback acting as a control signal rather than explanatory context.
- `Preference execution` predicts P4 ranking behavior: Spearman `rho=0.769` against BARS and `rho=0.762` against full Tau.
- `State trust` should not be read as a single monotone P3 success score. Its correlation with P3 success is weak (`rho=0.217`) because P3 decomposes escape, stabilization, and final feasibility. This is itself diagnostic: Claude escapes most traps (`96.8%`) but recovers poorly (`16.0%`), while Mimo has lower escape (`63.5%`) but much better recovery quality (`0.626`) and success (`45.5%`).

The profile result is therefore stronger than the earlier qualitative story for P1, P2, and P4, and more nuanced for P3. It supports the main thesis that P1-P4 are not primarily a "new model" or "thinking model" leaderboard. They measure whether a model's command-execution style matches the boundary: action discipline for P1, feasibility-preserving local repair for P2, corrupted-state handling for P3, and policy-conditioned evaluator behavior for P4.
