# DiagBench Construction Audit Summary

This document summarizes how the released task banks were constructed and audited. It is intentionally limited to paper-facing construction evidence; historical experiment snapshots and provider-specific run logs are not part of this construction audit.

## Released Scope

| Domain | P1 | P2 | P3 | P4 | Total |
|---|---:|---:|---:|---:|---:|
| VEH | 240 | 208 | 156 | 159 | 763 |
| Circuit | 32 | 32 | 18 | 24 | 106 |

The release exposes final task banks under `data/veh/` and `data/circuit/`. Any generator tokens preserved inside `task_id` fields are opaque stable identifiers retained for log alignment and checksum stability.

## VEH Construction

The VEH task bank is grounded in published piezoelectric vibration energy harvester literature. Source papers were screened into physically usable anchors, normalized into SI-compatible fields, and converted into verifier-scored P1-P4 probes.

Construction stages:

1. Source-paper collection and field extraction.
2. Anchor screening for complete geometry, material, boundary, and excitation information.
3. Unit normalization and schema validation.
4. Closed-form oracle verification.
5. Split assignment with DOI-disjoint evaluation where applicable.
6. Probe packaging into P1 triage, P2 repair, P3 corrupted-state recovery, and P4 policy-ranking tasks.

The released `data/manifests/release_manifest.json` records final counts, byte sizes, and SHA256 hashes for each task bank.

## Circuit Construction

The circuit audit is a cross-domain construct-validity check. It preserves the P1-P4 decision boundaries while changing the physics and oracle:

- P1: entry triage with near-boundary infeasible and missing-information cases.
- P2: dual-constraint repair where fixing one violation can break another.
- P3: corrupted-state recovery under progressive traps.
- P4: policy-conditioned ranking over feasible candidate pools.

Closed-form families include RC filters, loaded dividers, LED current limiting, op-amp amplifiers, and linear regulators.

## Oracle Boundary

Physical validity is always computed outside the model:

- VEH uses `diagbench.physics.oracle.PiezoelectricOracle`.
- Circuit uses `diagbench.domains.circuit.oracle.CircuitOracle`.

The model output is parsed as an action or ranking proposal; the oracle determines feasibility, constraint violations, objective value, and the per-probe score. This prevents language-only self-certification.

## Split and Provenance Policy

Final released split manifests live under `data/veh/splits/`. Each split manifest records records and SHA256 hashes. The release uses `release_v1_final` as the public snapshot identifier.

## Non-Included Materials

The anonymous release excludes:

- provider API keys,
- live provider runners,
- local scratch extraction folders,
- private model-name mappings,
- working-repo git history,
- historical intermediate benchmark versions.

Raw anonymized model JSONL outputs are retained under `results/model_outputs/` for the paper snapshot.
