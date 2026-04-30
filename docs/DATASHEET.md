# DiagBench Datasheet

## Motivation

- Purpose: diagnostic evaluation of LLM-based engineering design agents.
- Intended use: research on workflow-level decision failures in physical engineering design.
- Not intended for: production safety certification or standalone engineering design without expert review.

## Composition

- VEH: 763 tasks derived from published piezoelectric cantilever energy-harvesting literature.
- Circuit: 106 closed-form circuit tasks used as a construct-validity pilot.
- No personally identifiable information is included.

## Collection Process

- VEH tasks are derived from paper-anchored specifications, normalized into SI-compatible engineering fields, and checked by a deterministic analytical oracle.
- Circuit tasks are synthetically generated from closed-form formulas with deterministic seeds.
- P1 labels are oracle-computed; P2-P4 scores are verifier-computed.

## Preprocessing

- Units are normalized before scoring.
- Split assignment is recorded in manifests.
- Each released task bank has a SHA256 checksum in `data/manifests/release_manifest.json`.

## Uses

- Recommended: evaluating LLM agents on triage, repair, corrupted-state recovery, and policy-conditioned ranking.
- Not recommended: claiming real-world engineering safety or replacing domain expert review.

## Distribution

- Data license: CC-BY-4.0.
- Code license: MIT.
- Anonymous review release: `[anonymous URL]`.

## Maintenance

Releases are versioned. Future changes must preserve task bank checksums for prior versions or publish a new semantic version with a migration note.
