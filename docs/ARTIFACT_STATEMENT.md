# Artifact Statement

This anonymous artifact supports the paper's benchmark claims. It contains task banks, oracle/evaluator code, prompt templates, anonymized raw outputs, and aggregate analysis tables. The review URL is `https://anonymous.4open.science/r/diagbench-734D/`.

## License

Code is MIT licensed. Data, prompts, manifests, and anonymized results are CC-BY-4.0 licensed.

## Metadata

`croissant.json` and `data/manifests/release_manifest.json` describe file-level metadata, counts, and SHA256 checksums.

## Intended Use

The artifact is intended for research on engineering-agent evaluation and diagnostic benchmarking.

## Limitations

DiagBench is not a production safety benchmark and should not be used as a substitute for physical validation or professional engineering review.

Exact score reproduction for closed API models may drift as providers update model snapshots. The released prompts, raw JSONL logs, run manifests, and task-bank hashes preserve the reported paper snapshot.

## De-anonymization

Repository ownership, author metadata, and the private model-output mapping will be disclosed after review.
