# Xinchen Geng baseline predictions for round 1395

[Prediction CSV](XinchenGeng_baseline_live_round1395.csv): **7,118** predictions from the unchanged `B0_small_main` baseline, using official Numerai v6.0 live data pinned to round **1395** (`dataDatestamp=20261109`). The columns are `numerai_ticker,prediction`.

**Round 1395 was already closed when this file was generated.** Its recorded general submission deadline was November 11, 2026 at 05:12:22 America/Los_Angeles (`2026-11-11T13:12:22Z`). The public API reported round **1396** before and after generation. This is an archive for the explicitly requested round 1395. No Numerai account submission was performed.

The saved 100-feature, 300-tree LightGBM model was reused without retraining or parameter changes. Predictions use average percentile ranks. Independent full-model inference, ranking, ticker coverage and file-hash checks passed. The maximum absolute ranking difference was `2.22e-16`.

- [Generation record](live_generation.json): model identity, round metadata, timestamps and CSV hash.
- [Independent validation](validation.json): all 7,118 predictions and exact live ticker coverage.
- [Official-data download manifest](live_data_manifest.json): round-specific source URLs, sizes and hashes.

The unchanged saved model is also retained in the [round 1394 directory](../XinchenGeng_baseline_live_round1394/B0_small_main.txt). Original machine paths in the audit records are provenance, not repository-relative paths. This directory contains the CSV and audit records; raw live data and the execution environment are retained locally.
