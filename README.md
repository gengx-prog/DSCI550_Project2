# DSCI550_Project2

[Competition website submissions](numerai_submissions/README.md) lists the seven CSV uploads from November 16 through November 22, 2026, with model descriptions and submission receipts.

## November 16, 2026 optimized candidate

[Optimized candidate, model and full report](optimized_model_round1399/README.md) adds a 350-feature, 1.2-million-row, three-seed regularized LightGBM candidate and a verified [round 1399 CSV](optimized_model_round1399/output/optimized_live_round1399/Optimized_live_round1399.csv). Historical development mean NCORR is 0.027082 versus 0.026297 for the earlier 600k ensemble, but stability and 2025 performance regress and the paired interval includes zero. This is a forward-validation candidate, not a proven stable replacement. All 90 project tests and the independent historical/live audit passed. The user-selected model JACKIE_GENG now has a verified round 1399 upload of Optimized_live_round1399.csv, accepted 7,106/7,106, with website status **Late** at 2026-11-16 20:28 PST. Live scores remain unavailable; see the [current receipt](optimized_model_round1399/output/numerai_optimized_submission_20261116_round1399/submission_status.json). No staking or funds action occurred. [Complete handoff ZIP](downloads/Numerai_optimized_model_round1399.zip).

## November 16, 2026 baseline update

The fixed `B0_small_main` baseline was retrained and independently audited on
official v6.0 round **1399** data with its original parameters. The live CSV has
**7,106** predictions and passed a full independent saved-model,
rank and ticker-coverage check. The [website receipt](numerai_submissions/round1399/baseline_receipt.json) records the subsequent baseline upload on November 16 at 15:05 PST.

- [Round 1399 prediction CSV](XinchenGeng_baseline_live_round1399/XinchenGeng_baseline_live_round1399.csv)
- [Live model, audit and reproduction](XinchenGeng_baseline_live_round1399/README.md)
- [Complete live handoff ZIP](downloads/XinchenGeng_baseline_live_round1399_handoff.zip)
- [November 16 historical baseline and report](XinchenGeng_1116baseline/README.md)
- [Historical baseline ZIP](downloads/XinchenGeng_1116baseline.zip)

## Baseline predictions for round 1395

The requested round 1395 CSV contains **7,118** predictions from the unchanged baseline and passed full independent model, ticker-coverage and ranking checks. Round 1395 had already closed when generated; the API reported round 1396. No Numerai account submission was performed.

- [Prediction CSV](XinchenGeng_baseline_live_round1395/XinchenGeng_baseline_live_round1395.csv)
- [Round status and audit records](XinchenGeng_baseline_live_round1395/README.md)
- [Independent validation](XinchenGeng_baseline_live_round1395/validation.json)

## Baseline live predictions - round 1394

Generated November 7, 2026 (America/Los_Angeles), using the saved fixed baseline. The CSV contains 7,121 predictions for Numerai v6.0 round 1394 and passed full independent model, ticker-coverage and ranking checks.

- [Prediction CSV](XinchenGeng_baseline_live_round1394/XinchenGeng_baseline_live_round1394.csv)
- [Complete handoff ZIP](downloads/XinchenGeng_baseline_live_round1394_handoff.zip)
- [Model, generation/validation scripts and instructions](XinchenGeng_baseline_live_round1394/README.md)
- [Independent validation record](XinchenGeng_baseline_live_round1394/validation.json)

The files are prepared for the submission owner; no Numerai account submission was performed. This CSV is specific to round 1394. Check the current round before submitting, and regenerate for a new round using the included script.

## Xinchen Geng baseline archives

| Archive | Source version | Scored dates |
| --- | --- | ---: |
| [XinchenGeng_1030baseline](XinchenGeng_1030baseline/README.md) | October 30, 2026 original baseline | 369 |
| [XinchenGeng_1106baseline](XinchenGeng_1106baseline/README.md) | Last completed refresh before November 7: round 1392, November 5 Los Angeles / November 6 UTC | 370 |

Each directory contains its corresponding source code, pinned dependencies, historical results, figures, validation records and provenance. See its README for reproduction instructions. The requested archive labels preserve the actual run timestamps in the evidence files.
