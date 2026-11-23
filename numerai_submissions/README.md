# Numerai Classic website submissions

Seven prediction CSVs were uploaded directly from November 16 through November 22, 2026. Numerai carried three of those files into rounds 1400 and 1402. The table gives each destination round's dashboard time in Los Angeles.

| Round | Submitted (PST) | Model slot | Method | Accepted tickers | Status | Files |
|---|---|---|---|---:|---|---|
| 1399 | 11-16 15:05 | `JACKIE_GENG` | Fixed B0 baseline | 7,106 / 7,106 | Late | [CSV](../XinchenGeng_baseline_live_round1399/XinchenGeng_baseline_live_round1399.csv) / [Receipt](round1399/baseline_receipt.json) |
| 1399 | 11-16 20:28 | `JACKIE_GENG` | O3 regularized LightGBM ensemble | 7,106 / 7,106 | Late | [CSV](../optimized_model_round1399/output/optimized_live_round1399/Optimized_live_round1399.csv) / [Receipt](../optimized_model_round1399/output/numerai_optimized_submission_20261116_round1399/submission_status.json) |
| 1400 | 11-17 05:15 | `JACKIE_GENG` | O3 ensemble from round 1399 | 7,106 / 7,106 | On-time | [Source CSV](../optimized_model_round1399/output/optimized_live_round1399/Optimized_live_round1399.csv) / [Receipt](round1400/competition_record.json) |
| 1400 | — | `JACKIE_GENG_BASELINE` | — | — | No submission | [Dashboard and API record](round1400/competition_record.json) |
| 1401 | 11-18 12:26 | `JACKIE_GENG_BASELINE` | Fixed all-feature LightGBM | 7,092 / 7,092 | Late | [CSV](round1401/Fixed_baseline_round1401.csv) / [Receipt](round1401/website_observation.json) |
| 1401 | 11-18 12:28 | `JACKIE_GENG` | Equal-weight full-sample and annually retrained LightGBM blend | 7,092 / 7,092 | Late | [CSV](round1401/Improved_round1401.csv) / [Receipt](round1401/website_observation.json) |
| 1402 | 11-19 05:03 | `JACKIE_GENG_BASELINE` | Fixed all-feature LightGBM from round 1401 | 7,092 / 7,092 | On-time | [Source CSV](round1401/Fixed_baseline_round1401.csv) / [Receipt](round1402/competition_record.json) |
| 1402 | 11-19 05:03 | `JACKIE_GENG` | Equal-weight LightGBM blend from round 1401 | 7,092 / 7,092 | On-time | [Source CSV](round1401/Improved_round1401.csv) / [Receipt](round1402/competition_record.json) |
| 1403 | 11-20 12:03 | `JACKIE_GENG` | net25 blend with a 25% multitarget network component | 7,080 / 7,080 | Late | [CSV](round1403/JACKIE_GENG_round1403.csv) / [Receipt](round1403/JACKIE_GENG_receipt.json) |
| 1404 | 11-22 02:34 | `JACKIE_GENG_BASELINE` | Fixed all-feature LightGBM | 7,075 / 7,075 | On-time | [CSV](round1404/JACKIE_GENG_BASELINE_round1404.csv) / [Receipt](round1404/JACKIE_GENG_BASELINE_receipt.json) |
| 1404 | 11-22 02:38 | `JACKIE_GENG` | trailing_tree_25 blend with a 25% trailing four-year tree component | 7,075 / 7,075 | On-time | [CSV](round1404/JACKIE_GENG_round1404.csv) / [Receipt](round1404/JACKIE_GENG_receipt.json) |

Round 1399 had two successive uploads to the same model slot. The O3 ensemble was uploaded after the B0 baseline.

For round 1401, the [public API acceptance record](round1401/pair_summary.json) confirms both submission IDs and ticker counts. The [dashboard record](round1401/website_observation.json) shows that the same two CSVs were queued for round 1402.

## Predictions using each round's own live data

These files are archived for comparison. The competition receipts above identify the files Numerai scored or accepted for scoring.

| Round | Live feature date | Tickers per file | Prediction files |
|---|---|---:|---|
| 1400 | 2026-11-16 | 7,106 | [Original local upgraded candidate](round1400/local_predictions/XinchenGeng_upgraded_round1400.csv) · [Reconstructed baseline](round1400/local_predictions/Fixed_baseline_round1400_retrospective.csv) |
| 1402 | 2026-11-18 | 7,086 | [Reconstructed main model](round1402/local_predictions/Improved_round1402_retrospective.csv) · [Reconstructed baseline](round1402/local_predictions/Fixed_baseline_round1402_retrospective.csv) |
