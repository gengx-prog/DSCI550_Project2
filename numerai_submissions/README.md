# Numerai Classic website submissions

Seven prediction CSVs were uploaded directly to the competition website from November 16 through November 22, 2026. Times below are in Los Angeles. Each receipt records the dashboard status at upload.

| Round | Uploaded (PST) | Model slot | Method | Accepted tickers | Status | Files |
|---|---|---|---|---:|---|---|
| 1399 | 11-16 15:05 | `JACKIE_GENG` | Fixed B0 baseline | 7,106 / 7,106 | Late | [CSV](../XinchenGeng_baseline_live_round1399/XinchenGeng_baseline_live_round1399.csv) / [Receipt](round1399/baseline_receipt.json) |
| 1399 | 11-16 20:28 | `JACKIE_GENG` | O3 regularized LightGBM ensemble | 7,106 / 7,106 | Late | [CSV](../optimized_model_round1399/output/optimized_live_round1399/Optimized_live_round1399.csv) / [Receipt](../optimized_model_round1399/output/numerai_optimized_submission_20261116_round1399/submission_status.json) |
| 1401 | 11-18 12:26 | `JACKIE_GENG_BASELINE` | Fixed all-feature LightGBM | 7,092 / 7,092 | Late | [CSV](round1401/Fixed_baseline_round1401.csv) / [Receipt](round1401/website_observation.json) |
| 1401 | 11-18 12:28 | `JACKIE_GENG` | Equal-weight full-sample and annually retrained LightGBM blend | 7,092 / 7,092 | Late | [CSV](round1401/Improved_round1401.csv) / [Receipt](round1401/website_observation.json) |
| 1403 | 11-20 12:03 | `JACKIE_GENG` | net25 blend with a 25% multitarget network component | 7,080 / 7,080 | Late | [CSV](round1403/JACKIE_GENG_round1403.csv) / [Receipt](round1403/JACKIE_GENG_receipt.json) |
| 1404 | 11-22 02:34 | `JACKIE_GENG_BASELINE` | Fixed all-feature LightGBM | 7,075 / 7,075 | On-time | [CSV](round1404/JACKIE_GENG_BASELINE_round1404.csv) / [Receipt](round1404/JACKIE_GENG_BASELINE_receipt.json) |
| 1404 | 11-22 02:38 | `JACKIE_GENG` | trailing_tree_25 blend with a 25% trailing four-year tree component | 7,075 / 7,075 | On-time | [CSV](round1404/JACKIE_GENG_round1404.csv) / [Receipt](round1404/JACKIE_GENG_receipt.json) |

Round 1399 had two successive uploads to the same model slot. The O3 ensemble was uploaded after the B0 baseline.

For round 1401, the [public API acceptance record](round1401/pair_summary.json) confirms both submission IDs and ticker counts. The [dashboard record](round1401/website_observation.json) shows that the same two CSVs were queued for round 1402.
