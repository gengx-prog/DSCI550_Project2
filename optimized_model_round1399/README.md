# Numerai round 1399 optimized candidate

**Tournament upload verified: JACKIE_GENG, round 1399, Late.** The user explicitly chose `JACKIE_GENG`. The website confirmed `Optimized_live_round1399.csv`, submitted **November 16, 2026 at 20:28 PST**; the public API returned new submission ID `a47f8a97-130f-4dfe-a5e2-5e24cc873f06` and accepted **7,106 / 7,106** tickers. See [current submission status](results/model_optimization_20261116_round1399/CURRENT_SUBMISSION_STATUS.md), [combined receipt](output/numerai_optimized_submission_20261116_round1399/submission_status.json), and [original public response](output/numerai_optimized_submission_20261116_round1399/public_after_upload/response_body.json). **Late does not mean on time; live scores remain unavailable.** No staking or funds action occurred, and selection of this late upload for staking scores was not independently verified.

This directory publishes the candidate completed on November 16, 2026: **350 numeric features, up to 1.2 million training rows per member, and a regularized LightGBM rank ensemble with seeds 7, 19 and 43**. Each member uses 800 trees, learning rate 0.03 and L2 regularization 10. Training respects a strict 65-weekday label-maturity embargo.

| Reused historical development, January 2023 to August 14, 2026 | Mean NCORR |
| --- | ---: |
| Fixed local B0 baseline | 0.010132 |
| Earlier 600k-row ensemble O1 | 0.026297 |
| New regularized candidate O3 | 0.027082 |

O3 improves mean NCORR over O1 by approximately **2.99%**, but NCORR mean/std falls from 1.601 to 1.537, additive cumulative-score drawdown rises from 0.057755 to 0.069851, and 2025 performance regresses. The paired difference interval includes zero. **This is a candidate for forward validation, not a proven stable replacement.** These are reused historical scores, not live tournament results or investment returns. The screenshot model `ZEXUGAO` has not been matched to the local B0 artifact.

- [Round 1399 CSV: 7,106 tickers](output/optimized_live_round1399/Optimized_live_round1399.csv)
- [Portable final model bundle](output/optimized_live_round1399/model/bundle.json)
- [Full Chinese optimization report](results/model_optimization_20261116_round1399/OPTIMIZATION_REPORT_ZH.md)
- [90 passing project tests](results/model_optimization_20261116_round1399/tests.log)
- [Independent historical and live audit](results/model_optimization_20261116_round1399/validation.json)
- [Complete handoff ZIP](../downloads/Numerai_optimized_model_round1399.zip) and [ZIP hash receipt](../downloads/Numerai_optimized_model_round1399.zip.json)

The CSV is specific to round 1399, whose actual live date is November 13, 2026. Final training dates are strictly before August 14; the latest eligible training date is August 7. Generation records describe the historical state when no account upload had yet occurred. The later verified website upload is documented separately above; the ZIP and generation-time report retain their original pre-upload state.

The directory includes six new optimization entry points plus a public submission-check utility, required helper scripts, the complete tests, native final models, public live data, predictions, metrics and provenance hashes. Multi-GB historical data and annual fold models are omitted from Git. The 97 MB historical row-level prediction CSV is available inside the ZIP. Full historical reproduction requires downloading data and rerunning the experiment. Existing baseline directories and Git history are preserved.

Run from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Full experiment and weekly-update commands are in the Chinese report. `scripts/refit_optimized_model.py` refits only the frozen CV winner; `scripts/generate_optimized_live.py` checks the current round and creates the corresponding CSV. Runtime receipts preserve original absolute paths, so use CLI arguments to supply paths after moving to another machine.

The public checker stores its unmodified JSON response body and a summary. Its `account_upload_performed=false` only states that the checker did not upload; the combined receipt records the separate website action. Before/after responses show the old and new submission IDs.
