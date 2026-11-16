# XinchenGeng_1116baseline

Xinchen Geng's Baseline & Validation contribution for **2026-11-16** (America/Los_Angeles), using complete official **v6.0 round 1399** data. This is a local upload-ready archive; creating it performs no GitHub upload or Numerai account submission.

## Fixed baseline and historical results

`B0_small_main` uses 100 small numeric features, `target_jupiter_60`, a fixed 200,000-row training sample and seed 7. Parameters and the chronological validation protocol match the previous run. No validation-based model selection, tuning, early stopping or refitting was performed.

| Period | Actual dates | Dates | Rows | Mean CORR | CORR std (ddof=0) | Mean NCORR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tuning | 2019-04-05 to 2022-09-30 | 183 | 1,211,007 | 0.012133 | 0.011770 | 0.013674 |
| holdout | 2023-01-06 to 2026-08-14 | 189 | 1,275,804 | 0.009576 | 0.011736 | 0.010132 |

CORR and NCORR give equal weight to each observation date. The historical holdout includes dates viewed in prior experiments. These results are historical development and reproducibility evidence; they do not establish performance on a fresh unseen test set or in live rounds. Cumulative CORR is a score diagnostic, not an investment return.

## Changes and verification

Compared with `xinchen_baseline_20261107_round1394_latest`, the snapshot adds 7,119 prediction rows and removes 0; 2,479,692 rows are shared. Newly scored dates: **2026-08-14**. Raw validation dates extend to **2026-11-06**; mature eligible labels end on **2026-08-14**. The saved model hash is unchanged. Newly available dates alone do not establish stable improvement.

Both independent audits passed. See [all-date validation](results/xinchen_baseline_20261116_round1399_latest/validation.json), [refresh validation](results/xinchen_baseline_20261116_round1399_latest/freshness_validation.json), [frozen protocol](results/xinchen_baseline_20261116_round1399_latest/protocol.json), [metrics](results/xinchen_baseline_20261116_round1399_latest/metrics.csv), [data comparison](results/xinchen_baseline_20261116_round1399_latest/data_update_comparison.json), [annual CORR](results/xinchen_baseline_20261116_round1399_latest/annual_corr.csv) and [negative-score periods](results/xinchen_baseline_20261116_round1399_latest/failure_periods.csv).

The [PDF report](report/DSCI550_Baseline_20261116_Round1399.pdf) and [report sources](report/report_sources.json) accompany the evidence. [Data provenance](results/xinchen_baseline_20261116_round1399_latest/data_provenance/) contains the original raw-file manifest, feature metadata, cleaning/split records and exact frozen source copies. `ARCHIVE_MANIFEST.json` inventories the bytes of every packaged file except itself. Original audit paths and timestamps are preserved as provenance.

## Reproduce locally

Run these commands from this archive's root using Python 3.13 and a fresh virtual environment. Downloads are public and require roughly 8.3 GB plus room for processed data, models and predictions. A historical round must remain available from the official API; matching the recorded raw hashes is required for exact reproduction.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
$snapshot = Join-Path (Get-Location).Path "snapshots/reproduction_round1399"
$result = Join-Path (Get-Location).Path "results/reproduction_round1399"
& $python scripts/refresh_xinchen_snapshot.py --snapshot $snapshot --round 1399
& $python "$snapshot/scripts/prepare_data.py"
& $python "$snapshot/scripts/run_xinchen_baseline.py" --root $snapshot --output $result
& $python "$snapshot/scripts/validate_xinchen_baseline.py" --root $snapshot --output $result
& $python "$snapshot/scripts/plot_xinchen_baseline.py" --output $result
& $python "$snapshot/scripts/summarize_xinchen_baseline.py" --output $result
& $python -m unittest discover -s tests -v
```

The included tests cover cleaning, scoring and live prediction artifacts (`test_cleaning.py`, `test_metrics.py`, `test_live_generation.py` and `test_live_submission.py`). The project's separate research-model experiments and their tests are outside this baseline archive.

Regenerating the PDF with `scripts/build_latest_baseline_report.py` additionally requires the pinned ReportLab dependency: install `requirements-report.txt`. The original frozen `requirements.txt` is preserved unchanged.

Raw/processed Parquet, trained models and row-level predictions are retained locally and excluded from this lightweight archive. Full independent validation needs the matching raw data, trained model and predictions generated locally. The recorded refresh comparison additionally needs the previous snapshot and previous full local results. Live CSV generation and current-round submission readiness are separate from this historical report.
