# Xinchen Geng baseline live predictions — round 1399

Current-round prediction handoff. Pass [XinchenGeng_baseline_live_round1399.csv](XinchenGeng_baseline_live_round1399.csv) to the submission owner.
It contains **7,106** predictions for Numerai **v6.0**
round **1399**, with columns `numerai_ticker,prediction` and average
percentile ranks in `[0,1]`. Every official live ticker is covered exactly once.

## Recorded round and deadlines

- Generation: **2026-11-16 14:46:45 PST** (America/Los_Angeles).
- API data datestamp: `20261113`.
- General submission deadline (`closeTime`): **2026-11-17 04:00:00 PST** (America/Los_Angeles).
- Staking deadline (`closeStakingTime`): **2026-11-15 08:00:00 PST** (America/Los_Angeles).
- General submission window recorded at generation: **open**.
- API current round before/after generation: **1399 / 1399**.
- API current round at validation: **1399**.

The two deadlines are separate API fields. This package does not extend either
deadline. Check the current round and submission window before uploading; this
CSV is specific to round 1399. Packaging makes no additional API request.

## Model source and checks

The saved `B0_small_main` model comes from `G:\DSCI-550\Project2\results\xinchen_baseline_20261116_round1399_latest`. Its historical
training run completed **2026-11-16 14:46:06 PST** (America/Los_Angeles); training used
**200,000** sampled rows and **100**
features. The live generation stage reused that saved model and recorded
`training_performed=false`. The source run, frozen protocol and exact model
identity are included; the source run's actual timestamps describe when it was fitted.

Model SHA256: `17719df416662680e7c01aca30ab210265feb1d1ad9ded94b272eb0ac4f3db56`.

Independent validation passed for all **7,106**
predictions, including saved-model inference, independent average ranks, feature
order, exact ticker coverage and artifact hashes. Maximum absolute rank error:
`2.22e-16`.

**No Numerai account submission or staking was performed. No submission receipt
is included.** This script creates local files only and does not upload to GitHub.

## Contents and reproduction

- `XinchenGeng_baseline_live_round1399.csv`: prediction CSV for this round.
- `B0_small_main.txt`, `baseline_run.json`, `baseline_protocol.json`: saved model and source identity.
- `features.json`, `live_data_manifest.json`, `live_generation.json`: exact official-data and generation records.
- `validation.json`: original independent audit, preserved unchanged.
- `audit_data/live.parquet`: exact downloaded live data for offline checking.
- `generate_xinchen_live.py`, `validate_xinchen_live.py`, `package_xinchen_live.py`, `requirements.txt`: reproduction tools.
- `HANDOFF_MANIFEST.json`: byte counts and SHA256 for every delivered file except the manifest itself.

Install the pinned requirements in a fresh Python environment. Then, from the
extracted handoff directory, independently check the complete CSV:

```powershell
python validate_xinchen_live.py --bundle . --live audit_data/live.parquet --report validation_recheck.json
```

For a current-round API check, add `--check-current-round`. That check fails once
the round changes. Choose a fresh report path to preserve the original audit.

Generate a future current round from the included saved model using fresh paths:

```powershell
python generate_xinchen_live.py --baseline . --output ../baseline_live_next --data-dir ../baseline_live_next_data
```

Source machine paths retained in the JSON records are provenance. Offline checks
use the files bundled here. Official submission format:
https://docs.numer.ai/numerai-tournament/submissions
