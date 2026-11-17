"""Statistics for the prespecified paper-inspired Numerai experiment.

The stability-aware selection score is an engineering criterion, not the
training objective of V-REx. Bootstrap intervals describe paired historical
score differences; they do not correct for model selection or imply live gains.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from run_experiments import summarize_scores

CANDIDATES = (
    "P0_small_200k", "P1_medium_200k", "P2_medium_600k",
    "P3_recency_600k", "P4_xgboost", "P5_catboost",
    "P6_seed_ensemble", "P7_heterogeneous",
)
REQUIRED_COLUMNS = {"phase", "fold", "date", "model", "NCORR", "CORR", "n_rows"}
KEYS = ["fold", "date"]


def _validated(perdate: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(perdate.columns)
    if missing:
        raise ValueError(f"Missing per-date columns: {sorted(missing)}")
    if perdate.empty:
        raise ValueError("No per-date scores")
    frame = perdate.copy()
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Missing per-date keys, scores, or row counts")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any():
        raise ValueError("Invalid date key")
    if frame.duplicated(["phase", "fold", "date", "model"]).any():
        raise ValueError("Duplicate phase/fold/date/model scores")
    if not np.isfinite(frame[["NCORR", "CORR", "n_rows"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite score or row count")
    if (frame.n_rows <= 0).any() or not np.equal(frame.n_rows, np.floor(frame.n_rows)).all():
        raise ValueError("Row counts must be positive integers")
    if frame[["NCORR", "CORR"]].abs().gt(1 + 1e-10).any().any():
        raise ValueError("Correlation outside [-1,1]")
    return frame


def summarize(perdate: pd.DataFrame) -> pd.DataFrame:
    """Return equal-date metric summaries plus variation of fold means."""
    frame = _validated(perdate)
    records: list[dict[str, Any]] = []
    for (phase, model), group in frame.groupby(["phase", "model"], sort=True):
        group = group.sort_values(["date", "fold"])
        for metric in ["NCORR", "CORR"]:
            fold_means = group.groupby("fold", sort=True)[metric].mean()
            records.append({
                "phase": phase, "model": model, "metric": metric,
                "n_folds": int(len(fold_means)),
                "n_dates": int(len(group)),
                **summarize_scores(group[metric]),
                "fold_mean_std_ddof0": float(fold_means.std(ddof=0)),
                "worst_fold_mean": float(fold_means.min()),
                "fold_means_json": json.dumps({str(k): float(v) for k, v in fold_means.items()}, sort_keys=True),
            })
    return pd.DataFrame.from_records(records)


def select_candidate(cvperdate: pd.DataFrame) -> dict:
    """Select only from four prespecified CV folds, never development scores."""
    frame = _validated(cvperdate)
    if set(frame.phase.unique()) != {"cv"}:
        raise ValueError("Selection accepts phase=cv only; development data is forbidden")
    if set(frame.model.unique()) != set(CANDIDATES):
        raise ValueError("Selection requires exactly the eight prespecified candidate IDs")
    if frame.fold.nunique() != 4:
        raise ValueError("Selection requires all four chronological CV folds")
    # Every model must be scored on the same complete folds and populations.
    grid = frame.pivot(index=KEYS, columns="model", values="NCORR")
    if grid.isna().any().any() or set(grid.columns) != set(CANDIDATES):
        raise ValueError("Candidate scoring dates/folds differ")
    counts = frame.pivot(index=KEYS, columns="model", values="n_rows")
    if not counts.eq(counts.iloc[:, 0], axis=0).all().all():
        raise ValueError("Candidate scoring populations differ")
    # A date cannot appear twice in separate folds in this chronological design.
    if pd.Index(grid.index.get_level_values("date")).duplicated().any():
        raise ValueError("Chronological validation folds overlap on dates")
    records = {}
    for model in CANDIDATES:
        rows = frame.loc[frame.model.eq(model)]
        fold_means = rows.groupby("fold", sort=True).NCORR.mean()
        mean = float(rows.NCORR.mean())
        fold_std = float(fold_means.std(ddof=0))
        penalty = 0.5 * fold_std
        records[model] = {
            "mean_NCORR_equal_date": mean,
            "fold_mean_std_ddof0": fold_std,
            "stability_penalty": penalty,
            "selection_score": mean - penalty,
            "worst_fold_mean": float(fold_means.min()),
            "fold_mean_NCORR": {str(k): float(v) for k, v in fold_means.items()},
            "n_dates": int(len(rows)),
        }
    winner = sorted(CANDIDATES, key=lambda name: (-records[name]["selection_score"], name))[0]
    return {
        "selected_model": winner,
        "selection_phase": "cv",
        "metric": "NCORR",
        "criterion": "equal-date mean NCORR - 0.5 * population std of four fold mean NCORRs",
        "penalty_coefficient": 0.5,
        "tie_break": "alphabetical candidate ID",
        "n_folds": 4,
        "n_candidates": len(CANDIDATES),
        "candidates": records,
        "development_used_for_selection": False,
        "method_scope": "Stability-aware validation selection inspired by REx; not V-REx training",
    }


def _circular_block_indices(n: int, block_length: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Sample consecutive circular blocks; truncate each replicate to n dates."""
    if min(n, block_length, reps) < 1:
        raise ValueError("n, block_length, and reps must be positive")
    n_blocks = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=(reps, n_blocks))
    offsets = np.arange(block_length)
    return ((starts[:, :, None] + offsets) % n).reshape(reps, -1)[:, :n]


def paired_block_bootstrap(
    perdate: pd.DataFrame,
    selected: str,
    baseline: str = "P1_medium_200k",
    reps: int = 2000,
    seed: int = 20261026,
) -> dict:
    """Diagnostic paired circular bootstrap on reused development dates only."""
    if reps < 1:
        raise ValueError("reps must be positive")
    frame = _validated(perdate)
    if set(frame.phase.unique()) != {"reused_development"}:
        raise ValueError("Bootstrap accepts phase=reused_development only")
    selected_rows = frame.loc[frame.model.eq(selected), KEYS + ["NCORR", "n_rows"]]
    baseline_rows = frame.loc[frame.model.eq(baseline), KEYS + ["NCORR", "n_rows"]]
    if selected_rows.empty or baseline_rows.empty:
        raise ValueError("Selected model or baseline is absent")
    paired = selected_rows.merge(baseline_rows, on=KEYS, how="outer", validate="one_to_one",
                                 suffixes=("_selected", "_baseline"), indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("Selected model and baseline date/fold keys differ")
    if not paired.n_rows_selected.eq(paired.n_rows_baseline).all():
        raise ValueError("Selected model and baseline populations differ")
    if paired.date.duplicated().any():
        raise ValueError("Development dates overlap across refit folds")
    paired = paired.sort_values("date")
    differences = (paired.NCORR_selected - paired.NCORR_baseline).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    results = {}
    for length in [13, 26]:
        indices = _circular_block_indices(len(differences), length, reps, rng)
        means = differences[indices].mean(axis=1)
        lower, upper = np.percentile(means, [2.5, 97.5])
        results[str(length)] = {
            "block_length_dates": length,
            "n_blocks_per_replicate": math.ceil(len(differences) / length),
            "replicates": reps,
            "ci95_low": float(lower),
            "ci95_high": float(upper),
            "bootstrap_mean": float(means.mean()),
        }
    return {
        "phase": "reused_development",
        "selected_model": selected,
        "baseline_model": baseline,
        "observed_mean_NCORR_delta": float(differences.mean()),
        "n_dates": int(len(paired)),
        "date_min": paired.date.min().strftime("%Y-%m-%d"),
        "date_max": paired.date.max().strftime("%Y-%m-%d"),
        "seed": seed,
        "replicates": reps,
        "method": "paired circular moving-block bootstrap of chronological per-date NCORR differences",
        "primary_block_length_dates": 13,
        "sensitivity_block_length_dates": 26,
        "interval": "2.5th and 97.5th percentiles of resampled mean differences",
        "block_bootstrap": results,
        "refit_folds": [str(x) for x in paired.fold.drop_duplicates()],
        "qualifications": [
            "The development period was observed in the earlier pilot; these intervals are diagnostic historical summaries.",
            "Paired date differences preserve common scoring populations; individual stock rows are not independent bootstrap observations.",
            "Circular blocks may cross annual model-refit boundaries and wrap the last date to the first; structural breaks and nonstationarity are not modeled.",
            "Overlapping 60-day labels motivate contiguous blocks; results depend on the prespecified block length and effective historical sample size.",
            "The interval conditions on the frozen selected model and does not adjust for candidate selection or earlier adaptive development.",
            "An interval excluding zero is not confirmatory postselection significance or evidence of future live superiority.",
        ],
    }
