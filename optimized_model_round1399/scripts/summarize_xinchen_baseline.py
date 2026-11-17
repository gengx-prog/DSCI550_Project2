"""Summarize annual CORR and negative observation-date runs for the fixed baseline.

Annual boundaries are the actual observed dates, including partial years.
Failure periods join adjacent scored observations within one split, not adjacent
calendar days. Additive CORR is a score diagnostic, not an investment return or
portfolio drawdown. This script never changes the training protocol or run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


MODEL = "B0_small_main"
SPLITS = ("tuning", "holdout")
SOURCE_FILES = ("metrics.csv", "per_date_metrics.csv", "run.json", "validation.json")
TOLERANCE = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_close(actual: float, expected: float, label: str) -> None:
    require(bool(np.isclose(actual, expected, atol=TOLERANCE, rtol=0, equal_nan=True)),
            f"{label} differs from per-date scores")


def read_sources(output: Path) -> tuple[dict, pd.DataFrame]:
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    require(run.get("status") == "complete", "A completed baseline run is required")
    require(audit.get("status") == "passed", "A passed independent validation is required")
    require(run.get("model") == MODEL and run.get("primary_metric") == "CORR",
            "Only the fixed Xinchen Geng CORR baseline is supported")
    scores = pd.read_csv(output / "per_date_metrics.csv")
    summary = pd.read_csv(output / "metrics.csv")
    required_scores = {"split", "date", "model", "n_rows", "CORR", "NCORR"}
    required_summary = {"split", "model", "metric", "n_dates", "n_scored_dates", "mean",
                        "std_ddof0", "worst_date_score", "positive_fraction", "training_rows",
                        "target", "n_features", "seed"}
    require(required_scores.issubset(scores.columns), "Missing per-date score columns")
    require(required_summary.issubset(summary.columns), "Missing summary columns")
    require(set(scores["model"]) == set(summary["model"]) == {MODEL},
            "Unexpected model in input CSVs")
    require(set(scores["split"]) == set(SPLITS), "Unexpected per-date split coverage")
    require(not scores.duplicated(["split", "date"]).any(), "Duplicate split/date score")
    require(not summary.duplicated(["split", "metric"]).any(), "Duplicate summary row")
    require(set(zip(summary["split"], summary["metric"])) ==
            {(split, metric) for split in SPLITS for metric in ("CORR", "NCORR")},
            "Unexpected summary split/metric coverage")
    scores["date"] = pd.to_datetime(scores["date"], errors="raise")
    require(scores["date"].notna().all() and scores["date"].dt.tz is None,
            "Missing or timezone-aware observation dates")
    require(scores["date"].eq(scores["date"].dt.normalize()).all(),
            "Observation dates must not include times")
    require(not scores.duplicated(["split", "date"]).any(), "Duplicate parsed split/date score")
    for column in ("n_rows", "CORR", "NCORR"):
        scores[column] = pd.to_numeric(scores[column], errors="raise")
    counts = scores["n_rows"].to_numpy(dtype=float)
    require(np.isfinite(counts).all() and (counts >= 2).all() and
            np.equal(counts, np.floor(counts)).all(), "Invalid scored row counts")
    scores["n_rows"] = scores["n_rows"].astype("int64")
    require(np.isfinite(scores["CORR"]).all(), "Nonfinite CORR score")
    for metric in ("CORR", "NCORR"):
        available = scores[metric].dropna().to_numpy(dtype=float)
        require(np.isfinite(available).all() and (np.abs(available) <= 1 + TOLERANCE).all(),
                f"Invalid {metric} score")
    require(run["training"]["training_rows"] == run["train_sample_rows"] > 0,
            "Invalid training sample metadata")
    require(run["training"]["n_features"] == len(run["features"]) > 0,
            "Invalid training feature metadata")

    for split in SPLITS:
        group = scores[scores["split"].eq(split)].sort_values("date")
        expected = run["splits"][split]
        require(len(group) == expected["dates"] and
                int(group["n_rows"].sum()) == expected["rows"],
                f"Per-date coverage differs from run.json for {split}")
        require(group["date"].min() == pd.Timestamp(expected["min_date"]) and
                group["date"].max() == pd.Timestamp(expected["max_date"]),
                f"Observed date boundaries differ from run.json for {split}")
        for metric in ("CORR", "NCORR"):
            row = summary[summary["split"].eq(split) & summary["metric"].eq(metric)].iloc[0]
            values = group[metric].dropna().to_numpy(dtype=float)
            require(row["n_dates"] == len(group) and row["n_scored_dates"] == len(values),
                    f"Summary date counts differ for {split}/{metric}")
            expected_values = {
                "mean": float(values.mean()) if len(values) else np.nan,
                "std_ddof0": float(values.std(ddof=0)) if len(values) else np.nan,
                "worst_date_score": float(values.min()) if len(values) else np.nan,
                "positive_fraction": float((values > 0).mean()) if len(values) else np.nan,
            }
            for name, value in expected_values.items():
                check_close(float(row[name]), value, f"{split}/{metric}/{name}")
            require(row["training_rows"] == run["train_sample_rows"] and
                    row["target"] == run["target"] and
                    row["n_features"] == len(run["features"]) and row["seed"] == run["seed"],
                    f"Summary training metadata differs for {split}/{metric}")
    require(scores[scores["split"].eq("tuning")]["date"].max() <
            scores[scores["split"].eq("holdout")]["date"].min(),
            "Evaluation split dates overlap or are reversed")
    require(audit["all_date_corr"]["dates"] == len(scores) and
            audit["all_date_corr"]["rows"] == int(scores["n_rows"].sum()) and
            audit["all_summary_metrics"]["rows"] == len(summary),
            "Coverage differs from the independent validation record")
    return run, scores.sort_values(["split", "date"]).reset_index(drop=True)


def build_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_rows = []
    for (split, year), group in scores.groupby(["split", scores["date"].dt.year], sort=True):
        values = group["CORR"].to_numpy(dtype=float)
        annual_rows.append({
            "split": split, "year": int(year), "date_min": group["date"].min(),
            "date_max": group["date"].max(), "n_dates": len(group),
            "n_rows": int(group["n_rows"].sum()), "mean_corr": float(values.mean()),
            "std_ddof0": float(values.std(ddof=0)),
            "negative_date_count": int((values < 0).sum()),
            "positive_fraction": float((values > 0).mean()),
        })

    failure_rows = []
    for split, group in scores.groupby("split", sort=True):
        group = group.sort_values("date")
        negative = group["CORR"].lt(0)
        # Nonnegative observations interrupt a run; calendar gaps do not imply
        # extra observations. Never connect a run across split boundaries.
        segment = (~negative).cumsum()
        for _, period in group[negative].groupby(segment[negative], sort=True):
            values = period["CORR"].to_numpy(dtype=float)
            failure_rows.append({
                "split": split, "start_date": period["date"].min(),
                "end_date": period["date"].max(), "n_dates": len(period),
                "total_corr": float(values.sum()), "mean_corr": float(values.mean()),
                "worst_corr": float(values.min()),
            })
    failure_columns = ["split", "start_date", "end_date", "n_dates", "total_corr",
                       "mean_corr", "worst_corr"]
    failures = pd.DataFrame(failure_rows, columns=failure_columns)
    failures = failures.sort_values(["split", "total_corr", "start_date"]).reset_index(drop=True)
    return pd.DataFrame(annual_rows), failures


def summarize(output: Path) -> None:
    source_hashes = {name: sha256(output / name) for name in SOURCE_FILES}
    run, scores = read_sources(output)
    started = datetime.fromisoformat(run["started_utc"])
    require(started.tzinfo is not None, "Run start must have an explicit timezone")
    annual, failures = build_tables(scores)
    require(source_hashes == {name: sha256(output / name) for name in SOURCE_FILES},
            "Source files changed while generating summaries")
    annual.to_csv(output / "annual_corr.csv", index=False, date_format="%Y-%m-%d")
    failures.to_csv(output / "failure_periods.csv", index=False, date_format="%Y-%m-%d")
    metadata = {
        "status": "complete", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date_local": started.astimezone(ZoneInfo("America/Los_Angeles")).date().isoformat(),
        "timezone": "America/Los_Angeles",
        "model": MODEL, "metric": "CORR", "source_files_sha256": source_hashes,
        "summary_script_sha256": sha256(Path(__file__).resolve()),
        "output_files_sha256": {name: sha256(output / name) for name in
                                ("annual_corr.csv", "failure_periods.csv")},
        "definitions": {
            "annual": "Equal weight per observed date, grouped by split and calendar year; std ddof=0. Actual date_min/date_max retain partial-year boundaries.",
            "failure_period": "Maximal sequence of adjacent observed dates with CORR < 0 within a split. Zero or positive CORR ends a sequence; dates need not be consecutive calendar days.",
            "total_corr": "Additive sum of CORR scores; not an investment return or portfolio drawdown.",
        },
        "historical_holdout_previously_observed": run.get("historical_holdout_previously_observed"),
        "annual_rows": len(annual), "failure_periods": len(failures),
    }
    # A relocated contribution may not include the proposal; numerical summaries
    # depend only on the four required result sources above.
    proposal = Path(__file__).resolve().parents[1] / "DSCI550_Numerai_Project2_Proposal.pdf"
    if proposal.is_file():
        metadata["proposal_reference"] = {"path": str(proposal), "sha256": sha256(proposal)}
    (output / "analysis_sources.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(annual)} annual summaries and {len(failures)} negative-score periods: {output}",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Completed, independently validated baseline output directory")
    args = parser.parse_args()
    summarize(args.output.resolve())


if __name__ == "__main__":
    main()
