"""Independently audit a completed round-pinned baseline data refresh.

This supplements the original all-date CORR validator. It verifies complete raw
files, source copies, raw mature-label coverage, protocol equivalence, every
added or revised scored date, and annual/failure summaries. It does not train
models, download datasets, or claim unseen-test or live performance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath

import lightgbm as lgb
import numpy as np
from numerapi import NumerAPI
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import norm, rankdata
from threadpoolctl import threadpool_limits


KEYS = ["numerai_ticker", "date"]
GROUPS = ["split", "date"]
ROW_KEYS = KEYS + ["split"]
TARGET = "target_jupiter_60"
MODEL = "B0_small_main"
RAW_NAMES = ("features.json", "train.parquet", "validation.parquet",
             "validation_neutralizer.parquet", "validation_example_preds.parquet")
TOL = 1e-10


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(root, name):
    relative = PurePosixPath(name.replace("\\", "/"))
    require(not relative.is_absolute() and ".." not in relative.parts,
            "Invalid relative provenance path")
    path = root.joinpath(*relative.parts).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Missing provenance file")
    return path


def parse_dates(frame):
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    require(frame["date"].notna().all() and frame["date"].dt.tz is None,
            "Invalid observation dates")
    require(frame["date"].eq(frame["date"].dt.normalize()).all(),
            "Observation dates contain times")
    return frame


def date_text(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def date_frame(path, date, columns):
    kind = pq.read_schema(path).field("date").type
    value = (date_text(date) if pa.types.is_string(kind) or pa.types.is_large_string(kind)
             else pd.Timestamp(date).date() if pa.types.is_date(kind)
             else pd.Timestamp(date).to_pydatetime())
    return parse_dates(pd.read_parquet(path, columns=columns, filters=[("date", "==", value)]))


def compare_value(actual, expected, label, tolerance=1e-12):
    require(bool(np.isclose(actual, expected, rtol=0, atol=tolerance, equal_nan=True)),
            f"Independent value differs: {label}")


def check_annual_and_failures(output, scores):
    """Use explicit chronological loops, independently of the summary helper."""
    annual = {}
    failures = {}
    for split in sorted(scores["split"].unique()):
        rows = scores[scores["split"].eq(split)].sort_values("date")
        negative = []

        def finish_negative():
            if negative:
                values = np.asarray([r.CORR for r in negative], dtype=float)
                key = (split, date_text(negative[0].date), date_text(negative[-1].date))
                failures[key] = {"n_dates": len(negative), "total_corr": float(sum(values)),
                                 "mean_corr": float(sum(values) / len(values)),
                                 "worst_corr": float(min(values))}
                negative.clear()

        for row in rows.itertuples(index=False):
            key = (split, int(row.date.year))
            annual.setdefault(key, []).append(row)
            if row.CORR < 0:
                negative.append(row)
            else:
                finish_negative()
        finish_negative()

    recorded_annual = pd.read_csv(output / "annual_corr.csv")
    recorded_failures = pd.read_csv(output / "failure_periods.csv")
    require(not recorded_annual.duplicated(["split", "year"]).any(), "Duplicate annual keys")
    require(not recorded_failures.duplicated(["split", "start_date", "end_date"]).any(),
            "Duplicate failure-period keys")
    require(len(recorded_annual) == len(annual), "Annual row coverage differs")
    require(len(recorded_failures) == len(failures), "Failure-period coverage differs")
    for row in recorded_annual.itertuples(index=False):
        key = (row.split, row.year)
        require(key in annual, "Unexpected annual key")
        members = annual[key]
        vals = np.asarray([x.CORR for x in members], dtype=float)
        mean = sum(vals) / len(vals)
        require((row.date_min, row.date_max) == (date_text(members[0].date), date_text(members[-1].date)),
                "Annual date boundaries differ")
        expected = {"n_dates": len(members), "n_rows": sum(x.n_rows for x in members),
                    "mean_corr": mean, "std_ddof0": np.sqrt(sum((vals - mean) ** 2) / len(vals)),
                    "negative_date_count": sum(x < 0 for x in vals),
                    "positive_fraction": sum(x > 0 for x in vals) / len(vals)}
        for field, value in expected.items():
            compare_value(getattr(row, field), value, f"Annual {key}/{field}")
    for row in recorded_failures.itertuples(index=False):
        key = (row.split, row.start_date, row.end_date)
        require(key in failures, "Unexpected failure-period key")
        for field, value in failures[key].items():
            compare_value(getattr(row, field), value, f"Failure {key}/{field}")
    return {"annual_rows": len(annual), "negative_score_periods": len(failures),
            "method": "Separate chronological loops and year filters; actual partial-year boundaries retained."}


def audit(previous_root, previous, root, output):
    require(previous_root != root and previous != output, "Prior and new paths must differ")
    require(not (output / "freshness_validation.json").exists(), "Preserving existing freshness audit")
    old_run, run = read_json(previous / "run.json"), read_json(output / "run.json")
    old_protocol, protocol = read_json(previous / "protocol.json"), read_json(output / "protocol.json")
    for folder, info in ((previous, old_run), (output, run)):
        require(info.get("status") == "complete", "A completed run is required")
        require(read_json(folder / "validation.json").get("status") == "passed",
                "The original independent CORR audit must pass first")
    fields = ["features", "params", "seed", "max_train_rows", "target", "data_version", "embargo_weekdays"]
    for field in fields:
        require(old_protocol[field] == protocol[field], f"Baseline protocol changed: {field}")
    require(protocol["seed"] == 7 and protocol["max_train_rows"] == 200000,
            "Expected fixed seed 7 and 200000-row sample limit")
    require(protocol["target"] == TARGET and protocol["data_version"] == "v6.0", "Unexpected baseline data identity")
    require(run["train_sample_rows"] == min(protocol["max_train_rows"], run["train_input_rows"]),
            "Training sample size does not follow fixed protocol")

    snapshot = read_json(root / "snapshot.json")
    manifest = read_json(root / "data/raw/v6.0/manifest.json")
    previous_snapshot = read_json(previous_root / "snapshot.json")
    previous_manifest_path = previous_root / "data/raw/v6.0/manifest.json"
    previous_manifest = read_json(previous_manifest_path)
    require(previous_snapshot["status"] == previous_manifest["status"] == "complete",
            "Previous snapshot provenance is incomplete")
    require(sha256(previous_manifest_path) == previous_snapshot["raw_manifest_sha256"],
            "Previous raw manifest provenance changed")
    require(snapshot["status"] == manifest["status"] == "complete", "Incomplete raw snapshot")
    require(snapshot["round"] == manifest["round"], "Snapshot and manifest rounds differ")
    round_number = snapshot["round"]
    expected = {f"v6.0/{name}" for name in RAW_NAMES}
    require(len(previous_manifest["files"]) == len(expected) and
            {r["dataset"] for r in previous_manifest["files"]} == expected,
            "Expected five previous raw dataset provenance records")
    previous_raw = {r["dataset"]: {"bytes": r["bytes"], "sha256": r["sha256"]}
                    for r in previous_manifest["files"]}
    require(len(manifest["files"]) == len(expected) and {r["dataset"] for r in manifest["files"]} == expected,
            "Expected exactly five complete v6 raw datasets")
    verified_files = []
    for record in manifest["files"]:
        path = source_path(root, record["path"])
        require(record["round"] == round_number and record["status"] == "downloaded",
                "Raw file round/status differs")
        require(path.stat().st_size == record["bytes"], "Raw file byte count changed")
        require(sha256(path) == record["sha256"], "Raw file hash changed")
        content_length = record.get("remote_metadata", {}).get("ContentLength")
        require(content_length is None or content_length == record["bytes"], "Remote object byte count differs")
        verified_files.append({"dataset": record["dataset"], "bytes": record["bytes"],
                               "sha256": record["sha256"]})
        print(f"Verified full raw file: {record['dataset']}", flush=True)
    require(sha256(root / "data/raw/v6.0/manifest.json") == snapshot["raw_manifest_sha256"],
            "Snapshot raw manifest hash differs")
    source_copies = {}
    for relative, expected_hash in snapshot["source_copy_sha256"].items():
        require(sha256(source_path(root, relative)) == expected_hash, "Copied source hash differs")
        source_copies[relative] = expected_hash

    raw = parse_dates(pd.read_parquet(root / "data/raw/v6.0/validation.parquet",
                                     columns=KEYS + ["data_type", TARGET]))
    require(not raw.duplicated(KEYS).any(), "Raw validation has duplicate stock/date keys")
    eligible = raw["data_type"].eq("validation") & raw[TARGET].notna()
    require(eligible.any(), "No eligible mature raw validation rows")
    mature = raw.loc[eligible]
    raw_max = date_text(raw["date"].max())
    mature_max = date_text(mature["date"].max())
    plan = read_json(root / "data/processed/v6.0/split_manifest.json")
    holdout_min = raw.loc[raw["date"].ge(pd.Timestamp(plan["holdout_start_requested"])), "date"].min()
    require(holdout_min == pd.Timestamp(plan["holdout_date_at_or_after"]), "Raw holdout boundary differs")
    train_max = pd.Timestamp(run["train_max_date_before_sample"])
    embargo = int(protocol["embargo_weekdays"])
    tuning_after, tuning_before = train_max + pd.offsets.BDay(embargo), holdout_min - pd.offsets.BDay(embargo)
    tuning_mask = eligible & raw["date"].gt(tuning_after) & raw["date"].lt(tuning_before)
    holdout_mask = eligible & raw["date"].ge(holdout_min)
    raw_evaluation = pd.concat([
        raw.loc[mask, KEYS + [TARGET]].assign(split=split)
        for split, mask in (("tuning", tuning_mask), ("holdout", holdout_mask))
    ], ignore_index=True)
    raw_evaluation[TARGET] = raw_evaluation[TARGET].astype("float32")
    raw_coverage = {"raw_rows": len(raw), "eligible_mature_rows": len(mature),
                    "eligible_mature_dates": int(mature["date"].nunique()),
                    "raw_max_date": raw_max, "mature_max_date": mature_max}
    predictions = parse_dates(pd.read_parquet(output / "predictions.parquet", columns=ROW_KEYS + [TARGET, MODEL]))
    joined = raw_evaluation.merge(predictions[ROW_KEYS + [TARGET]], on=ROW_KEYS, how="outer",
                                  indicator=True, validate="one_to_one", suffixes=("_raw", "_saved"))
    require(joined["_merge"].eq("both").all(), "Saved prediction keys differ from independent raw eligibility masks")
    require(np.array_equal(joined[TARGET + "_raw"], joined[TARGET + "_saved"]),
            "Saved targets differ from eligible raw targets")
    del raw, mature, raw_evaluation, joined

    old_predictions = parse_dates(pd.read_parquet(previous / "predictions.parquet", columns=ROW_KEYS + [TARGET, MODEL]))
    paired = old_predictions.merge(predictions, on=ROW_KEYS, how="outer", validate="one_to_one",
                                    indicator=True, suffixes=("_old", "_new"))
    both = paired["_merge"].eq("both")
    changed = {}
    affected = ~both
    for label, column in (("target", TARGET), ("prediction", MODEL)):
        delta = (paired[column + "_new"] - paired[column + "_old"]).abs()
        revised = both & delta.ne(0)
        affected |= revised
        changed[label] = {"changed_rows": int(revised.sum()),
                          "maximum_absolute_difference": float(delta[both].max()) if both.any() else None}
    affected_keys = set(paired.loc[affected, GROUPS].itertuples(index=False, name=None))
    common_rows = int(both.sum())
    added_rows = int(paired["_merge"].eq("right_only").sum())
    removed_rows = int(paired["_merge"].eq("left_only").sum())
    del old_predictions, paired

    old_scores = parse_dates(pd.read_csv(previous / "per_date_metrics.csv"))
    scores = parse_dates(pd.read_csv(output / "per_date_metrics.csv"))
    old_lookup, lookup = old_scores.set_index(GROUPS), scores.set_index(GROUPS)
    require(old_lookup.index.is_unique and lookup.index.is_unique, "Duplicate score dates")
    old_dates, current_dates = set(old_lookup.index), set(lookup.index)
    new_dates, removed_dates = current_dates - old_dates, old_dates - current_dates
    for key in old_dates & current_dates:
        for metric in ("CORR", "NCORR"):
            old_value, value = old_lookup.loc[key, metric], lookup.loc[key, metric]
            if not ((pd.isna(old_value) and pd.isna(value)) or old_value == value):
                affected_keys.add(key)
    affected_keys |= new_dates
    revised_dates = (affected_keys & current_dates) - new_dates

    booster = lgb.Booster(model_file=str(output / "models" / f"{MODEL}.txt"))
    require(booster.feature_name() == protocol["features"], "Saved model feature order differs")
    require(booster.num_trees() == protocol["params"]["n_estimators"], "Saved model tree count differs")
    neutralizer_path = root / "data/raw/v6.0/validation_neutralizer.parquet"
    neutralizer_columns = [c for c in pq.read_schema(neutralizer_path).names if c.startswith("neutralizer_column_")]
    require(neutralizer_columns, "No neutralizer columns")
    affected_checks = []
    for split, date in sorted((affected_keys & current_dates), key=lambda key: (key[0], key[1])):
        saved = predictions[predictions["split"].eq(split) & predictions["date"].eq(date)].sort_values("numerai_ticker")
        require(not saved.duplicated(KEYS).any(), "Duplicate affected prediction keys")
        source = date_frame(root / f"data/processed/v6.0/{split}.parquet", date,
                            KEYS + protocol["features"] + [TARGET])
        combined = saved[KEYS + [MODEL]].merge(source, on=KEYS, how="outer", indicator=True, validate="one_to_one")
        require(combined["_merge"].eq("both").all(), "Affected-date processed/model coverage differs")
        generated = booster.predict(combined[protocol["features"]], num_threads=12)
        model_error = float(np.max(np.abs(generated - combined[MODEL].to_numpy())))
        require(model_error <= TOL, "Affected-date saved-model predictions differ")

        neutralizers = date_frame(neutralizer_path, date, KEYS + neutralizer_columns)
        projected = saved[KEYS + [MODEL, TARGET]].merge(neutralizers, on=KEYS, how="left", validate="one_to_one")
        require(len(projected) == len(saved) and np.isfinite(projected[neutralizer_columns].to_numpy()).all(),
                "Affected-date neutralizer coverage is incomplete")
        require(np.isfinite(float(lookup.loc[(split, date), "NCORR"])), "Affected-date NCORR unavailable")
        gaussian = norm.ppf((rankdata(projected[MODEL].to_numpy(), method="average") - .5) / len(projected))
        design = np.column_stack([projected[neutralizer_columns].to_numpy(dtype=float), np.ones(len(projected))])
        residual = gaussian - design @ np.linalg.lstsq(design, gaussian, rcond=1e-6)[0]
        y = projected[TARGET].to_numpy()
        centered = y - y.mean()
        independent_ncorr = float(np.corrcoef(residual, centered.astype(float))[0, 1])
        recorded_ncorr = float(lookup.loc[(split, date), "NCORR"])
        ncorr_error = abs(independent_ncorr - recorded_ncorr)
        require(ncorr_error <= TOL, "Affected-date independent NCORR differs")
        affected_checks.append({"split": split, "date": date_text(date),
                                "reasons": ["added_scored_date" if (split, date) in new_dates else "revised_existing_date"],
                                "rows": len(saved), "saved_model_maximum_absolute_error": model_error,
                                "independent_ncorr": independent_ncorr, "independent_ncorr_absolute_error": ncorr_error,
                                "neutralizer_columns": len(neutralizer_columns)})
        print(f"Checked model and independent NCORR: {split}/{date_text(date)}", flush=True)

    summary_checks = check_annual_and_failures(output, scores)
    logging.disable(logging.CRITICAL)
    try:
        api = NumerAPI(show_progress_bars=False)
        current_round = api.get_current_round()
        available = api.list_datasets(round_num=current_round)
        available_versions = sorted({item.split("/", 1)[0] for item in available if item.startswith("v") and "/" in item})
    except Exception as exc:
        raise RuntimeError(f"Post-run official API check failed: {type(exc).__name__}") from None
    require(isinstance(current_round, int) and current_round >= round_number, "Unexpected post-run official round")
    relevant_sources = ["run.json", "protocol.json", "validation.json", "per_date_metrics.csv", "metrics.csv",
                        "data_update_comparison.json", "new_dates_metrics.csv", "annual_corr.csv", "failure_periods.csv"]
    result = {
        "status": "passed", "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round": round_number, "current_round_after_validation": current_round,
        "available_versions_after_validation": available_versions,
        "round_still_current_after_validation": current_round == round_number,
        "raw_validation_max_date": raw_max, "mature_validation_max_date": mature_max,
        "raw_mature_coverage": raw_coverage, "raw_files_verified": verified_files,
        "downloaded_bytes": sum(item["bytes"] for item in verified_files),
        "previous_raw_manifest_sha256": sha256(previous_manifest_path),
        "raw_dataset_comparison": {
            item["dataset"]: {"previous": previous_raw[item["dataset"]],
                              "current": {"bytes": item["bytes"], "sha256": item["sha256"]},
                              "identical": previous_raw[item["dataset"]] ==
                                           {"bytes": item["bytes"], "sha256": item["sha256"]}}
            for item in verified_files},
        "source_copy_hashes_verified": source_copies,
        "protocol_equivalence": {"fields": fields, "identical": True},
        "training_sample": {"previous_rows": old_run["train_sample_rows"], "current_rows": run["train_sample_rows"],
                            "keys_sha256_identical": old_run["train_sample_keys_sha256"] == run["train_sample_keys_sha256"]},
        "model_sha256_identical": old_run["model_sha256"] == run["model_sha256"],
        "common_prediction_rows": common_rows, "new_prediction_rows": added_rows, "removed_prediction_rows": removed_rows,
        "common_prediction_changes": changed,
        "new_scored_dates": sorted({date_text(date) for _, date in new_dates}),
        "revised_scored_dates": sorted({date_text(date) for _, date in revised_dates}),
        "removed_scored_dates": sorted({date_text(date) for _, date in removed_dates}),
        "affected_date_checks": affected_checks,
        "affected_date_check_scope": "Every currently scored date added or revised by prediction keys, targets, predictions, or CORR/NCORR; none when the list is empty.",
        "independent_summary_checks": summary_checks,
        "snapshot_sha256": sha256(root / "snapshot.json"),
        "raw_manifest_sha256": sha256(root / "data/raw/v6.0/manifest.json"),
        "source_files_sha256": {name: sha256(output / name) for name in relevant_sources},
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "scope": "Independent raw completeness/maturity, frozen-protocol equivalence, affected-date model/NCORR, and annual/failure summary audit. Historical development evidence; no unseen-test or live-performance claim.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=12):
        result = audit(args.previous_root.resolve(), args.previous.resolve(), args.root.resolve(), args.output.resolve())
    path = args.output.resolve() / "freshness_validation.json"
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({key: result[key] for key in ("status", "round", "current_round_after_validation",
                     "raw_validation_max_date", "mature_validation_max_date", "new_scored_dates",
                     "revised_scored_dates", "independent_summary_checks")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
