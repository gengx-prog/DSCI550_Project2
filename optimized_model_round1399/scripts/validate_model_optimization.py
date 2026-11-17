"""Independently audit a completed model-optimization experiment.

Recompute saved-score arithmetic without importing the runner, check every
saved prediction's CORR, and check NCORR on one complete date in every fold.
Existing runs and their recorded provenance are never modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import linalg, stats
from threadpoolctl import threadpool_limits

KEYS = ["numerai_ticker", "date"]
TARGET = "target_jupiter_60"
TOL = 1e-10
REFERENCE = "B0_fixed_uploaded"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message: str) -> None:
    require(np.isclose(actual, expected, atol=TOL, rtol=0, equal_nan=True),
            f"{message}: saved={actual}, independently computed={expected}")


def independent_summary(values) -> dict:
    values = np.asarray(values, dtype=float)
    require(len(values) > 0 and np.isfinite(values).all(), "Invalid score array")
    mean = float(np.sum(values) / len(values))
    std = float(np.sqrt(np.mean((values - mean) ** 2)))
    cumulative = np.r_[0.0, np.cumsum(values)]
    return {"n_scored_dates": len(values), "mean": mean, "std_ddof0": std,
            "sharpe": mean / std if std else np.nan,
            "worst_date_score": float(values.min()),
            "positive_fraction": float(np.count_nonzero(values > 0) / len(values)),
            "max_drawdown": float(np.max(np.maximum.accumulate(cumulative) - cumulative))}


def independent_scores(predictions, target, neutralizers=None):
    """Match official rank/transform precision, using independent Pearson sums."""
    x, y = np.asarray(predictions, dtype=float), np.asarray(target)
    require(x.ndim == 2 and len(x) == len(y) and len(y) > 1,
            "Invalid score dimensions")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite score input")
    require((np.ptp(x, axis=0) > 0).all(), "Constant saved prediction")
    gaussian = stats.norm.ppf((stats.rankdata(x, method="average", axis=0) - .5) / len(y))
    # Preserve native float32 label operations, as in the official scorer.
    native_centered = y - y.mean()
    powered_y = (np.sign(native_centered) * np.abs(native_centered) ** 1.5).astype(float)
    powered_x = np.sign(gaussian) * np.abs(gaussian) ** 1.5
    powered_x -= powered_x.mean(axis=0)
    powered_y -= powered_y.mean()
    corr = np.sum(powered_x * powered_y[:, None], axis=0) / np.sqrt(
        np.sum(powered_x ** 2, axis=0) * np.sum(powered_y ** 2))
    ncorr = None
    if neutralizers is not None:
        z = np.asarray(neutralizers, dtype=float)
        require(len(z) == len(y) and np.isfinite(z).all(), "Invalid neutralizer matrix")
        design = np.column_stack([z, np.ones(len(y))])
        coefficients = linalg.lstsq(design, gaussian, cond=1e-6, lapack_driver="gelsd")[0]
        residual = gaussian - design @ coefficients
        residual -= residual.mean(axis=0)
        centered_y = native_centered.astype(float)
        centered_y -= centered_y.mean()
        ncorr = np.sum(residual * centered_y[:, None], axis=0) / np.sqrt(
            np.sum(residual ** 2, axis=0) * np.sum(centered_y ** 2))
    require(np.isfinite(corr).all() and (ncorr is None or np.isfinite(ncorr).all()),
            "Undefined independent score")
    return corr, ncorr


def parquet_date_filter(path: Path, date: pd.Timestamp):
    dtype = pq.read_schema(path).field("date").type
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        value = date.strftime("%Y-%m-%d")
    elif pa.types.is_date(dtype):
        value = date.date()
    else:
        value = date.to_pydatetime()
    return [("date", "==", value)]


def verify_source_hashes(mapping: dict, base: Path) -> int:
    count = 0
    for filename, expected in mapping.items():
        path = Path(filename)
        if not path.is_absolute():
            path = base / path
        require(path.is_file(), f"Missing recorded input: {path}")
        require(sha256(path) == expected, f"Recorded input changed: {path}")
        count += 1
    return count


def check_pinned_snapshot(root: Path, known_hashes: dict) -> dict:
    manifest_path = root / "data/raw/v6.0/manifest.json"
    manifest, snapshot = read_json(manifest_path), read_json(root / "snapshot.json")
    require(manifest["status"] == snapshot["status"] == "complete"
            and manifest["version"] == snapshot["data_version"] == "v6.0",
            "Pinned source snapshot is incomplete or has a different version")
    number = manifest["round"]
    require(isinstance(number, int) and number > 0 and snapshot["round"] == number
            and snapshot["requested_round"] == number, "Pinned snapshot round differs")
    require(sha256(manifest_path) == snapshot["raw_manifest_sha256"], "Snapshot raw manifest changed")
    already_checked = {str((root / filename).resolve()): expected for filename, expected in known_hashes.items()}
    checks, scoring_hashes = [], {}
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file(), "Pinned raw path invalid")
        require(entry["round"] == number and entry["status"] == "downloaded", "Raw source round/status differs")
        require(path.stat().st_size == entry["bytes"] == int(entry["remote_metadata"]["ContentLength"]),
                f"Pinned source byte count differs: {path.name}")
        actual = already_checked.get(str(path)) or sha256(path)
        require(actual == entry["sha256"], f"Pinned raw source hash differs: {path.name}")
        scoring_hashes[entry["dataset"]] = actual
        checks.append({"dataset": entry["dataset"], "round": number, "bytes": entry["bytes"], "sha256": actual})
    require("v6.0/validation_neutralizer.parquet" in scoring_hashes, "Pinned scoring neutralizer missing")
    references = {}
    for filename in ("references/official_sources.json", "references/v6_example_model.ipynb"):
        path = root / filename
        expected = snapshot["source_copy_sha256"][filename]
        require(sha256(path) == expected, f"Snapshot reference changed: {filename}")
        references[filename] = expected
    return {"round": number, "manifest_sha256": sha256(manifest_path),
            "snapshot_sha256": sha256(root / "snapshot.json"), "files": checks,
            "scoring_input_sha256": scoring_hashes, "reference_sha256": references}


def check_summary(scores: pd.DataFrame, metrics: pd.DataFrame) -> int:
    require({"phase", "model", "metric", "mean", "std_ddof0", "sharpe", "worst_date_score",
             "positive_fraction", "max_drawdown", "n_scored_dates", "n_dates", "n_folds",
             "fold_mean_std_ddof0", "worst_fold_mean"}.issubset(metrics.columns),
            "Summary columns missing")
    require(not metrics.duplicated(["phase", "model", "metric"]).any(), "Duplicate metric rows")
    expected_keys = set()
    checked = 0
    for (phase, model), group in scores.groupby(["phase", "model"], sort=True):
        group = group.sort_values("date")
        for metric in ("CORR", "NCORR"):
            expected_keys.add((phase, model, metric))
            rows = metrics.loc[metrics.phase.eq(phase) & metrics.model.eq(model) & metrics.metric.eq(metric)]
            require(len(rows) == 1, f"Missing summary {phase}/{model}/{metric}")
            row = rows.iloc[0]
            for name, value in independent_summary(group[metric]).items():
                if name in row:
                    close(row[name], value, f"{phase}/{model}/{metric}/{name}")
            fold_means = group.groupby("fold")[metric].mean()
            for name, value in {"n_dates": len(group), "n_folds": len(fold_means),
                                "fold_mean_std_ddof0": np.std(fold_means),
                                "worst_fold_mean": fold_means.min()}.items():
                if name in row:
                    close(row[name], value, f"{phase}/{model}/{metric}/{name}")
            checked += 1
    require(set(map(tuple, metrics[["phase", "model", "metric"]].to_numpy())) == expected_keys,
            "Unexpected or incomplete summary matrix")
    return checked


def check_selection(scores: pd.DataFrame, selection: dict, candidates: list[str]) -> str:
    cv = scores.loc[scores.phase.eq("cv") & scores.model.isin(candidates)]
    require(cv.fold.nunique() == 4, "Selection needs all four CV folds")
    require(set(selection["candidates"]) == set(candidates), "Selection candidate matrix changed")
    ranking = {}
    for model in candidates:
        rows = cv.loc[cv.model.eq(model)]
        means = rows.groupby("fold").NCORR.mean()
        mean, std = float(rows.NCORR.mean()), float(np.std(means))
        expected = {"mean_NCORR": mean, "mean_NCORR_equal_date": mean, "fold_mean_std_ddof0": std,
                    "stability_penalty": .5 * std, "selection_score": mean - .5 * std,
                    "worst_fold_mean": float(means.min()), "n_dates": len(rows)}
        saved = selection["candidates"][model]
        require({"mean_NCORR", "fold_mean_std_ddof0", "selection_score", "fold_means", "n_dates"}.issubset(saved),
                "Selection reporting fields missing")
        for name, value in expected.items():
            if name in saved:
                close(saved[name], value, f"Selection {model}/{name}")
        require("selection_score" in saved, "Selection arithmetic not recorded")
        require(set(saved["fold_means"]) == set(means.index), "Selection fold-mean coverage differs")
        for fold_id, value in means.items():
            close(saved["fold_means"][fold_id], value, f"Selection {model}/{fold_id}")
        ranking[model] = mean - .5 * std
    winner = sorted(candidates, key=lambda model: (-ranking[model], model))[0]
    require(selection["selected_model"] == winner, "Independent CV winner differs")
    require(selection.get("development_used_for_selection", False) is False,
            "Selection used reused-development scores")
    require(winner != REFERENCE, "Fixed uploaded baseline entered model selection")
    return winner


def sample_keys_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame[KEYS].to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")).hexdigest()


def check_bundle(path: Path, model_id: str, feature_sets: dict,
                 eligible_keys: pd.DataFrame, prediction_date: pd.Timestamp,
                 cutoff: pd.Timestamp, evaluation: pd.DataFrame | None = None,
                 expected_predictions=None, n_jobs: int = 12) -> dict:
    bundle = read_json(path)
    prefix = model_id.split("_", 1)[0]
    require(prefix in {"O0", "O1", "O2", "O3"}, "Invalid bundle candidate")
    feature_set = "small" if prefix == "O0" else "medium"
    features = feature_sets[feature_set]
    seeds = [7] if prefix == "O0" else [7, 19, 43]
    max_rows = {"O0": 200_000, "O1": 600_000, "O2": 1_200_000, "O3": 1_200_000}[prefix]
    permutation = eligible_keys.sample(n=min(1_200_000, len(eligible_keys)), random_state=7).reset_index(drop=True)
    sample = permutation.iloc[:min(max_rows, len(permutation))]
    require(bundle["schema_version"] == 1 and bundle["model_id"] == model_id
            and bundle["target"] == TARGET and bundle["family"] == "lightgbm_rank_ensemble",
            f"Unexpected bundle identity: {path}")
    require(bundle["feature_set"] == feature_set and bundle["features"] == features,
            f"Feature schema differs: {path}")
    require(bundle["sample_seed"] == 7 and bundle["sample_keys_sha256"] == sample_keys_hash(sample),
            f"Independent nested sample hash differs: {path}")
    require(bundle["training_rows"] == len(sample) and bundle["sampling_permutation_rows"] == len(permutation),
            f"Bundle training/sample size differs: {path}")
    for name, expected in {"full_training_date_min": eligible_keys.date.min(),
                           "full_training_date_max": eligible_keys.date.max(),
                           "training_date_min": sample.date.min(),
                           "training_date_max": sample.date.max()}.items():
        require(pd.Timestamp(bundle[name]) == expected, f"Bundle {name} differs: {path}")
    require(eligible_keys.date.max() < cutoff and
            eligible_keys.date.max() + pd.offsets.BDay(65) < prediction_date,
            f"Bundle label maturity boundary violated: {path}")
    require(bundle["evaluation_labels_used_in_fit"] is False, "Bundle permits evaluation-label fitting")
    require(bundle["member_seeds"] == seeds and [m["seed"] for m in bundle["members"]] == seeds,
            "Bundle seeds differ from frozen candidate")
    require(bundle["rank_method"] == "average_percentile_within_date_then_equal_mean",
            "Unexpected rank aggregation")
    require(len(bundle["weights"]) == len(bundle["members"]) == len(seeds) and
            np.allclose(bundle["weights"], [1 / len(seeds)] * len(seeds), atol=1e-15, rtol=0),
            "Unexpected ensemble weights")
    fixed_params = {"n_estimators": 800 if prefix == "O3" else 300,
                    "learning_rate": .03 if prefix == "O3" else .05,
                    "max_depth": 5, "num_leaves": 31,
                    "colsample_bytree": .3 if prefix == "O3" else .1,
                    "deterministic": True, "force_col_wise": True}
    if prefix == "O3":
        fixed_params.update(min_child_samples=1000, reg_lambda=10.0, subsample=.8, subsample_freq=1)
    actual_names = {"n_estimators": "num_iterations", "colsample_bytree": "feature_fraction",
                    "min_child_samples": "min_data_in_leaf", "reg_lambda": "lambda_l2",
                    "subsample": "bagging_fraction", "subsample_freq": "bagging_freq"}
    values, member_audits = [], []
    for member in bundle["members"]:
        member_path = (path.parent / member["file"]).resolve()
        require(member_path.parent == path.parent.resolve() and member_path.suffix == ".txt",
                "Bundle member escapes bundle directory")
        require(sha256(member_path) == member["sha256"], f"Member hash differs: {member_path}")
        for name, expected in fixed_params.items():
            require(member["params"][name] == expected, f"Frozen parameter changed: {model_id}/{name}")
        for name in ("random_state", "feature_fraction_seed", "bagging_seed", "data_random_seed"):
            require(member["params"][name] == member["seed"], "Recorded member seed differs")
        booster = lgb.Booster(model_file=str(member_path))
        require(booster.feature_name() == features, f"Saved feature schema differs: {member_path}")
        require(booster.num_trees() == member["num_trees"] == fixed_params["n_estimators"],
                f"Saved tree count differs: {member_path}")
        for name, expected in fixed_params.items():
            actual_name = actual_names.get(name, name)
            require(booster.params.get(actual_name) == expected,
                    f"Actual serialized parameter differs: {model_id}/{actual_name}")
        require(booster.params.get("seed") == member["seed"] and
                all(booster.params.get(name) == member["seed"] for name in
                    ("feature_fraction_seed", "bagging_seed", "data_random_seed")),
                "Actual serialized member seeds differ")
        if evaluation is not None:
            values.append(booster.predict(evaluation[features], num_threads=n_jobs, validate_features=True))
        member_audits.append({"file": member_path.name, "sha256": member["sha256"],
                              "trees": booster.num_trees(), "seed": member["seed"]})
    prediction_error = None
    if evaluation is not None:
        ranked = pd.DataFrame(np.column_stack(values)).groupby(
            evaluation.date.reset_index(drop=True), sort=False).rank(method="average", pct=True).mean(axis=1).to_numpy()
        expected_predictions = np.asarray(expected_predictions)
        require(ranked.shape == expected_predictions.shape, "Bundle prediction count differs")
        prediction_error = float(np.max(np.abs(ranked - expected_predictions)))
        require(prediction_error < TOL, f"Saved bundle prediction differs: {model_id}: {prediction_error}")
    return {"model": model_id, "bundle_sha256": sha256(path), "training_rows": len(sample),
            "sample_keys_hash_reconstructed": True, "members": member_audits,
            "recomputed_prediction_rows": 0 if evaluation is None else len(evaluation),
            "prediction_max_abs_error": prediction_error}


def check_fold_predictions(root: Path, output: Path, fold: dict,
                           scores: pd.DataFrame, models: list[str]) -> dict:
    fold_id = fold["id"]
    path = output / "folds" / fold_id / "predictions.parquet"
    predictions = pd.read_parquet(path, columns=KEYS + [TARGET] + models)
    predictions["date"] = pd.to_datetime(predictions.date)
    require(len(predictions) > 0 and not predictions.duplicated(KEYS).any(),
            f"Invalid saved prediction keys: {fold_id}")
    require(not predictions[KEYS + [TARGET]].isna().any().any(), "Missing saved keys/labels")
    start, end = pd.Timestamp(fold["eval_start"]), pd.Timestamp(fold["eval_end"])
    cutoff = pd.Timestamp(fold["train_cutoff_exclusive"])
    embargo = int(fold["embargo_weekdays"])
    require(embargo >= 65 and cutoff == start - pd.offsets.BDay(embargo),
            f"Invalid maturity boundary: {fold_id}")
    require(predictions.date.between(start, end).all() and predictions.date.min() == start
            and predictions.date.max() == end, f"Saved evaluation dates differ: {fold_id}")
    local = scores.loc[scores.fold.eq(fold_id)]
    require(set(local.phase) == {fold["phase"]}, f"Fold phase changed: {fold_id}")
    require(set(local.model) == set(models), f"Model coverage changed: {fold_id}")
    grid = local.pivot(index="date", columns="model", values="n_rows")
    counts = predictions.groupby("date").size()
    require(grid.index.equals(counts.index) and grid.notna().all().all(), "Incomplete score-date matrix")
    require(grid.eq(counts, axis=0).all().all(), "Scoring populations differ")
    if "n_eval_dates" in fold:
        require(len(counts) == fold["n_eval_dates"], "Frozen date count differs")
    max_corr_error = 0.0
    sample = None
    neutralizer_path = root / "data/raw/v6.0/validation_neutralizer.parquet"
    neutralizer_cols = [c for c in pq.read_schema(neutralizer_path).names if c.startswith("neutralizer_column_")]
    require(len(neutralizer_cols) > 0, "Missing NCORR neutralizers")
    for date, group in predictions.groupby("date", sort=True):
        group = group.sort_values("numerai_ticker").reset_index(drop=True)
        recorded = local.loc[local.date.eq(date)].set_index("model").loc[models]
        z = None
        if sample is None:
            neutralizers = pd.read_parquet(neutralizer_path, columns=KEYS + neutralizer_cols,
                                          filters=parquet_date_filter(neutralizer_path, date))
            neutralizers["date"] = pd.to_datetime(neutralizers.date)
            aligned = group[KEYS].merge(neutralizers, on=KEYS, how="left", sort=False,
                                        validate="one_to_one")
            require(not aligned[neutralizer_cols].isna().any().any(), "Incomplete neutralizer coverage")
            z = aligned[neutralizer_cols].to_numpy(dtype=float)
            raw_path = root / "data/raw/v6.0/validation.parquet"
            raw = pd.read_parquet(raw_path, columns=KEYS + [TARGET, "data_type"],
                                  filters=parquet_date_filter(raw_path, date))
            raw["date"] = pd.to_datetime(raw.date)
            raw = raw.loc[raw.data_type.eq("validation") & raw[TARGET].notna()]
            match = group[KEYS + [TARGET]].merge(raw, on=KEYS, how="outer", validate="one_to_one",
                                               suffixes=("_saved", "_raw"), indicator=True)
            require(match["_merge"].eq("both").all(), "Sampled date omits or adds eligible raw rows")
            require(np.array_equal(match[TARGET + "_saved"], match[TARGET + "_raw"]),
                    "Saved labels differ from raw source")
        corr, ncorr = independent_scores(group[models].to_numpy(), group[TARGET].to_numpy(), z)
        error = float(np.max(np.abs(corr - recorded.CORR.to_numpy())))
        max_corr_error = max(max_corr_error, error)
        require(error < TOL, f"Independent CORR discrepancy: {fold_id}/{date}: {error}")
        if ncorr is not None:
            ncorr_error = float(np.max(np.abs(ncorr - recorded.NCORR.to_numpy())))
            require(ncorr_error < TOL, f"Independent NCORR discrepancy: {fold_id}/{date}: {ncorr_error}")
            sample = {"date": str(date.date()), "rows": len(group),
                      "models": len(models), "neutralizer_columns": len(neutralizer_cols),
                      "NCORR_max_abs_error": ncorr_error, "raw_target_and_population_match": True}
    return {"fold": fold_id, "phase": fold["phase"], "prediction_rows": len(predictions),
            "prediction_dates": len(counts), "all_dates_CORR_max_abs_error": max_corr_error,
            "independent_NCORR_sample": sample, "predictions_sha256": sha256(path)}


def check_reference(output: Path, plan: dict, scores: pd.DataFrame) -> dict:
    """Keyed equality against the exact previously uploaded fixed baseline."""
    reference = plan["reference_model"]
    require(reference["id"] == REFERENCE, "Unexpected uploaded reference identity")
    directory = Path(reference["directory"])
    model_path = directory / "models/B0_small_main.txt"
    if not model_path.is_file():
        model_path = directory / "B0_small_main.txt"
    require(sha256(model_path) == reference["model_sha256"], "Uploaded reference model hash changed")
    path = directory / "predictions.parquet"
    baseline = pd.read_parquet(path, columns=KEYS + [TARGET, "B0_small_main"])
    baseline["date"] = pd.to_datetime(baseline.date)
    require(not baseline.duplicated(KEYS).any(), "Reference has duplicate saved keys")
    original_scores = pd.read_csv(directory / "per_date_metrics.csv", parse_dates=["date"])
    require(not original_scores.date.duplicated().any(), "Reference has duplicate per-date scores")
    matched_rows, matched_dates = 0, set()
    max_score_error = {"CORR": 0.0, "NCORR": 0.0}
    for fold in plan["folds"]:
        candidate = pd.read_parquet(output / "folds" / fold["id"] / "predictions.parquet",
                                     columns=KEYS + [TARGET, REFERENCE])
        candidate["date"] = pd.to_datetime(candidate.date)
        overlap = baseline.loc[baseline.date.between(pd.Timestamp(fold["eval_start"]),
                                                     pd.Timestamp(fold["eval_end"]))]
        match = overlap.merge(candidate, on=KEYS, how="left", validate="one_to_one",
                               suffixes=("_original", "_optimization"), indicator=True)
        require(match["_merge"].eq("both").all(), "Previously saved baseline rows missing from experiment")
        require(np.array_equal(match["B0_small_main"].to_numpy(), match[REFERENCE].to_numpy()),
                f"Uploaded reference raw predictions differ: {fold['id']}")
        require(np.array_equal(match[TARGET + "_original"].to_numpy(), match[TARGET + "_optimization"].to_numpy()),
                "Uploaded reference target labels changed")
        matched_rows += len(match)
        matched_dates.update(match.date.unique())
    require(matched_rows == len(baseline), "Not every previous baseline row was checked")
    recorded = scores.loc[scores.model.eq(REFERENCE)]
    matched = original_scores.merge(recorded, on="date", how="left", validate="one_to_one",
                                     suffixes=("_original", "_optimization"), indicator=True)
    require(matched["_merge"].eq("both").all(), "Previous baseline scoring dates missing")
    for metric in ("CORR", "NCORR"):
        error = float(np.max(np.abs(matched[metric + "_original"] - matched[metric + "_optimization"])))
        max_score_error[metric] = error
        require(error < TOL, f"Reference original {metric} differs: {error}")
    return {"reference_directory": str(directory), "original_prediction_sha256": sha256(path),
            "matched_rows": matched_rows, "matched_dates": len(matched_dates),
            "raw_predictions_exactly_equal": True, "target_labels_exactly_equal": True,
            "original_per_date_max_abs_error": max_score_error}


def check_live_handoff(output: Path, live_output: Path, n_jobs: int) -> dict:
    """Reconstruct both ensemble ranks and CSV ranks with scipy, from disk."""
    live_output = live_output.resolve()
    record = read_json(live_output / "live_generation.json")
    require(record["status"] == "complete" and record["account_submission_performed"] is False
            and record["training_performed"] is False, "Live handoff is incomplete or scope changed")
    require(record["current_round_before"] == record["round"] == record["current_round_after"],
            "Round changed while generating the handoff")
    run = read_json(output / "run.json")
    source = Path(run["final_bundle"])
    bundle_path = live_output / "model" / "bundle.json"
    require(sha256(source) == sha256(bundle_path) == record["bundle_sha256"],
            "Live handoff did not use the frozen final-fit bundle")
    bundle = read_json(bundle_path)
    require(bundle["model_id"] == run["selected_model"], "Live candidate differs from CV winner")
    verify_source_hashes(record["model_files_sha256"], live_output / "model")
    for field, filename in {"csv_sha256": record["csv_file"], "predictions_sha256": "predictions.parquet",
                            "live_parquet_sha256": "data/live.parquet",
                            "feature_metadata_sha256": "data/features.json",
                            "download_manifest_sha256": "data/download_manifest.json",
                            "validation_sha256": "validation.json"}.items():
        require(sha256(live_output / filename) == record[field], f"Live artifact hash differs: {filename}")
    metadata = read_json(live_output / "data/features.json")
    features = bundle["features"]
    require(features == [f for f in metadata["feature_sets"][bundle["feature_set"]] if f != "feature_country"],
            "Live metadata feature ordering differs")
    live = pd.read_parquet(live_output / "data/live.parquet", columns=KEYS + features)
    live["date"] = pd.to_datetime(live.date)
    require(live.date.nunique() == 1 and not live.numerai_ticker.isna().any()
            and not live.numerai_ticker.duplicated().any(), "Invalid live keys")
    date = live.date.iloc[0]
    expected_date = pd.to_datetime(str(record["round_info"]["dataDatestamp"]), format="%Y%m%d")
    require(date == expected_date and pd.Timestamp(bundle["full_training_date_max"]) < date - pd.offsets.BDay(65),
            "Live date or strict label maturity embargo differs")
    ranks = []
    for member in bundle["members"]:
        path = bundle_path.parent / member["file"]
        require(sha256(path) == member["sha256"], "Live native member hash differs")
        booster = lgb.Booster(model_file=str(path))
        require(booster.feature_name() == features and booster.num_trees() == member["num_trees"],
                "Live saved model schema/tree count differs")
        values = booster.predict(live[features], num_threads=n_jobs, validate_features=True)
        ranks.append(stats.rankdata(values, method="average") / len(values))
    raw = np.mean(np.column_stack(ranks), axis=1)
    final = stats.rankdata(raw, method="average") / len(raw)
    csv = pd.read_csv(live_output / record["csv_file"], dtype={"numerai_ticker": str},
                      float_precision="round_trip")
    require(csv.columns.tolist() == ["numerai_ticker", "prediction"]
            and csv.numerai_ticker.tolist() == live.numerai_ticker.tolist(), "Live CSV ticker/order mismatch")
    require(np.array_equal(csv.prediction.to_numpy(), final), "Independent live CSV ranking differs")
    saved = pd.read_parquet(live_output / "predictions.parquet")
    require(saved.numerai_ticker.tolist() == live.numerai_ticker.tolist(), "Live audit ticker order differs")
    raw_error = float(np.max(np.abs(raw - saved.raw_prediction.to_numpy())))
    require(raw_error < TOL and np.array_equal(saved.prediction.to_numpy(), final),
            "Independent live saved predictions differ")
    return {"status": "passed", "output": str(live_output), "round": record["round"],
            "rows": len(live), "csv_sha256": record["csv_sha256"],
            "independent_member_ranking": "scipy.stats.rankdata(method=average)",
            "independent_final_ranks_exactly_equal": True,
            "raw_ensemble_max_abs_error": raw_error, "account_submission_performed": False}


def audit(root: Path, output: Path, n_jobs: int = 12) -> dict:
    root, output = root.resolve(), output.resolve()
    run = read_json(output / "run.json")
    require(run.get("status") == "complete", "Refusing to audit an incomplete experiment")
    plan = read_json(output / "plan.json")
    folds = plan["folds"]
    selection = read_json(output / "selection.json")
    candidates = list(selection["candidates"])
    require(len(candidates) == 4 and all(name.startswith("O") for name in candidates),
            "Expected exactly the four frozen O0/O1/O2/O3 candidates")
    require({name.split("_", 1)[0] for name in candidates} == {"O0", "O1", "O2", "O3"},
            "Optimization candidate identities changed")
    require(set(plan["candidates"]) == set(candidates), "Plan candidate matrix differs from selection")
    models = candidates + [REFERENCE]
    scores = pd.read_csv(output / "per_date_metrics.csv", parse_dates=["date"])
    metrics = pd.read_csv(output / "metrics.csv")
    require(set(scores.phase) == {"cv", "reused_development"}, "Historical phases mislabeled")
    require(set(scores.model) == set(models), "Unexpected scoring models")
    require(not scores.duplicated(["phase", "fold", "date", "model"]).any(), "Duplicate scores")
    require(np.isfinite(scores[["CORR", "NCORR", "n_rows"]].to_numpy()).all(), "Nonfinite scores")
    require(scores[["CORR", "NCORR"]].abs().le(1 + TOL).all().all(), "Invalid correlations")
    require((scores.n_rows > 1).all() and scores.n_rows.eq(np.floor(scores.n_rows)).all(), "Invalid row counts")
    require({f["id"] for f in folds} == set(scores.fold), "Frozen fold coverage differs")
    require(len({f["id"] for f in folds}) == len(folds), "Duplicate frozen fold")
    require(run["completed_folds"] == [f["id"] for f in folds], "Recorded completed folds differ from plan")
    date_grid = scores.loc[scores.model.eq(models[0]), ["fold", "date"]]
    require(not date_grid.date.duplicated().any(), "Evaluation fold dates overlap")
    summaries = check_summary(scores, metrics)
    winner = check_selection(scores, selection, candidates)
    require(run.get("selected_model", winner) == winner, "Run winner differs from selection")
    started = pd.Timestamp(run["started_utc"])
    frozen = pd.Timestamp(plan["frozen_at_utc"])
    selected_at = pd.Timestamp(selection["frozen_at_utc"])
    completed = pd.Timestamp(run["completed_utc"])
    require(started <= frozen <= selected_at <= completed, "Invalid plan/selection/run chronology")
    source_hashes = verify_source_hashes(plan["source_provenance"]["source_sha256"], root)
    pinned_snapshot = check_pinned_snapshot(root, plan["source_provenance"]["source_sha256"])
    code_hashes = verify_source_hashes(run["code_sha256"], Path(__file__).resolve().parent)
    output_hashes = verify_source_hashes(run["output_sha256"], output)
    pool_path = root / plan["source_provenance"]["cache_path"]
    require(sha256(pool_path) == plan["source_provenance"]["cache_sha256"], "Research-pool cache changed")
    pool_keys = pd.read_parquet(pool_path, columns=KEYS)
    pool_keys["date"] = pd.to_datetime(pool_keys.date)
    pool_keys = pool_keys.sort_values(["date", "numerai_ticker"], kind="stable").reset_index(drop=True)
    require(not pool_keys.duplicated(KEYS).any() and len(pool_keys) == plan["source_provenance"]["rows"],
            "Invalid source-pool keys/counts")
    metadata = read_json(root / "data/raw/v6.0/features.json")
    feature_sets = {name: [f for f in metadata["feature_sets"][name] if f != "feature_country"]
                    for name in ("small", "medium")}
    union = list(dict.fromkeys(feature_sets["small"] + feature_sets["medium"]))
    audits = []
    for fold in folds:
        fold_audit = check_fold_predictions(root, output, fold, scores, models)
        training = read_json(output / "folds" / fold["id"] / "training.json")
        require(training["fold"] == fold, "Fold training plan differs")
        require({r["model"] for r in training["models"]} == set(candidates), "Fold training coverage differs")
        date = pd.Timestamp(fold["eval_start"])
        raw_path = root / "data/raw/v6.0/validation.parquet"
        evaluation = pd.read_parquet(raw_path, columns=KEYS + union + [TARGET, "data_type"],
                                      filters=parquet_date_filter(raw_path, date))
        evaluation["date"] = pd.to_datetime(evaluation.date)
        evaluation = evaluation.loc[evaluation.data_type.eq("validation") & evaluation[TARGET].notna()]
        evaluation = evaluation.sort_values("numerai_ticker").reset_index(drop=True)
        saved = pd.read_parquet(output / "folds" / fold["id"] / "predictions.parquet",
                                filters=parquet_date_filter(output / "folds" / fold["id"] / "predictions.parquet", date))
        saved = saved.sort_values("numerai_ticker").reset_index(drop=True)
        require(np.array_equal(saved.numerai_ticker.to_numpy(), evaluation.numerai_ticker.to_numpy()),
                "Model replay sample keys differ")
        cutoff = pd.Timestamp(fold["train_cutoff_exclusive"])
        eligible = pool_keys.loc[pool_keys.date.lt(cutoff)]
        fold_audit["bundles"] = []
        for model in candidates:
            bundle_path = output / "folds" / fold["id"] / "models" / model / "bundle.json"
            bundle = read_json(bundle_path)
            record = next(r for r in training["models"] if r["model"] == model)
            require(all(record.get(key) == value for key, value in bundle.items()),
                    "Training record differs from serialized bundle")
            fold_audit["bundles"].append(check_bundle(bundle_path, model, feature_sets, eligible,
                date, cutoff, evaluation=evaluation, expected_predictions=saved[model].to_numpy(), n_jobs=n_jobs))
        expected_keys = pool_keys.loc[pool_keys.date.between(date, pd.Timestamp(fold["eval_end"]))]
        require(fold_audit["prediction_rows"] == len(expected_keys) and
                fold_audit["prediction_dates"] == expected_keys.date.nunique(),
                "Fold prediction population differs from mature research pool")
        audits.append(fold_audit)
        print(f"Independent score audit passed: {fold['id']}", flush=True)
    final_plan = plan["final_fit"]
    live_path = Path(final_plan["live_data"])
    require(sha256(live_path) == final_plan["live_data_sha256"], "Pinned live dataset changed")
    live = pd.read_parquet(live_path, columns=KEYS + union)
    live["date"] = pd.to_datetime(live.date)
    date = pd.Timestamp(final_plan["live_date"])
    cutoff = pd.Timestamp(final_plan["train_cutoff_exclusive"])
    require(live.date.nunique() == 1 and live.date.iloc[0] == date and
            cutoff == date - pd.offsets.BDay(65) and final_plan["embargo_weekdays"] == 65,
            "Final live date/maturity cutoff differs")
    final_training = read_json(output / "final_training.json")
    require(final_training["selected_model"] == winner and
            final_training["selection_sha256"] == sha256(output / "selection.json"),
            "Final model is not linked to frozen selection")
    eligible = pool_keys.loc[pool_keys.date.lt(cutoff)]
    require(final_training["full_training_rows"] == len(eligible) and
            pd.Timestamp(final_training["full_training_date_max"]) == eligible.date.max(),
            "Final eligible training population differs")
    preview = pd.read_parquet(output / "final_live_preview.parquet")
    preview["date"] = pd.to_datetime(preview.date)
    require(preview[KEYS].equals(live[KEYS]) and not preview.duplicated(KEYS).any(),
            "Final live preview keys differ from pinned live data")
    final_path = output / "final_model" / winner / "bundle.json"
    require(Path(run["final_bundle"]).resolve() == final_path.resolve(), "Run final bundle path differs")
    final_audit = check_bundle(final_path, winner, feature_sets, eligible, date, cutoff,
                               evaluation=live, expected_predictions=preview[winner].to_numpy(), n_jobs=n_jobs)
    reference_audit = check_reference(output, plan, scores)
    return {"status": "passed", "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_root": str(root), "output": str(output), "selected_model": winner,
            "summary_records_recomputed": summaries, "candidate_count": len(candidates),
            "all_saved_prediction_CORR_recomputed": True,
            "NCORR_scope": "One complete date in every frozen evaluation fold",
            "folds": audits, "final_bundle": final_audit,
            "uploaded_reference_comparison": reference_audit,
            "verified_source_hashes": source_hashes, "verified_code_hashes": code_hashes,
            "verified_output_hashes": output_hashes, "validator_sha256": sha256(Path(__file__)),
            "pinned_snapshot": pinned_snapshot,
            "scoring_input_sha256": pinned_snapshot["scoring_input_sha256"],
            "references": {"official_sources": plan.get("official_sources", []),
                            "snapshot_reference_sha256": pinned_snapshot["reference_sha256"],
                            "uploaded_reference_model_sha256": plan["reference_model"]["model_sha256"]},
            "protocol": {"plan_sha256": sha256(output / "plan.json"),
                         "selection_sha256": sha256(output / "selection.json"),
                         "embargo_weekdays": 65,
                         "calendar": "pandas BDay Monday-Friday; strict less-than cutoff",
                         "final_live_date": str(date.date()),
                         "final_train_cutoff_exclusive": str(cutoff.date())},
            "code_sha256": {**run["code_sha256"], Path(__file__).name: sha256(Path(__file__))},
            "interpretation": "Historical development audit; does not establish live performance"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Pinned dataset snapshot root; defaults to recorded run data_root")
    parser.add_argument("--output", type=Path, required=True, help="Completed experiment directory")
    parser.add_argument("--live-output", type=Path, help="Optional completed optimized live handoff directory")
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    require(args.n_jobs > 0, "Thread count must be positive")
    root = args.root or Path(read_json(args.output / "run.json")["data_root"])
    with threadpool_limits(limits=args.n_jobs):
        result = audit(root, args.output, args.n_jobs)
        if args.live_output:
            result["live_handoff"] = check_live_handoff(args.output, args.live_output, args.n_jobs)
    (args.output / "validation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected_model": result["selected_model"],
                      "folds": len(result["folds"]), "summaries": result["summary_records_recomputed"]}), flush=True)


if __name__ == "__main__":
    main()
