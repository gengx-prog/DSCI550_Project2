"""Independently audit a fixed-baseline live prediction handoff, without submitting it.

The default check is offline and uses the original downloaded live Parquet file.
An explicit --check-current-round additionally queries Numerai's public API.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.stats import rankdata


VERSION = "v6.0"
MODEL = "B0_small_main"
COLUMNS = ["numerai_ticker", "prediction"]
TRANSFORM = "pandas.Series.rank(method='average', pct=True)"
TOLERANCE = 1e-14


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path.name}")
    return value


def validate_tickers(values: pd.Series, source: str) -> list[str]:
    require(len(values) > 0, f"{source}: no tickers")
    require(values.notna().all(), f"{source}: null ticker")
    require(values.map(lambda value: isinstance(value, str)).all(),
            f"{source}: ticker must be a string")
    require(values.map(lambda value: bool(value) and value == value.strip()).all(),
            f"{source}: blank ticker or surrounding whitespace")
    require(not values.duplicated().any(), f"{source}: duplicate ticker")
    return values.tolist()


def validate_features(frame: pd.DataFrame, features: list[str]) -> int:
    require(len(features) == len(set(features)) and bool(features),
            "Feature names must be nonempty and unique")
    require(all(isinstance(name, str) and name.startswith("feature_") for name in features),
            "Invalid feature name")
    require("feature_country" not in features, "Country feature is outside the numeric baseline")
    require(set(features).issubset(frame.columns), "Live data is missing model features")
    missing = 0
    for feature in features:
        column = frame[feature]
        require(is_numeric_dtype(column.dtype) and not is_bool_dtype(column.dtype),
                f"Non-numeric feature: {feature}")
        numeric = column.to_numpy(dtype=np.float64, na_value=np.nan)
        require(not np.isinf(numeric).any(), f"Infinite feature value: {feature}")
        missing += int(np.isnan(numeric).sum())
    return missing


def compare_predictions(csv_frame: pd.DataFrame, live_tickers: list[str],
                        raw_predictions: np.ndarray) -> dict:
    require(list(csv_frame.columns) == COLUMNS, "CSV columns must be numerai_ticker,prediction")
    csv_tickers = validate_tickers(csv_frame["numerai_ticker"], "CSV")
    require(len(csv_tickers) == len(live_tickers), "CSV/live row counts differ")
    require(set(csv_tickers) == set(live_tickers), "CSV has missing or extra live tickers")
    values = pd.to_numeric(csv_frame["prediction"], errors="raise").to_numpy(dtype=np.float64)
    require(np.isfinite(values).all(), "Predictions must be finite")
    require(((values >= 0) & (values <= 1)).all(), "Predictions must be within [0,1]")
    require(np.unique(values).size > 1, "CSV predictions must not be constant")
    raw_predictions = np.asarray(raw_predictions, dtype=np.float64)
    require(raw_predictions.ndim == 1 and len(raw_predictions) == len(live_tickers),
            "Model prediction dimensions differ from live rows")
    require(np.isfinite(raw_predictions).all(), "Model produced nonfinite predictions")
    require(np.unique(raw_predictions).size > 1, "Model predictions must not be constant")
    expected = pd.Series(rankdata(raw_predictions, method="average") / len(raw_predictions),
                         index=live_tickers)
    error = np.abs(values - expected.loc[csv_tickers].to_numpy())
    maximum_error = float(error.max())
    require(maximum_error <= TOLERANCE,
            f"CSV differs from independent saved-model inference/ranking: {maximum_error}")
    return {"rows": len(values), "maximum_absolute_error": maximum_error,
            "absolute_tolerance": TOLERANCE, "minimum_prediction": float(values.min()),
            "maximum_prediction": float(values.max()),
            "independent_transform": "scipy.stats.rankdata(method='average') / number of live rows"}


def validate_bundle(bundle: Path, live: Path, check_current_round: bool = False) -> dict:
    bundle, live = bundle.resolve(), live.resolve()
    require(bundle.is_dir(), "Bundle directory does not exist")
    require(live.is_file(), "Original live Parquet file does not exist")
    generation = read_json(bundle / "live_generation.json")
    manifest = read_json(bundle / "live_data_manifest.json")
    run = read_json(bundle / "baseline_run.json")
    protocol = read_json(bundle / "baseline_protocol.json")
    metadata = read_json(bundle / "features.json")
    round_number = generation.get("round")
    require(type(round_number) is int and round_number > 0, "Invalid generated round")
    require(generation.get("status") == manifest.get("status") == "complete",
            "Generation/download records are incomplete")
    require(generation.get("data_version") == manifest.get("data_version") == VERSION,
            "Unexpected data version")
    require(manifest.get("round") == round_number, "Download and generation rounds differ")
    require(generation.get("current_round_before") == generation.get("current_round_after") == round_number,
            "Round changed during live generation")
    require(generation.get("columns") == COLUMNS, "Recorded CSV columns differ")
    require(generation.get("prediction_transform") == TRANSFORM, "Unexpected prediction transform")
    require(run.get("status") == "complete" and run.get("model") == protocol.get("model") == MODEL,
            "Incomplete or unexpected source baseline")
    require(run.get("data_version") == protocol.get("data_version") == VERSION,
            "Baseline data version differs")
    require(run.get("target") == protocol.get("target") == "target_jupiter_60",
            "Baseline target differs")
    if "round_info" in manifest or "round_info" in generation:
        round_info = generation.get("round_info")
        require(isinstance(round_info, dict) and round_info == manifest.get("round_info"),
                "Generation/download round metadata differs")
        require(round_info.get("number") == round_number
                and round_info.get("target") == "target_jupiter_60",
                "Round metadata number or target differs")
    if "tournament" in manifest:
        recorded_tournament = generation.get("tournament", generation.get("round_info", {}).get("tournament"))
        require(type(manifest["tournament"]) is int and manifest["tournament"] > 0
                and manifest["tournament"] == recorded_tournament,
                "Generation/download tournament differs")
    artifacts = {"model_sha256": bundle / "B0_small_main.txt",
                 "protocol_sha256": bundle / "baseline_protocol.json",
                 "baseline_run_sha256": bundle / "baseline_run.json",
                 "feature_metadata_sha256": bundle / "features.json",
                 "live_parquet_sha256": live,
                 "download_manifest_sha256": bundle / "live_data_manifest.json"}
    hashes = {}
    for field, path in artifacts.items():
        hashes[field] = sha256_file(path)
        require(generation.get(field) == hashes[field], f"Recorded artifact hash differs: {field}")
    require(run.get("model_sha256") == hashes["model_sha256"], "Model differs from recorded baseline")
    require(run.get("protocol_sha256") == hashes["protocol_sha256"], "Protocol differs from baseline run")
    downloads = manifest.get("files")
    require(isinstance(downloads, list) and len(downloads) == 2, "Expected two original live download objects")
    by_filename = {}
    for item in downloads:
        require(isinstance(item, dict), "Invalid download manifest entry")
        name = item.get("filename")
        require(name in {"live.parquet", "features.json"} and name not in by_filename,
                "Unexpected/duplicate downloaded filename")
        require(item.get("dataset") == f"{VERSION}/{name}" and item.get("round") == round_number,
                "Download item dataset or round differs")
        url = urlsplit(item.get("url", ""))
        require(url.scheme == "https" and bool(url.hostname) and not url.username
                and not url.password and not url.query and not url.fragment,
                "Download URL must be sanitized HTTPS without credentials/query/fragment")
        path = live if name == "live.parquet" else bundle / "features.json"
        require(item.get("bytes") == path.stat().st_size and item.get("sha256") == sha256_file(path),
                f"Downloaded object bytes/hash differs: {name}")
        by_filename[name] = item
    features = generation.get("features")
    require(isinstance(features, list), "Missing ordered model features")
    small = metadata.get("feature_sets", {}).get("small")
    require(isinstance(small, list), "Missing official small feature list")
    numeric_small = [name for name in small if name != "feature_country"]
    booster = lgb.Booster(model_file=str(bundle / "B0_small_main.txt"))
    require(features == run.get("features") == protocol.get("features") == numeric_small
            == booster.feature_name(), "Model/run/protocol/official feature order differs")
    require(len(features) == 100 and booster.num_feature() == 100,
            "Fixed baseline must have 100 numeric small features")
    require(booster.num_trees() == 300, "Fixed baseline must have 300 trees")
    frame = pd.read_parquet(live, columns=["numerai_ticker"] + features)
    live_tickers = validate_tickers(frame["numerai_ticker"], "Official live data")
    require(generation.get("live_rows") == len(frame), "Recorded live row count differs")
    missing_feature_values = validate_features(frame, features)
    csv_name = generation.get("csv_file")
    require(isinstance(csv_name, str) and Path(csv_name).name == csv_name
            and csv_name == f"XinchenGeng_baseline_live_round{round_number}.csv",
            "Unexpected CSV filename")
    csv_path = bundle / csv_name
    require(sha256_file(csv_path) == generation.get("csv_sha256"), "CSV hash differs")
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        require(next(csv.reader(source), None) == COLUMNS, "CSV header differs")
    csv_frame = pd.read_csv(csv_path, dtype={"numerai_ticker": str}, keep_default_na=False)
    raw_predictions = booster.predict(frame[features], num_threads=1)
    comparison = compare_predictions(csv_frame, live_tickers, raw_predictions)
    observed_round = None
    if check_current_round:
        from numerapi import NumerAPI
        observed_round = NumerAPI(show_progress_bars=False).get_current_round()
        require(observed_round == round_number,
                f"Bundle round {round_number} is no longer current (API: {observed_round})")
    return {"status": "passed", "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "round": round_number, "data_version": VERSION,
            "generation_sha256": sha256_file(bundle / "live_generation.json"),
            "artifact_sha256": hashes, "csv_file": csv_name,
            "csv_sha256": sha256_file(csv_path), "exact_live_ticker_coverage": True,
            "feature_count": len(features), "allowed_missing_feature_values": missing_feature_values,
            "full_model_and_rank_check": comparison,
            "current_round_queried": check_current_round, "current_round_at_validation": observed_round,
            "scope": "Independent saved-model live CSV validation only; no Numerai account submission."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True,
                        help="Original downloaded live.parquet, retained outside the handoff")
    parser.add_argument("--check-current-round", action="store_true",
                        help="Explicitly query Numerai's public current-round API")
    parser.add_argument("--report", type=Path,
                        help="Fresh report path; default is bundle/validation.json. Existing reports are preserved.")
    args = parser.parse_args()
    destination = args.report if args.report is not None else args.bundle / "validation.json"
    require(not destination.exists(), "Report already exists; preserve the earlier audit and choose --report PATH")
    report = validate_bundle(args.bundle, args.live, args.check_current_round)
    with destination.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")
    print(f"Independent live validation passed: round {report['round']}, "
          f"{report['full_model_and_rank_check']['rows']:,} predictions")


if __name__ == "__main__":
    main()
