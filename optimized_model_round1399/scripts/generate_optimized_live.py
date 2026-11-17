"""Generate and independently recheck a round-pinned v6 optimized-model CSV.

Only public datasets are accessed. This entry point never authenticates, uploads
predictions, changes stake, or trains a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import shutil
import sys
from zoneinfo import ZoneInfo

import lightgbm as lgb
from numerapi import NumerAPI
import numpy as np
import pandas as pd

from generate_xinchen_live import (
    VERSION, LOCAL_ZONE, check_live, current_round, download, read_json,
    sha256, utc_now, write_json,
)
from optimized_models import predict_bundle

TARGET = "target_jupiter_60"
TRANSFORM = "equal member average-percentile ranks within date; final average-percentile rank"


def checked_round(api: NumerAPI, requested: int | None) -> tuple[int, dict]:
    """Require the current Classic round and its explicit 60-day target."""
    if api.tournament_id != 8:
        raise ValueError("Expected Numerai Classic tournament 8")
    before = current_round(api)
    number = before if requested is None else requested
    if number != before:
        raise ValueError("Requested round is no longer the current round")
    entries = [entry for entry in api.list_rounds(number=number)
               if entry.get("number") == number and entry.get("tournament") == 8
               and entry.get("target") == TARGET]
    if len(entries) != 1:
        raise ValueError("Round metadata does not uniquely match tournament 8 and Jupiter60")
    info = {key: entries[0].get(key) for key in (
        "number", "tournament", "target", "dataDatestamp", "openTime",
        "closeTime", "closeStakingTime",
    )}
    info = {key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in info.items()}
    closes = pd.Timestamp(info["closeTime"])
    if pd.isna(closes) or closes.tzinfo is None:
        raise ValueError("Round closeTime must be an explicit timezone-aware timestamp")
    if closes.to_pydatetime() <= datetime.now(timezone.utc):
        raise ValueError("Current round's general submission window has closed")
    required = {f"{VERSION}/features.json", f"{VERSION}/live.parquet"}
    if not required.issubset(api.list_datasets(round_num=number)):
        raise ValueError("Current round does not expose the required v6.0 live datasets")
    return before, info


def checked_bundle(path: Path) -> dict:
    bundle = read_json(path)
    features = bundle.get("features")
    if bundle.get("schema_version") != 1 or bundle.get("target") != TARGET:
        raise ValueError("Expected a version-1 Jupiter60 optimized-model bundle")
    if (not isinstance(features, list) or not features
            or len(features) != len(set(features))
            or any(not isinstance(name, str) or not name.startswith("feature_")
                   or name == "feature_country" for name in features)):
        raise ValueError("Bundle must contain unique ordered numeric feature names")
    if not bundle.get("members") or not bundle.get("model_id"):
        raise ValueError("Bundle must identify its model and members")
    return bundle


def checked_live(frame: pd.DataFrame, bundle: dict, metadata: dict,
                 round_info: dict) -> pd.DataFrame:
    """Validate the actual dataset date and the final fit's strict label embargo."""
    features = bundle["features"]
    feature_set = bundle.get("feature_set")
    published = metadata.get("feature_sets", {}).get(feature_set)
    if not isinstance(published, list):
        raise ValueError("Live metadata is missing the bundle's feature set")
    published = [name for name in published if name != "feature_country"]
    if published != features:
        raise ValueError("Live feature-set order differs from the saved model bundle")
    check_live(frame, features)
    if "date" not in frame:
        raise ValueError("Live dataset must include its actual date")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.date.isna().any() or frame.date.nunique() != 1:
        raise ValueError("Live dataset must contain exactly one non-null date")
    live_date = frame.date.iloc[0]
    if live_date != live_date.normalize() or live_date.tzinfo is not None:
        raise ValueError("Live date must be a timezone-naive calendar date")
    expected = pd.to_datetime(str(round_info["dataDatestamp"]), format="%Y%m%d", errors="raise")
    if live_date != expected:
        raise ValueError("Live date differs from the pinned round's dataDatestamp")
    training_max = pd.Timestamp(bundle.get("full_training_date_max"))
    cutoff = live_date - pd.offsets.BDay(65)
    if pd.isna(training_max) or training_max >= cutoff:
        raise ValueError("Saved model violates the strict 65-weekday live label embargo")
    return frame


def checked_submission(frame: pd.DataFrame, raw: np.ndarray) -> pd.DataFrame:
    values = np.asarray(raw, dtype=float)
    if (values.shape != (len(frame),) or not np.isfinite(values).all()
            or np.unique(values).size < 2):
        raise ValueError("Invalid or constant optimized-model live predictions")
    result = pd.DataFrame({
        "numerai_ticker": frame.numerai_ticker.to_numpy(),
        "prediction": pd.Series(values).rank(method="average", pct=True).to_numpy(),
    })
    validate_submission(result, frame, values)
    return result


def validate_submission(submission: pd.DataFrame, frame: pd.DataFrame,
                        raw: np.ndarray) -> None:
    """Check CSV identities, finite ranks and independently reconstructed ties."""
    if submission.columns.tolist() != ["numerai_ticker", "prediction"]:
        raise ValueError("Submission must contain only numerai_ticker,prediction")
    if (len(submission) != len(frame) or submission.numerai_ticker.isna().any()
            or submission.numerai_ticker.duplicated().any()
            or submission.numerai_ticker.tolist() != frame.numerai_ticker.tolist()):
        raise ValueError("Submission ticker coverage/order differs from the live dataset")
    values = submission.prediction.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values > 0) & (values <= 1)).all():
        raise ValueError("Submission predictions must be finite percentiles in (0,1]")
    # scipy's rankdata provides a second implementation of average ranks.
    from scipy.stats import rankdata
    expected = rankdata(np.asarray(raw), method="average") / len(raw)
    if not np.array_equal(values, expected):
        raise ValueError("Submission predictions differ from independent average ranks")


def generate(args: argparse.Namespace) -> dict:
    api = NumerAPI(show_progress_bars=False)
    before, info = checked_round(api, args.round_number)
    number = info["number"]
    source = args.bundle.resolve()
    bundle = checked_bundle(source)
    output = args.output.resolve()
    if output == source.parent or output.is_relative_to(source.parent) or source.parent.is_relative_to(output):
        raise ValueError("Output and source bundle directories must be separate")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Preserving existing files: {output}; choose a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    data_dir.mkdir()
    record = {
        "status": "started", "data_version": VERSION, "round": number,
        "model_name": bundle["model_id"], "target": TARGET,
        "started_at_utc": utc_now(), "timezone": LOCAL_ZONE,
        "run_date_local": datetime.now(ZoneInfo(LOCAL_ZONE)).date().isoformat(),
        "current_round_before": before, "round_info": info,
        "generation_mode": "current_round", "submission_window_open_at_generation": True,
        "bundle_source": str(source), "source_bundle_sha256": sha256(source),
        "training_performed": False, "account_submission_performed": False,
    }
    write_json(output / "live_generation.json", record)
    try:
        manifest = {"status": "started", "round": number, "data_version": VERSION,
                    "tournament": 8, "round_info": info, "files": []}
        for name in ("features.json", "live.parquet"):
            entry = download(api, name, number, data_dir / name)
            manifest["files"].append(entry)
            write_json(data_dir / "download_manifest.json", manifest)
        manifest.update(status="complete", completed_at_utc=utc_now())
        write_json(data_dir / "download_manifest.json", manifest)
        features = bundle["features"]
        live = pd.read_parquet(data_dir / "live.parquet", columns=["numerai_ticker", "date"] + features)
        live = checked_live(live, bundle, read_json(data_dir / "features.json"), info)
        copied_model = output / "model"
        shutil.copytree(source.parent, copied_model)
        saved_bundle = copied_model / source.name
        if sha256(saved_bundle) != sha256(source):
            raise ValueError("Copied bundle differs from its source")
        raw = predict_bundle(saved_bundle, live, n_jobs=args.n_jobs)
        submission = checked_submission(live, raw)
        filename = f"Optimized_live_round{number}.csv"
        temporary = output / (filename + ".part")
        submission.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
        roundtrip = pd.read_csv(temporary, dtype={"numerai_ticker": str}, float_precision="round_trip")
        # Reload all member files and recalculate the ensemble from the saved copy.
        repeated = predict_bundle(saved_bundle, live, n_jobs=args.n_jobs)
        if not np.array_equal(raw, repeated):
            raise ValueError("Saved-model reload changed optimized predictions")
        validate_submission(roundtrip, live, repeated)
        audit = live[["numerai_ticker", "date"]].copy()
        audit["raw_prediction"] = raw
        audit["prediction"] = roundtrip.prediction.to_numpy()
        audit.to_parquet(output / "predictions.parquet", index=False)
        after = current_round(api)
        if before != after:
            raise ValueError("Numerai round changed during generation; generate a fresh CSV")
        temporary.replace(output / filename)
        model_hashes = {str(path.relative_to(copied_model)).replace("\\", "/"): sha256(path)
                        for path in sorted(copied_model.rglob("*")) if path.is_file()}
        validation = {
            "status": "passed", "checked_at_utc": utc_now(), "live_rows": len(live),
            "model_reload_prediction_max_abs_error": float(np.max(np.abs(raw - repeated))),
            "csv_roundtrip_exact": True, "independent_average_ranks": "scipy.stats.rankdata(method=average)",
            "ticker_coverage_and_order_exact": True, "strict_65_weekday_label_embargo": True,
            "bundle_sha256": sha256(saved_bundle), "csv_sha256": sha256(output / filename),
        }
        write_json(output / "validation.json", validation)
        record.update(
            status="complete", generated_at_utc=utc_now(), current_round_after=after,
            live_date=live.date.iloc[0].date().isoformat(), live_rows=len(live),
            features=features, n_features=len(features), csv_file=filename,
            columns=submission.columns.tolist(), prediction_transform=TRANSFORM,
            unique_predictions=int(submission.prediction.nunique()),
            prediction_min=float(submission.prediction.min()), prediction_max=float(submission.prediction.max()),
            raw_prediction_min=float(np.min(raw)), raw_prediction_max=float(np.max(raw)),
            bundle_sha256=sha256(saved_bundle), model_files_sha256=model_hashes,
            csv_sha256=sha256(output / filename), predictions_sha256=sha256(output / "predictions.parquet"),
            live_parquet_sha256=sha256(data_dir / "live.parquet"),
            feature_metadata_sha256=sha256(data_dir / "features.json"),
            download_manifest_sha256=sha256(data_dir / "download_manifest.json"),
            validation_sha256=sha256(output / "validation.json"),
            generator_sha256=sha256(Path(__file__).resolve()),
            inference_sha256=sha256(Path(__file__).resolve().with_name("optimized_models.py")),
            environment={"python": platform.python_version(), "lightgbm": lgb.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__},
            official_submission_spec="https://docs.numer.ai/numerai-tournament/submissions",
            scope="Current-round optimized prediction handoff. No Numerai account upload or submission receipt.",
        )
        write_json(output / "live_generation.json", record)
        print(json.dumps({"status": "complete", "round": number, "rows": len(live),
                          "csv": str(output / filename), "validation": "passed"}), flush=True)
        return record
    except BaseException as exc:
        record.update(status="failed", failed_at_utc=utc_now(), error_type=type(exc).__name__)
        write_json(output / "live_generation.json", record)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Selected final-fit bundle.json")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--round", dest="round_number", type=int, help="Expected current round")
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    if args.n_jobs < 1 or (args.round_number is not None and args.round_number < 1):
        parser.error("n-jobs and round must be positive")
    logging.disable(logging.CRITICAL)
    try:
        generate(args)
        return 0
    except BaseException as exc:
        response = getattr(exc, "response", None)
        print(f"Optimized live generation failed: {type(exc).__name__}, HTTP {getattr(response, 'status_code', None)}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
