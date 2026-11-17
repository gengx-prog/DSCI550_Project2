"""Refit a completed experiment's frozen CV winner for a new live date.

Uses a supplied v6 snapshot and public live parquet only. Candidate selection
is never rerun. No network access, authentication or account submission occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from generate_xinchen_live import check_live
from optimized_models import (KEYS, TARGET, MODEL_IDS, candidate_specs,
                              predict_bundle, train_predict_candidates)
from paper_protocol import load_research_pool
from run_experiments import PeakMemoryMonitor, sha256_file, write_json

HERE = Path(__file__).resolve().parent
EMBARGO_WEEKDAYS = 65


def checked_selection(path: Path) -> tuple[dict, dict]:
    """Check the lightweight completed-run receipts, without loading old models."""
    path = Path(path).resolve()
    if path.name != "selection.json":
        raise ValueError("--selection must identify the completed experiment's selection.json")
    run_path, plan_path = path.parent / "run.json", path.parent / "plan.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    source_run = json.loads(run_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (source_run.get("status") != "complete" or source_run.get("data_version") != "v6.0"
            or source_run.get("target") != TARGET):
        raise ValueError("Selection must belong to a completed v6.0 Jupiter60 experiment")
    hashes = source_run.get("output_sha256", {})
    for name, candidate_path in [("selection.json", path), ("plan.json", plan_path)]:
        if hashes.get(name) != sha256_file(candidate_path):
            raise ValueError(f"Completed source run hash mismatch: {name}")
    selected = selection.get("selected_model")
    if (selected not in MODEL_IDS or selection.get("selection_phase") != "cv"
            or selection.get("development_used_for_selection") is not False
            or source_run.get("selected_model") != selected):
        raise ValueError("Expected a frozen CV winner selected without development scores")
    source_n_jobs = source_run.get("n_jobs")
    expected_specs = candidate_specs(source_n_jobs)
    if plan.get("candidates") != expected_specs:
        raise ValueError("Source plan differs from the four fixed optimized-model candidates")
    records = selection.get("candidates", {})
    if set(records) != set(MODEL_IDS):
        raise ValueError("Source selection must retain all four candidate records")
    for name, record in records.items():
        score = record.get("selection_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score):
            raise ValueError(f"Invalid source CV selection score: {name}")
    winner = sorted(MODEL_IDS, key=lambda name: (-records[name]["selection_score"], name))[0]
    if winner != selected:
        raise ValueError("Frozen selected model disagrees with its saved CV candidate scores")
    source_model_hash = source_run.get("code_sha256", {}).get("optimized_models.py")
    if source_model_hash is not None and source_model_hash != sha256_file(HERE / "optimized_models.py"):
        raise ValueError("Current optimized-model implementation differs from the frozen source run")
    provenance = {
        "selected_model": selected, "selection_file": path,
        "selection_sha256": sha256_file(path), "source_run_file": run_path,
        "source_run_sha256": sha256_file(run_path), "source_plan_file": plan_path,
        "source_plan_sha256": sha256_file(plan_path),
        "selected_record": records[selected], "source_status": "complete",
        "selection_phase": "cv", "development_used_for_selection": False,
        "historical_models_required": False,
    }
    return selection, provenance


def checked_live(path: Path, features: list[str]) -> tuple[pd.DataFrame, pd.Timestamp]:
    frame = pd.read_parquet(path, columns=KEYS + features)
    check_live(frame, features)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.date.isna().any() or frame.date.nunique() != 1:
        raise ValueError("Live input must contain exactly one complete prediction date")
    live_date = frame.date.iloc[0]
    if live_date.tzinfo is not None or live_date != live_date.normalize():
        raise ValueError("Live date must be a timezone-naive calendar date")
    return frame, live_date


def latest_input_manifests(root: Path) -> dict:
    paths = [root / "snapshot.json", root / "data/raw/v6.0/manifest.json",
             root / "data/processed/v6.0/split_manifest.json",
             root / "data/processed/v6.0/research_pool_manifest.json"]
    return {str(path.relative_to(root)).replace("\\", "/"):
            {"path": path, "sha256": sha256_file(path)} for path in paths if path.is_file()}


def run(args, monitor) -> dict:
    """Train only the frozen winner on strictly mature labels for this live date."""
    candidate_specs(args.n_jobs)
    output, root = args.output.resolve(), args.data_root.resolve()
    selection_path, live_path = args.selection.resolve(), args.live_data.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Preserving existing run: {output}; choose a fresh --output")
    if (output == root or root.is_relative_to(output)
            or selection_path.is_relative_to(output) or live_path.is_relative_to(output)):
        raise ValueError("Output must not contain or replace the input snapshot or source receipts")
    output.mkdir(parents=True, exist_ok=True)
    monitor.run_started = True
    started = time.perf_counter()
    record = {
        "status": "started", "started_utc": datetime.now(timezone.utc),
        "kind": "frozen_cv_winner_weekly_refit", "data_version": "v6.0", "target": TARGET,
        "data_root": root, "selection_file": selection_path, "live_data": live_path,
        "n_jobs": args.n_jobs, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["lightgbm", "numpy", "pandas", "pyarrow"]},
        "code_sha256": {name: sha256_file(HERE / name) for name in
                        [Path(__file__).name, "optimized_models.py", "paper_protocol.py",
                         "run_experiments.py", "generate_xinchen_live.py"]},
        "candidate_selection_performed": False, "account_submission_performed": False,
        "scope": "Refit the already frozen CV winner; historical experiment is not rerun",
    }
    write_json(output / "run.json", record)
    try:
        selection, selection_provenance = checked_selection(selection_path)
        selected = selection["selected_model"]
        metadata_path = root / "data/raw/v6.0/features.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_set = candidate_specs(args.n_jobs)[selected]["feature_set"]
        features = [name for name in metadata["feature_sets"][feature_set] if name != "feature_country"]
        if not features or len(features) != len(set(features)):
            raise ValueError("Invalid or duplicate metadata feature names")
        live, live_date = checked_live(live_path, features)
        cutoff = live_date - pd.offsets.BDay(EMBARGO_WEEKDAYS)
        pool, feature_sets, source_provenance = load_research_pool(root)
        if feature_sets.get(feature_set) != features:
            raise ValueError("Research-pool feature order differs from current metadata")
        dates = pd.to_datetime(pool["date"], errors="raise")
        if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
            raise ValueError("Research-pool dates must be complete calendar dates")
        train = pool.loc[dates.lt(cutoff)].copy()
        if train.empty or pd.to_datetime(train.date).max() + pd.offsets.BDay(EMBARGO_WEEKDAYS) >= live_date:
            raise ValueError("No valid training pool satisfies the strict 65-weekday maturity embargo")
        record.update(selected_model=selected, selection_provenance=selection_provenance,
                      live_date=live_date, live_data_sha256=sha256_file(live_path),
                      feature_metadata_sha256=sha256_file(metadata_path),
                      source_provenance=source_provenance,
                      input_manifests=latest_input_manifests(root),
                      train_cutoff_exclusive=cutoff, embargo_weekdays=EMBARGO_WEEKDAYS,
                      embargo_calendar="pandas BDay Monday-Friday",
                      full_training_rows=len(train), full_training_date_min=pd.to_datetime(train.date).min(),
                      full_training_date_max=pd.to_datetime(train.date).max(),
                      excluded_at_or_after_maturity_cutoff=int((~dates.lt(cutoff)).sum()))
        write_json(output / "run.json", record)
        print(f"Refitting frozen {selected}: {len(train):,} eligible rows, "
              f"training dates strictly before {cutoff.date()}, live {live_date.date()}", flush=True)
        predictions, training = train_predict_candidates(train, live, feature_sets,
            output / "final_model", n_jobs=args.n_jobs, candidate_ids=[selected])
        if len(training) != 1 or training[0]["model"] != selected:
            raise ValueError("Refitting must produce exactly the frozen selected candidate")
        bundle_path = output / "final_model" / selected / "bundle.json"
        repeated = predict_bundle(bundle_path, live, n_jobs=args.n_jobs)
        values = predictions[selected].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or len(values) != len(live)
                or not np.array_equal(repeated, values)):
            raise ValueError("Saved selected-model predictions do not exactly reproduce the live preview")
        training_record = {"selected_model": selected, "live_date": live_date,
            "train_cutoff_exclusive": cutoff, "embargo_weekdays": EMBARGO_WEEKDAYS,
            "full_training_rows": len(train), "full_training_date_max": pd.to_datetime(train.date).max(),
            "models": training, "selection_sha256": selection_provenance["selection_sha256"]}
        write_json(output / "final_training.json", training_record)
        predictions.to_parquet(output / "final_live_preview.parquet", index=False)
        record.update(status="complete", completed_utc=datetime.now(timezone.utc),
            elapsed_seconds=time.perf_counter() - started, final_bundle=bundle_path,
            final_bundle_sha256=sha256_file(bundle_path), live_rows=len(live),
            model_reload_prediction_max_abs_error=0.0,
            output_sha256={name: sha256_file(output / name) for name in
                           ["final_training.json", "final_live_preview.parquet"]})
        write_json(output / "run.json", record)
        print(f"Refit complete: {bundle_path}", flush=True)
        return record
    except BaseException as exc:
        record.update(status="failed", failed_utc=datetime.now(timezone.utc),
                      error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run.json", record)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--live-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    monitor = PeakMemoryMonitor()
    try:
        with monitor, threadpool_limits(limits=args.n_jobs):
            run(args, monitor)
    finally:
        if monitor.run_started:
            path = args.output.resolve() / "run.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result["memory"] = monitor.result()
            write_json(path, result)


if __name__ == "__main__":
    main()
