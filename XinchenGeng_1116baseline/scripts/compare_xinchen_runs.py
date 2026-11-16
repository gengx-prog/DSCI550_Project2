"""Compare two validated baseline runs after a historical data refresh.

New dates are observations added to the archived evaluation, not a claim of
unseen-test or live performance. Only the current result directory is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FILES = ("run.json", "validation.json", "metrics.csv", "per_date_metrics.csv", "predictions.parquet")
KEYS = ["numerai_ticker", "date", "split"]
DATES = ["split", "date"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(path: Path) -> dict:
    return {name: sha256(path / name) for name in FILES}


def raw_manifest(root: Path) -> tuple[dict, dict]:
    path = root / "data/raw/v6.0/manifest.json"
    digest = sha256(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest["files"]
    datasets = {record["dataset"]: {"sha256": record["sha256"], "bytes": record["bytes"]}
                for record in records}
    expected = {f"v6.0/{name}" for name in ("features.json", "train.parquet", "validation.parquet",
                "validation_neutralizer.parquet", "validation_example_preds.parquet")}
    require(manifest.get("version") == "v6.0" and len(records) == len(datasets) and
            set(datasets) == expected, f"Expected five unique v6.0 raw datasets: {path}")
    for record in datasets.values():
        require(isinstance(record["bytes"], int) and record["bytes"] >= 0 and
                isinstance(record["sha256"], str) and len(record["sha256"]) == 64 and
                all(c in "0123456789abcdef" for c in record["sha256"]), "Invalid raw manifest record")
    return {"root": str(root), "path": str(path), "sha256": digest}, datasets


def dates(frame: pd.DataFrame) -> None:
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    require(frame["date"].notna().all() and frame["date"].dt.tz is None,
            "Missing or timezone-aware observation date")
    require(frame["date"].eq(frame["date"].dt.normalize()).all(), "Non-midnight observation date")


def load_run(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    audit = json.loads((path / "validation.json").read_text(encoding="utf-8"))
    require(run.get("status") == "complete" and audit.get("status") == "passed",
            f"A completed, independently validated run is required: {path}")
    require(run.get("primary_metric") == "CORR", "Expected the fixed CORR baseline")
    model, target = run["model"], run["target"]
    summary = pd.read_csv(path / "metrics.csv")
    scores = pd.read_csv(path / "per_date_metrics.csv")
    predictions = pd.read_parquet(path / "predictions.parquet", columns=KEYS + [target, model])
    require({"split", "model", "metric", "mean", "std_ddof0"}.issubset(summary.columns),
            "Missing summary columns")
    require(set(DATES + ["model", "n_rows", "CORR", "NCORR"]).issubset(scores.columns),
            "Missing date-level score columns")
    require(set(scores["model"]) == set(summary["model"]) == {model}, "Unexpected scored model")
    require(not summary[["split", "metric"]].isna().any().any() and
            not summary.duplicated(["split", "metric"]).any(), "Invalid summary keys")
    for frame, keys in ((scores, DATES), (predictions, KEYS)):
        dates(frame)
        require(not frame[keys].isna().any().any() and not frame.duplicated(keys).any(),
                f"Missing or duplicate keys in {path}")
    require(predictions["numerai_ticker"].map(lambda value: isinstance(value, str) and bool(value.strip())).all(),
            "Ticker keys must be nonempty strings")
    require(set(scores["split"]) == set(predictions["split"]) == {"tuning", "holdout"},
            "Unexpected split coverage")
    require(set(zip(summary["split"], summary["metric"])) ==
            {(split, metric) for split in ("tuning", "holdout") for metric in ("CORR", "NCORR")},
            "Unexpected summary metric coverage")
    for frame, columns, nullable in ((predictions, [target, model], False),
                                     (scores, ["CORR", "NCORR"], True),
                                     (summary, ["mean", "std_ddof0"], True)):
        for column in columns:
            require(pd.api.types.is_numeric_dtype(frame[column]), f"Nonnumeric column: {column}")
            values = frame[column].dropna() if nullable else frame[column]
            require(np.isfinite(values.to_numpy(dtype=float)).all(), f"Nonfinite column: {column}")
    require(scores["CORR"].notna().all() and predictions[target].between(0, 1).all(),
            "Missing CORR or target outside [0, 1]")
    counts = predictions.groupby(DATES).size().rename("actual_rows")
    aligned = scores.set_index(DATES).join(counts, how="outer")
    require(aligned["actual_rows"].notna().all() and aligned["n_rows"].eq(aligned["actual_rows"]).all(),
            "Prediction/date-level row coverage differs")
    require(len(scores) == audit["all_date_corr"]["dates"] and
            len(predictions) == audit["all_date_corr"]["rows"], "Audit coverage differs")
    return run, summary, scores, predictions


def metric_stats(frame: pd.DataFrame) -> dict:
    result = {"n_dates": len(frame)}
    for metric in ("CORR", "NCORR"):
        values = frame[metric].dropna().to_numpy(dtype=float)
        result[metric] = {"n_scored_dates": len(values),
                          "mean": float(values.mean()) if len(values) else None,
                          "std_ddof0": float(values.std(ddof=0)) if len(values) else None}
    return result


def compare(previous: Path, current: Path, previous_root: Path | None = None,
            current_root: Path | None = None) -> None:
    require(previous != current, "Previous and current directories must differ")
    hashes = {"previous": source_hashes(previous), "current": source_hashes(current)}
    manifests, raw_datasets = {}, {}
    for label, root in (("previous", previous_root), ("current", current_root)):
        if root is not None:
            manifests[label], raw_datasets[label] = raw_manifest(root)
    old, old_metrics, old_scores, old_predictions = load_run(previous)
    new, new_metrics, new_scores, new_predictions = load_run(current)
    require((old["model"], old["target"]) == (new["model"], new["target"]),
            "Runs must use the same model name and target")
    shared = old_predictions.merge(new_predictions, on=KEYS, how="inner",
                                   suffixes=("_previous", "_current"), validate="one_to_one")
    changes = {}
    for label, column in (("target", new["target"]), ("prediction", new["model"])):
        delta = np.abs(shared[f"{column}_current"].to_numpy(dtype=float) -
                       shared[f"{column}_previous"].to_numpy(dtype=float))
        changes[label] = {"changed_rows": int(np.count_nonzero(delta)),
                          "maximum_absolute_difference": float(delta.max()) if len(delta) else None}
    old_index, new_index = old_scores.set_index(DATES), new_scores.set_index(DATES)
    common_dates = old_index.index.intersection(new_index.index)
    added = new_scores.set_index(DATES).loc[new_index.index.difference(old_index.index)].reset_index()
    added = added.sort_values(DATES).reset_index(drop=True)
    score_changes = {}
    for metric in ("CORR", "NCORR"):
        before, after = old_index.loc[common_dates, metric], new_index.loc[common_dates, metric]
        comparable = before.notna() & after.notna()
        delta = np.abs(after[comparable].to_numpy() - before[comparable].to_numpy())
        score_changes[metric] = {"comparable_dates": len(delta), "changed_dates": int(np.count_nonzero(delta)),
                                 "availability_changed_dates": int((before.isna() != after.isna()).sum()),
                                 "maximum_absolute_difference": float(delta.max()) if len(delta) else None}
    input_hashes = [{str(key).replace("\\", "/"): value for key, value in run.get("input_sha256", {}).items()}
                    for run in (old, new)]
    input_comparison = {key: {"previous": input_hashes[0].get(key), "current": input_hashes[1].get(key),
                              "identical": input_hashes[0].get(key) == input_hashes[1].get(key)}
                        for key in sorted(set(input_hashes[0]) | set(input_hashes[1]))}
    summary_columns = ["split", "metric", "mean", "std_ddof0"]
    summary_comparison = old_metrics[summary_columns].merge(new_metrics[summary_columns],
        on=["split", "metric"], suffixes=("_previous", "_current"), validate="one_to_one").astype(object)
    report = {
        "status": "complete", "compared_at_utc": datetime.now(timezone.utc).isoformat(),
        "previous_directory": str(previous), "current_directory": str(current),
        "model": new["model"], "target": new["target"], "input_sha256_comparison": input_comparison,
        "data_roots": {label: manifests.get(label, {}).get("root", run.get("data_root", run.get("root")))
                       for label, run in (("previous", old), ("current", new))},
        "raw_manifest_sources": manifests,
        "raw_manifest_dataset_comparison": {
            dataset: {"previous": raw_datasets.get("previous", {}).get(dataset),
                      "current": raw_datasets.get("current", {}).get(dataset),
                      "identical": (raw_datasets["previous"][dataset] == raw_datasets["current"][dataset])
                                   if len(raw_datasets) == 2 else None}
            for dataset in sorted({key for records in raw_datasets.values() for key in records})},
        "raw_input_sha256_comparison": {k: v for k, v in input_comparison.items() if k.startswith("data/raw/")},
        "partitions": {"previous": old["splits"], "current": new["splits"]},
        "summary_metric_comparison": summary_comparison.where(summary_comparison.notna(), None).to_dict("records"),
        "model_sha256": {"previous": old["model_sha256"], "current": new["model_sha256"],
                         "identical": old["model_sha256"] == new["model_sha256"]},
        "training_sample_keys_sha256": {"previous": old["train_sample_keys_sha256"],
                                        "current": new["train_sample_keys_sha256"],
                                        "identical": old["train_sample_keys_sha256"] == new["train_sample_keys_sha256"]},
        "prediction_keys": {"previous": len(old_predictions), "current": len(new_predictions),
                            "common": len(shared), "added": len(new_predictions) - len(shared),
                            "removed": len(old_predictions) - len(shared)},
        "common_prediction_changes": changes, "common_date_score_changes": score_changes,
        "new_dates": [{"split": row.split, "date": row.date.strftime("%Y-%m-%d")}
                      for row in added.itertuples()],
        "removed_dates": [{"split": split, "date": date.strftime("%Y-%m-%d")}
                          for split, date in old_index.index.difference(new_index.index)],
        "new_date_scores": {"overall": metric_stats(added),
                            **{split: metric_stats(added[added["split"].eq(split)])
                               for split in ("tuning", "holdout")}},
        "source_files_sha256": hashes, "comparison_script_sha256": sha256(Path(__file__).resolve()),
        "interpretation": "Historical data refresh comparison; newly added dates are not evidence of unseen-test or live performance.",
    }
    require(hashes == {"previous": source_hashes(previous), "current": source_hashes(current)},
            "Source files changed during comparison")
    require(all(sha256(Path(source["path"])) == source["sha256"] for source in manifests.values()),
            "Raw data manifest changed during comparison")
    added.to_csv(current / "new_dates_metrics.csv", index=False, date_format="%Y-%m-%d")
    report["output_files_sha256"] = {"new_dates_metrics.csv": sha256(current / "new_dates_metrics.csv")}
    (current / "data_update_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Compared {len(shared):,} common prediction rows; added {len(added)} scoring dates: {current}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, help="Optional project root containing the previous raw manifest")
    parser.add_argument("--current-root", type=Path, help="Optional snapshot root containing the current raw manifest")
    args = parser.parse_args()
    compare(args.previous.resolve(), args.current.resolve(),
            args.previous_root.resolve() if args.previous_root is not None else None,
            args.current_root.resolve() if args.current_root is not None else None)


if __name__ == "__main__":
    main()
