"""Delayed-label yearly walk-forward protocol for the paper-informed experiments.

The 2023--2026 period has already been observed. Its phase is therefore named
``reused_development`` throughout; this module creates no new untouched test.
The research pool restores mature records that the first static split purged,
then applies the maturity embargo separately for every prediction block.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KEYS = ["numerai_ticker", "date"]
TARGET = "target_jupiter_60"
CACHE_VERSION = 1
EMBARGO_WEEKDAYS = 65


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if frame[KEYS].isna().any().any():
        raise ValueError("Research pool contains null ticker/date keys")
    frame["numerai_ticker"] = frame["numerai_ticker"].astype("str").str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].isna().any() or frame["numerai_ticker"].eq("").any():
        raise ValueError("Research pool contains empty ticker or NaT date")
    return frame


def _eligible_batch(frame: pd.DataFrame, expected_type: str,
                    features: list[str], continuous: set[str]) -> pd.DataFrame:
    """Filter by source-specific type and exact label, then validate model inputs."""
    required = KEYS + features + [TARGET, "data_type"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Research source is missing columns: {sorted(missing)}")
    frame = frame.loc[frame["data_type"].eq(expected_type) & frame[TARGET].notna(),
                      KEYS + features + [TARGET]].copy()
    if frame.empty:
        return frame
    frame = _normalize_keys(frame)
    target = frame[TARGET]
    if not pd.api.types.is_numeric_dtype(target) or not np.isfinite(target.to_numpy()).all():
        raise ValueError("Invalid mature Jupiter60 target")
    if not target.between(0, 1).all():
        raise ValueError("Jupiter60 target outside 0 through 1")
    for feature in features:
        values = frame[feature]
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"Nonnumeric research feature: {feature}")
        nonnull = values.dropna().to_numpy()
        if not np.isfinite(nonnull).all():
            raise ValueError(f"Infinite research feature: {feature}")
        if feature not in continuous and not np.isin(nonnull, [0, 1, 2, 3, 4]).all():
            raise ValueError(f"Invalid binned research feature: {feature}")
    frame[TARGET] = frame[TARGET].astype("float32")
    return frame


def load_research_pool(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Return all eligible raw histories, feature sets and content provenance.

    This writes only research_pool.parquet/research_pool_manifest.json in the
    existing processed directory, and never changes the initial pilot splits.
    """
    root = Path(root).resolve()
    raw = root / "data/raw/v6.0"
    processed = root / "data/processed/v6.0"
    processed.mkdir(parents=True, exist_ok=True)
    metadata_path = raw / "features.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_sets = {name: [f for f in metadata["feature_sets"][name] if f != "feature_country"]
                    for name in ("small", "medium")}
    if not all(feature_sets.values()):
        raise ValueError("Required research feature set is empty")
    features = list(dict.fromkeys(feature_sets["small"] + feature_sets["medium"]))
    if any(not c.startswith("feature_") for c in features):
        raise ValueError("Research model features must come only from feature_ columns")
    continuous = set(metadata["feature_sets"].get("signals", [])) - {"feature_country"}
    source_paths = [metadata_path, raw / "train.parquet", raw / "validation.parquet"]
    fingerprint = {str(p.relative_to(root)).replace("\\", "/"): sha256_file(p) for p in source_paths}
    cache = processed / "research_pool.parquet"
    manifest_path = processed / "research_pool_manifest.json"
    provenance = None
    if cache.exists() and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (previous.get("cache_version") == CACHE_VERSION and
                previous.get("source_sha256") == fingerprint and
                previous.get("features") == features):
            if sha256_file(cache) != previous["cache_sha256"]:
                raise ValueError("Research pool cache hash mismatch")
            provenance = previous
    if provenance is None:
        temp = cache.with_suffix(".parquet.part")
        writer = None
        source_reports = {}
        try:
            for filename, expected_type in [("train.parquet", "train"),
                                             ("validation.parquet", "validation")]:
                pf = pq.ParquetFile(raw / filename)
                columns = KEYS + features + [TARGET, "data_type"]
                missing = set(columns) - set(pf.schema_arrow.names)
                if missing:
                    raise ValueError(f"{filename} missing required research columns: {sorted(missing)}")
                output_schema = pa.schema([
                    pa.field("numerai_ticker", pa.string()), pa.field("date", pa.timestamp("ns")),
                    *[pa.field(c, pf.schema_arrow.field(c).type) for c in features],
                    pa.field(TARGET, pa.float32())])
                report = {"raw_rows": pf.metadata.num_rows, "kept_rows": 0,
                          "excluded_other_data_type": 0, "excluded_null_target": 0,
                          "date_min": None, "date_max": None}
                for batch in pf.iter_batches(columns=columns, batch_size=100000):
                    frame = batch.to_pandas()
                    matching = frame["data_type"].eq(expected_type)
                    report["excluded_other_data_type"] += int((~matching).sum())
                    report["excluded_null_target"] += int((matching & frame[TARGET].isna()).sum())
                    eligible = _eligible_batch(frame, expected_type, features, continuous)
                    if eligible.empty:
                        continue
                    date_min = str(eligible.date.min().date())
                    date_max = str(eligible.date.max().date())
                    report["date_min"] = min(report["date_min"] or date_min, date_min)
                    report["date_max"] = max(report["date_max"] or date_max, date_max)
                    report["kept_rows"] += len(eligible)
                    # Keep nullable integer features consistent across batches:
                    # pandas can otherwise widen only a batch containing NaNs.
                    table = pa.Table.from_pandas(eligible, schema=output_schema,
                                                preserve_index=False, safe=True)
                    if writer is None:
                        writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
                    writer.write_table(table)
                source_reports[filename] = report
                print(f"Research pool {filename}: retained {report['kept_rows']:,}/{report['raw_rows']:,}", flush=True)
            if writer is None:
                raise ValueError("No eligible research rows")
        finally:
            if writer is not None:
                writer.close()
        # Check uniqueness across all batches and both sources before promotion.
        keys = _normalize_keys(pd.read_parquet(temp, columns=KEYS))
        if keys.duplicated(KEYS).any():
            raise ValueError("Research pool has duplicate stock-date keys across raw files or batches")
        del keys
        temp.replace(cache)
        provenance = {
            "cache_version": CACHE_VERSION, "source_sha256": fingerprint,
            "cache_path": str(cache.relative_to(root)).replace("\\", "/"),
            "cache_sha256": sha256_file(cache), "features": features,
            "target": TARGET, "source_reports": source_reports,
            "pool_policy": "All mature main-target train/validation records; no static pilot embargo filtering",
            "original_pilot_splits_modified": False,
            "embargo_policy": "Per prediction block only; date strictly before actual first scored date minus BDay65",
            "development_status": "2023 and later dates have already been observed in the initial pilot",
            "rows": sum(x["kept_rows"] for x in source_reports.values())}
        manifest_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    pool = _normalize_keys(pd.read_parquet(cache, columns=KEYS + features + [TARGET]))
    if (pool.empty or len(pool) != provenance["rows"] or
            pool.duplicated(KEYS).any() or pool[TARGET].isna().any()):
        raise ValueError("Invalid cached research pool keys or labels")
    pool.sort_values(["date", "numerai_ticker"], kind="stable", inplace=True)
    pool.reset_index(drop=True, inplace=True)
    provenance = dict(provenance)
    provenance["pool_date_min"] = str(pool.date.min().date())
    provenance["pool_date_max"] = str(pool.date.max().date())
    provenance["full_original_train_max_date"] = provenance["source_reports"]["train.parquet"]["date_max"]
    return pool, feature_sets, provenance


def make_fold_plan(pool: pd.DataFrame) -> list[dict]:
    """Fix yearly scoring blocks using actual available dates before modeling."""
    if pool.empty or pool["date"].isna().any():
        raise ValueError("Cannot plan folds from empty or missing dates")
    dates = pd.DatetimeIndex(pd.to_datetime(pool["date"].unique())).sort_values()
    if dates.hasnans:
        raise ValueError("Cannot plan folds from NaT dates")
    plan = []
    for phase, years, lower, upper, prefix in [
        ("cv", range(2019, 2023), pd.Timestamp("2019-04-05"), pd.Timestamp("2022-09-30"), "cv"),
        ("reused_development", range(2023, 2027), pd.Timestamp("2023-01-06"), dates.max(), "dev")]:
        for year in years:
            scoring = dates[(dates.year == year) & (dates >= lower) & (dates <= upper)]
            if len(scoring) == 0:
                continue
            start, end = scoring.min(), scoring.max()
            cutoff = start - pd.offsets.BDay(EMBARGO_WEEKDAYS)
            plan.append({"id": f"{prefix}_{year}", "phase": phase,
                         "eval_start": str(start.date()), "eval_end": str(end.date()),
                         "train_cutoff_exclusive": str(cutoff.date()),
                         "embargo_weekdays": EMBARGO_WEEKDAYS,
                         "n_eval_dates": len(scoring)})
    if not plan:
        raise ValueError("No scoring dates satisfy the declared research windows")
    return plan


def split_fold(pool: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen block plan with a strict horizon-aware training cutoff."""
    start, end = pd.Timestamp(fold["eval_start"]), pd.Timestamp(fold["eval_end"])
    embargo = int(fold["embargo_weekdays"])
    if embargo < EMBARGO_WEEKDAYS:
        raise ValueError("Research fold requires at least 65 weekdays of label delay")
    cutoff = pd.Timestamp(fold["train_cutoff_exclusive"])
    if cutoff != start - pd.offsets.BDay(embargo):
        raise ValueError("Training cutoff does not match declared first scoring date and embargo")
    train = pool.loc[pool.date < cutoff].copy()
    evaluate = pool.loc[pool.date.between(start, end)].copy()
    if train.empty or evaluate.empty:
        raise ValueError(f"Fold {fold['id']} has no training or scoring rows")
    if train.date.max() + pd.offsets.BDay(embargo) >= evaluate.date.min():
        raise ValueError("Training labels overlap the prediction period")
    if train[TARGET].isna().any() or evaluate[TARGET].isna().any():
        raise ValueError("Fold contains unmatured Jupiter60 labels")
    if train.duplicated(KEYS).any() or evaluate.duplicated(KEYS).any():
        raise ValueError("Fold contains duplicate ticker-date keys")
    # Date separation makes cross-partition key overlap impossible; assert the
    # controlling date relationship directly instead of an expensive full merge.
    if train.date.max() >= evaluate.date.min():
        raise ValueError("Training and scoring dates overlap")
    return train, evaluate
