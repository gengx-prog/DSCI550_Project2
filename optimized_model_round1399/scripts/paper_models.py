"""Fixed paper-informed tree comparisons with no evaluation-label fitting.

This module implements library applications and controlled adaptations. In
particular, the CatBoost control uses Plain boosting; it does not claim to
reproduce ordered boosting or the paper's categorical target statistics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

KEYS = ["numerai_ticker", "date"]
TARGET = "target_jupiter_60"
MODEL_IDS = ["P0_small_200k", "P1_medium_200k", "P2_medium_600k", "P3_recency_600k",
             "P4_xgboost", "P5_catboost", "P6_seed_ensemble", "P7_heterogeneous"]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _sample_hash(sample: pd.DataFrame) -> str:
    return hashlib.sha256(sample[KEYS].to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")).hexdigest()


def _rank_mean(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    return frame.groupby("date", sort=False)[names].rank(method="average", pct=True).mean(axis=1)


def recency_date_weights(sample: pd.DataFrame, reference_date: pd.Timestamp,
                        half_life_years: float = 4.0) -> np.ndarray:
    """Equal total mass per sampled date, then a past-only exponential decay."""
    if sample.empty or half_life_years <= 0:
        raise ValueError("Positive half life and a nonempty sample are required")
    dates = pd.to_datetime(sample["date"])
    age_days = (pd.Timestamp(reference_date) - dates).dt.days.to_numpy(dtype=float)
    if np.any(age_days < 0) or not np.isfinite(age_days).all():
        raise ValueError("Recency reference must be at or after every training date")
    date_counts = dates.map(dates.value_counts()).to_numpy(dtype=float)
    weights = np.exp2(-age_days / (365.25 * half_life_years)) / date_counts
    weights /= weights.mean()
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Invalid training weights")
    return weights


def _validate_inputs(train_df: pd.DataFrame, eval_df: pd.DataFrame,
                     feature_sets: dict) -> tuple[list[str], list[str]]:
    if train_df.empty or eval_df.empty:
        raise ValueError("Nonempty training and evaluation frames are required")
    small, medium = list(feature_sets["small"]), list(feature_sets["medium"])
    for name, features in [("small", small), ("medium", medium)]:
        if not features or len(features) != len(set(features)):
            raise ValueError(f"Empty or repeated {name} features")
        if any(not f.startswith("feature_") or f == "feature_country" for f in features):
            raise ValueError("Only explicitly selected numeric feature_ columns are permitted")
        for label, frame in [("train", train_df), ("evaluation", eval_df)]:
            missing = set(KEYS + [TARGET] + features) - set(frame.columns)
            if missing:
                raise ValueError(f"{label}: missing columns {sorted(missing)}")
            if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in features):
                raise ValueError(f"{label}: nonnumeric feature in {name}")
    if train_df[KEYS + [TARGET]].isna().any().any() or eval_df[KEYS].isna().any().any():
        raise ValueError("Missing training key/target or evaluation key")
    if train_df.duplicated(KEYS).any() or eval_df.duplicated(KEYS).any():
        raise ValueError("Duplicate stock/date keys")
    if not train_df[TARGET].between(0, 1).all():
        raise ValueError("Training Jupiter60 labels must be finite and in 0..1")
    return small, medium


def train_predict_suite(train_df: pd.DataFrame, eval_df: pd.DataFrame,
                        feature_sets: dict, output_model_dir: Path,
                        max_train_rows: int = 600000, seed: int = 7,
                        n_jobs: int = 12) -> tuple[pd.DataFrame, list[dict]]:
    """Return the eight frozen controls; no evaluation targets enter training.

    P0/P1 use the first 200k of the same seed-selected P2/P3/P4/P5 sample.
    P6 varies only model seed on that same large sample. P7 reuses family
    predictions. The caller owns chronological split and embargo verification.
    """
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor

    if max_train_rows < 1 or n_jobs < 1:
        raise ValueError("max_train_rows and n_jobs must be positive")
    small, medium = _validate_inputs(train_df, eval_df, feature_sets)
    out = Path(output_model_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference_date = pd.to_datetime(train_df["date"]).max()
    features_union = list(dict.fromkeys(small + medium))
    sample = train_df[KEYS + [TARGET] + features_union].sample(
        n=min(max_train_rows, len(train_df)), random_state=seed).copy().reset_index(drop=True)
    sample["date"] = pd.to_datetime(sample["date"])
    nested = sample.iloc[:min(200000, len(sample))]
    if np.isinf(sample[features_union].to_numpy()).any():
        raise ValueError("Infinite training features are invalid; NaNs remain native")
    if np.isinf(eval_df[features_union].to_numpy()).any():
        raise ValueError("Infinite evaluation features are invalid; NaNs remain native")
    large_hash, nested_hash = _sample_hash(sample), _sample_hash(nested)
    predictions = eval_df[KEYS + [TARGET]].copy().reset_index(drop=True)
    predictions["date"] = pd.to_datetime(predictions["date"])
    records = []
    lgb_base = dict(n_estimators=300, learning_rate=0.05, max_depth=5, num_leaves=31,
                    colsample_bytree=0.1, verbosity=-1, n_jobs=n_jobs,
                    deterministic=True, force_col_wise=True)

    def fit_lgb(name, data, features, model_seed, weights=None):
        params = dict(lgb_base, random_state=model_seed, feature_fraction_seed=model_seed,
                      bagging_seed=model_seed, data_random_seed=model_seed)
        return fit_one(name, lgb.LGBMRegressor(**params), "lightgbm", data, features,
                       model_seed, params, weights)

    def fit_one(name, model, family, data, features, model_seed, params, weights=None):
        tick = time.perf_counter()
        kwargs = {} if weights is None else {"sample_weight": weights}
        model.fit(data[features], data[TARGET], **kwargs)
        fit_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        values = model.predict(eval_df[features])
        predict_seconds = time.perf_counter() - tick
        if len(values) != len(eval_df) or not np.isfinite(values).all():
            raise ValueError(f"Invalid predictions: {name}")
        predictions[name] = values
        if family == "lightgbm":
            model_path = out / f"{name}.txt"
            model.booster_.save_model(str(model_path))
            importance = pd.DataFrame({"feature": features,
                "gain": model.booster_.feature_importance(importance_type="gain"),
                "split_count": model.booster_.feature_importance(importance_type="split")})
            actual_params = model.get_params()
        elif family == "xgboost":
            model_path = out / f"{name}.json"
            model.save_model(model_path)
            importance = pd.DataFrame({"feature": features, "normalized_gain": model.feature_importances_})
            actual_params = model.get_params()
        elif family == "catboost":
            model_path = out / f"{name}.cbm"
            model.save_model(str(model_path))
            importance = pd.DataFrame({"feature": features,
                "prediction_values_change": model.get_feature_importance()})
            actual_params = model.get_all_params()
        else:
            raise ValueError(f"Unknown family: {family}")
        importance_path = out / f"{name}_feature_importance.csv"
        importance.to_csv(importance_path, index=False)
        record = {"model": name, "family": family, "target": TARGET,
            "training_rows": len(data), "n_features": len(features), "seed": model_seed,
            "sample_seed": seed, "sample_keys_sha256": _sample_hash(data),
            "training_date_min": data["date"].min(), "training_date_max": data["date"].max(),
            "training_seconds": fit_seconds, "prediction_seconds": predict_seconds,
            "params": params, "actual_params": actual_params, "feature_names": features,
            "model_file": str(model_path), "importance_file": str(importance_path),
            "weights": "uniform" if weights is None else "equal_date_mass_times_recency",
            "evaluation_labels_used_in_fit": False}
        if weights is not None:
            record["weight_config"] = {"half_life_years": 4.0,
                "reference_date": reference_date, "reference_is_full_training_max": True,
                "normalized_mean": float(weights.mean()), "min": float(weights.min()),
                "max": float(weights.max()),
                "formula": "normalize_mean1(2**(-age_days/(365.25*4))/sampled_rows_on_date)"}
        record = _json_safe(record)
        print(f"Trained {name}: {len(data):,} rows, {fit_seconds:.2f}s", flush=True)
        return record

    records.append(fit_lgb("P0_small_200k", nested, small, seed))
    records.append(fit_lgb("P1_medium_200k", nested, medium, seed))
    records.append(fit_lgb("P2_medium_600k", sample, medium, seed))
    weights = recency_date_weights(sample, reference_date)
    records.append(fit_lgb("P3_recency_600k", sample, medium, seed, weights))
    xgb_params = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
        tree_method="hist", device="cpu", colsample_bytree=0.5, subsample=0.8,
        min_child_weight=100, reg_lambda=5.0, objective="reg:squarederror",
        random_state=seed, n_jobs=n_jobs, verbosity=0)
    records.append(fit_one("P4_xgboost", XGBRegressor(**xgb_params), "xgboost",
                           sample, medium, seed, xgb_params))
    cat_params = dict(iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=6,
        loss_function="RMSE", boosting_type="Plain", task_type="CPU",
        thread_count=n_jobs, random_seed=seed, allow_writing_files=False, verbose=False)
    records.append(fit_one("P5_catboost", CatBoostRegressor(**cat_params), "catboost",
                           sample, medium, seed, cat_params))
    seed_records = [fit_lgb(f"internal_medium_seed_{s}", sample, medium, s) for s in [19, 43]]
    seed_names = ["P2_medium_600k", "internal_medium_seed_19", "internal_medium_seed_43"]
    tick = time.perf_counter()
    predictions["P6_seed_ensemble"] = _rank_mean(predictions, seed_names)
    seed_rank_seconds = time.perf_counter() - tick
    records.append(_json_safe({"model": "P6_seed_ensemble", "family": "rank_ensemble",
        "target": TARGET, "training_rows": len(sample), "n_features": len(medium),
        "sample_seed": seed, "seed": seed, "sample_keys_sha256": large_hash,
        "training_seconds": sum(r["training_seconds"] for r in seed_records),
        "prediction_seconds": seed_rank_seconds,
        "prediction_cost_definition": "Incremental rank aggregation only; member predictions are reused",
        "training_cost_definition": "Incremental fitting cost of seed19/43; seed7 P2 is reused",
        "members": seed_names, "member_seeds": [seed, 19, 43],
        "member_training_records": [records[2]] + seed_records,
        "params": {"weights": [1/3, 1/3, 1/3], "rank": "average percentiles within evaluation date"},
        "evaluation_labels_used_in_fit": False}))
    family_names = ["P2_medium_600k", "P4_xgboost", "P5_catboost"]
    tick = time.perf_counter()
    predictions["P7_heterogeneous"] = _rank_mean(predictions, family_names)
    family_rank_seconds = time.perf_counter() - tick
    records.append(_json_safe({"model": "P7_heterogeneous", "family": "rank_ensemble",
        "target": TARGET, "training_rows": len(sample), "n_features": len(medium),
        "sample_seed": seed, "seed": seed, "sample_keys_sha256": large_hash,
        "training_seconds": 0.0, "training_cost_definition": "No incremental fit; reuses P2/P4/P5",
        "prediction_seconds": family_rank_seconds,
        "prediction_cost_definition": "Incremental rank aggregation only; member predictions are reused",
        "members": family_names,
        "params": {"weights": [1/3, 1/3, 1/3], "rank": "average percentiles within evaluation date"},
        "evaluation_labels_used_in_fit": False}))
    output = predictions[KEYS + [TARGET] + MODEL_IDS]
    if output[MODEL_IDS].isna().any().any():
        raise ValueError("Missing scored model predictions")
    manifest = {"model_ids": MODEL_IDS, "sample_seed": seed,
        "large_sample_rows": len(sample), "large_sample_keys_sha256": large_hash,
        "nested_sample_rows": len(nested), "nested_sample_keys_sha256": nested_hash,
        "nested_sampling": "First rows of the identical randomly permuted large sample",
        "training_records": records}
    (out / "training_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2,
        allow_nan=False) + "\n", encoding="utf-8")
    return output, records
