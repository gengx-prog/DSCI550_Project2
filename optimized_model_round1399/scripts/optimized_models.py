"""Frozen LightGBM candidates and portable, hash-checked rank ensembles.

All sampling uses one seed-7 permutation of at most 1.2 million past rows.
Evaluation labels are never read by fitting or prediction. The experiment
runner owns the stronger 65-business-day maturity embargo.
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
MODEL_IDS = ["O0_small_200k", "O1_medium_600k_ensemble",
             "O2_medium_1200k_ensemble", "O3_regularized_1200k_ensemble"]
SAMPLE_SEED = 7
PERMUTATION_ROWS = 1_200_000
RANK_METHOD = "average_percentile_within_date_then_equal_mean"


def candidate_specs(n_jobs: int = 12) -> dict:
    """Return fresh fixed specifications; model seeds do not change sampling."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    base = dict(n_estimators=300, learning_rate=0.05, max_depth=5,
                num_leaves=31, colsample_bytree=0.1, verbosity=-1,
                deterministic=True, force_col_wise=True, n_jobs=n_jobs)
    regularized = dict(base, n_estimators=800, learning_rate=0.03,
                       colsample_bytree=0.3, min_child_samples=1000,
                       reg_lambda=10.0, subsample=0.8, subsample_freq=1)
    return {
        MODEL_IDS[0]: {"feature_set": "small", "featureset": "small",
                       "max_rows": 200_000, "seeds": [7], "params": dict(base)},
        MODEL_IDS[1]: {"feature_set": "medium", "featureset": "medium",
                       "max_rows": 600_000, "seeds": [7, 19, 43], "params": dict(base)},
        MODEL_IDS[2]: {"feature_set": "medium", "featureset": "medium",
                       "max_rows": 1_200_000, "seeds": [7, 19, 43], "params": dict(base)},
        MODEL_IDS[3]: {"feature_set": "medium", "featureset": "medium",
                       "max_rows": 1_200_000, "seeds": [7, 19, 43], "params": regularized},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_hash(frame: pd.DataFrame) -> str:
    keys = frame[KEYS].copy()
    keys["date"] = pd.to_datetime(keys["date"])
    return hashlib.sha256(keys.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")).hexdigest()


def _feature_names(features: list[str]) -> list[str]:
    values = list(features)
    if (not values or len(values) != len(set(values)) or
            any(not isinstance(c, str) or not c.startswith("feature_") or
                c == "feature_country" for c in values)):
        raise ValueError("Expected distinct numeric feature_ names excluding feature_country")
    return values


def _validate_frame(frame: pd.DataFrame, features: list[str], label: str,
                    training: bool = False) -> pd.Series:
    required = KEYS + features + ([TARGET] if training else [])
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if frame.empty or frame[KEYS].isna().any().any():
        raise ValueError(f"{label}: nonempty rows with complete keys are required")
    if not frame["numerai_ticker"].map(
            lambda x: isinstance(x, str) and bool(x) and x == x.strip()).all():
        raise ValueError(f"{label}: invalid ticker")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError(f"{label}: dates must be complete and normalized")
    key_frame = frame[["numerai_ticker"]].copy()
    key_frame["date"] = dates.to_numpy()
    if key_frame.duplicated(KEYS).any():
        raise ValueError(f"{label}: duplicate ticker/date keys")
    for name in features:
        column = frame[name]
        if (not pd.api.types.is_numeric_dtype(column) or
                pd.api.types.is_bool_dtype(column)):
            raise ValueError(f"{label}: nonnumeric feature {name}")
        values = column.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"{label}: infinite feature {name}")
    if training:
        labels = frame[TARGET]
        if (not pd.api.types.is_numeric_dtype(labels) or
                pd.api.types.is_bool_dtype(labels) or labels.isna().any() or
                not labels.between(0, 1).all()):
            raise ValueError("Training Jupiter60 labels must be finite and in [0, 1]")
    return dates


def rank_mean(dates, member_predictions) -> np.ndarray:
    """Average tied percentiles per date, preserving input row position."""
    values = np.asarray(member_predictions, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    dates = pd.to_datetime(pd.Series(np.asarray(dates)), errors="raise")
    if (values.ndim != 2 or len(values) != len(dates) or
            values.shape[1] < 1 or len(values) < 1 or
            dates.isna().any() or not np.isfinite(values).all()):
        raise ValueError("Finite member predictions and aligned complete dates are required")
    frame = pd.DataFrame(values)
    return frame.groupby(dates, sort=False).rank(method="average", pct=True).mean(axis=1).to_numpy()


def _chosen_candidates(candidate_ids) -> list[str]:
    chosen = list(MODEL_IDS if candidate_ids is None else candidate_ids)
    if not chosen or len(chosen) != len(set(chosen)) or set(chosen).difference(MODEL_IDS):
        raise ValueError("Choose distinct fixed candidate IDs")
    return chosen


def train_predict_candidates(train: pd.DataFrame, evaluation: pd.DataFrame,
                             feature_sets: dict, out_dir: Path,
                             n_jobs: int = 12, candidate_ids=None) -> tuple[pd.DataFrame, list[dict]]:
    """Fit fixed candidates on nested samples; return per-date ranked predictions.

    Every candidate has a self-contained directory with native member models
    and a bundle manifest. Existing candidate directories are never reused.
    """
    specs = candidate_specs(n_jobs)
    chosen = _chosen_candidates(candidate_ids)
    used_sets = list(dict.fromkeys(specs[name]["feature_set"] for name in chosen))
    selected_features = {name: _feature_names(feature_sets[name]) for name in used_sets}
    union = list(dict.fromkeys(c for name in used_sets for c in selected_features[name]))
    train_dates = _validate_frame(train, union, "training", training=True)
    eval_dates = _validate_frame(evaluation, union, "evaluation")
    if train_dates.max() >= eval_dates.min():
        raise ValueError("Training dates must be strictly before every prediction date")
    root = Path(out_dir).resolve()
    for name in chosen:
        if (root / name).exists():
            raise FileExistsError(f"Preserving existing candidate: {root / name}")
    root.mkdir(parents=True, exist_ok=True)
    sample = train[KEYS + [TARGET] + union].sample(
        n=min(PERMUTATION_ROWS, len(train)), random_state=SAMPLE_SEED).reset_index(drop=True)
    sample["date"] = pd.to_datetime(sample["date"])
    return_columns = KEYS + ([TARGET] if TARGET in evaluation.columns else [])
    predictions = evaluation[return_columns].copy().reset_index(drop=True)
    predictions["date"] = eval_dates.to_numpy()
    records = []
    for name in chosen:
        spec = specs[name]
        features = selected_features[spec["feature_set"]]
        data = sample.iloc[:min(spec["max_rows"], len(sample))]
        candidate_dir = root / name
        candidate_dir.mkdir()
        members, values = [], []
        for seed in spec["seeds"]:
            params = dict(spec["params"], random_state=seed, feature_fraction_seed=seed,
                          bagging_seed=seed, data_random_seed=seed)
            tick = time.perf_counter()
            model = lgb.LGBMRegressor(**params)
            model.fit(data[features], data[TARGET])
            training_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            value = model.predict(evaluation[features], num_threads=n_jobs)
            prediction_seconds = time.perf_counter() - tick
            if len(value) != len(evaluation) or not np.isfinite(value).all():
                raise ValueError(f"Nonfinite or incomplete predictions: {name}, seed {seed}")
            model_path = candidate_dir / f"member_seed_{seed}.txt"
            model.booster_.save_model(str(model_path))
            if model.booster_.feature_name() != features:
                raise ValueError("Saved-model feature order differs from requested features")
            members.append({"file": model_path.name, "sha256": _sha256(model_path),
                            "seed": seed, "params": params,
                            "num_trees": model.booster_.num_trees(),
                            "training_seconds": training_seconds,
                            "prediction_seconds": prediction_seconds})
            values.append(value)
            print(f"Trained {name}, seed {seed}: {len(data):,} rows, "
                  f"{len(features)} features, {training_seconds:.2f}s", flush=True)
            del model
        tick = time.perf_counter()
        predictions[name] = rank_mean(predictions["date"], np.column_stack(values))
        aggregation_seconds = time.perf_counter() - tick
        bundle = {
            "schema_version": 1, "model_id": name, "family": "lightgbm_rank_ensemble",
            "target": TARGET, "feature_set": spec["feature_set"], "features": features,
            "training_rows": len(data), "training_date_min": str(data.date.min().date()),
            "training_date_max": str(data.date.max().date()),
            "full_training_date_min": str(train_dates.min().date()),
            "full_training_date_max": str(train_dates.max().date()),
            "sample_seed": SAMPLE_SEED, "sample_keys_sha256": _sample_hash(data),
            "sampling_permutation_rows": len(sample),
            "sampling_policy": "Prefix of one seed-7 pandas permutation of min(1200000, eligible rows)",
            "member_seeds": spec["seeds"], "members": members,
            "weights": [1 / len(members)] * len(members), "rank_method": RANK_METHOD,
            "training_seconds": sum(m["training_seconds"] for m in members),
            "prediction_seconds": sum(m["prediction_seconds"] for m in members) + aggregation_seconds,
            "evaluation_labels_used_in_fit": False,
        }
        bundle_path = candidate_dir / "bundle.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        records.append({"model": name, "bundle_file": str(bundle_path), **bundle})
    return predictions, records


def predict_bundle(bundle_path: Path, frame: pd.DataFrame, n_jobs: int = 12) -> np.ndarray:
    """Recompute a bundle's rank ensemble after checking every native model."""
    candidate_specs(n_jobs)  # Validate the public thread-count argument.
    path = Path(bundle_path).resolve()
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if (bundle.get("schema_version") != 1 or
            bundle.get("family") != "lightgbm_rank_ensemble" or
            bundle.get("model_id") not in MODEL_IDS or bundle.get("target") != TARGET or
            bundle.get("rank_method") != RANK_METHOD):
        raise ValueError("Unsupported or invalid optimized-model bundle")
    features = _feature_names(bundle["features"])
    dates = _validate_frame(frame, features, "prediction")
    if dates.min() <= pd.Timestamp(bundle["full_training_date_max"]):
        raise ValueError("Bundle training dates must precede prediction dates")
    members = bundle["members"]
    weights = np.asarray(bundle["weights"], dtype=float)
    if (not isinstance(members, list) or not members or weights.shape != (len(members),) or
            not np.isfinite(weights).all() or
            not np.allclose(weights, np.repeat(1 / len(members), len(members)), rtol=0, atol=1e-15)):
        raise ValueError("Bundle requires equal weights for every member")
    if [m["seed"] for m in members] != bundle["member_seeds"]:
        raise ValueError("Member seed order differs from the bundle")
    if len({m["file"] for m in members}) != len(members):
        raise ValueError("Duplicate bundle member files")
    values = []
    for member in members:
        model_path = (path.parent / member["file"]).resolve()
        if model_path.parent != path.parent or model_path.suffix != ".txt":
            raise ValueError("Member file must be a native model inside its bundle directory")
        if _sha256(model_path) != member["sha256"]:
            raise ValueError(f"Saved model hash mismatch: {model_path.name}")
        booster = lgb.Booster(model_file=str(model_path))
        if booster.feature_name() != features or booster.num_trees() != member["num_trees"]:
            raise ValueError("Saved model feature order or tree count differs from its bundle")
        predicted = booster.predict(frame[features], num_threads=n_jobs)
        if len(predicted) != len(frame) or not np.isfinite(predicted).all():
            raise ValueError("Nonfinite or incomplete saved-model prediction")
        values.append(predicted)
    return rank_mean(dates, np.column_stack(values))
