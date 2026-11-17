"""Compare four frozen v6 candidates on a supplied, versioned data snapshot.

2019--2022 selects the candidate; 2023+ is reused historical development.
This command neither uploads predictions nor changes an existing baseline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from numerai_tools.scoring import neutral_correlation, numerai_corr
from threadpoolctl import threadpool_limits

from generate_xinchen_live import verify_baseline, check_live
from optimized_models import KEYS, TARGET, MODEL_IDS, candidate_specs, train_predict_candidates
from paper_protocol import load_research_pool, make_fold_plan, split_fold
from paper_statistics import summarize, paired_block_bootstrap
from run_experiments import PeakMemoryMonitor, read_neutralizers, sha256_file, write_json

HERE = Path(__file__).resolve().parent
REFERENCE = "B0_fixed_uploaded"


def select_candidate(scores: pd.DataFrame) -> dict:
    """Select using complete matched historical CV dates, never development."""
    if scores.empty or set(scores.phase) != {"cv"}:
        raise ValueError("Selection accepts CV scores only")
    if scores.duplicated(["fold", "date", "model"]).any():
        raise ValueError("Duplicated candidate scores")
    frame = scores.loc[scores.model.isin(MODEL_IDS)].copy()
    if set(frame.model) != set(MODEL_IDS) or set(frame.fold) != {f"cv_{y}" for y in range(2019, 2023)}:
        raise ValueError("All four candidates and four chronological CV folds are required")
    grid = frame.pivot(index=["fold", "date"], columns="model", values="NCORR")
    if grid.isna().any().any() or not np.isfinite(grid.to_numpy()).all():
        raise ValueError("Candidate scoring dates differ or contain invalid scores")
    if pd.Index(grid.index.get_level_values("date")).duplicated().any():
        raise ValueError("CV folds overlap")
    counts = frame.pivot(index=["fold", "date"], columns="model", values="n_rows")
    if not counts.eq(counts.iloc[:, 0], axis=0).all().all():
        raise ValueError("Candidate scoring populations differ")
    candidates = {}
    for name in MODEL_IDS:
        rows = frame.loc[frame.model.eq(name)]
        means = rows.groupby("fold").NCORR.mean()
        candidates[name] = {"mean_NCORR": float(rows.NCORR.mean()),
                            "fold_mean_std_ddof0": float(means.std(ddof=0)),
                            "selection_score": float(rows.NCORR.mean() - 0.5 * means.std(ddof=0)),
                            "fold_means": means.to_dict(), "n_dates": len(rows)}
    chosen = sorted(MODEL_IDS, key=lambda m: (-candidates[m]["selection_score"], m))[0]
    return {"selected_model": chosen, "candidates": candidates, "selection_phase": "cv",
            "criterion": "equal-date mean NCORR - 0.5 * population std of four annual mean NCORRs",
            "development_used_for_selection": False,
            "all_historical_periods_are_development_evidence": True,
            "frozen_at_utc": datetime.now(timezone.utc)}


def score_predictions(predictions: pd.DataFrame, neutralizer_path: Path, fold: dict):
    names = MODEL_IDS + [REFERENCE]
    neutralizers, coverage = read_neutralizers(neutralizer_path, predictions)
    if neutralizers is None:
        raise FileNotFoundError("Official v6 neutralizers are required")
    records = []
    for date, rows in predictions.groupby("date", sort=True):
        values = rows[names]
        if (values.nunique() < 2).any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Nonfinite/constant predictions on {date}")
        corr = numerai_corr(values, rows[TARGET], max_filtered_index_ratio=0)
        ncorr = neutral_correlation(values, neutralizers.loc[rows.index], rows[TARGET],
                                    max_filtered_index_ratio=0)
        for name in names:
            records.append({"phase": fold["phase"], "fold": fold["id"], "date": date,
                            "model": name, "n_rows": len(rows),
                            "CORR": float(corr[name]), "NCORR": float(ncorr[name])})
    result = pd.DataFrame(records)
    if not np.isfinite(result[["CORR", "NCORR"]].to_numpy()).all():
        raise ValueError("Undefined score")
    return result, coverage


def render_figures(scores: pd.DataFrame, selected: str, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    names = [REFERENCE] + MODEL_IDS
    means = scores.groupby(["model", "phase"]).NCORR.mean().unstack().loc[names]
    ax = means.rename(columns={"cv": "2019-2022 CV", "reused_development": "2023+ reused development"}).plot.bar(
        figsize=(12, 5), color=["#3769a8", "#239782"], width=.75)
    ax.set_ylabel("Mean neutral correlation per date")
    ax.set_title("Historical v6 comparison (not live performance)")
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.axhline(0, color="#444444", linewidth=.7)
    ax.figure.tight_layout()
    ax.figure.savefig(output / "ncorr_comparison.png", dpi=160)
    plt.close(ax.figure)
    dev = scores.loc[scores.phase.eq("reused_development")]
    series = dev.pivot(index="date", columns="model", values="NCORR").sort_index()
    plot_names = list(dict.fromkeys([REFERENCE, "O0_small_200k", "O1_medium_600k_ensemble", selected]))
    ax = series[plot_names].cumsum().plot(figsize=(11, 5))
    ax.set_ylabel("Sum of per-date NCORR (not investment returns)")
    ax.set_title("Reused historical development: cumulative NCORR")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "cumulative_ncorr.png", dpi=160)
    plt.close(ax.figure)


def run(args, monitor):
    out, root = args.output.resolve(), args.data_root.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Preserving existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    monitor.run_started = True
    started = time.perf_counter()
    run_info = {"status": "started", "started_utc": datetime.now(timezone.utc),
                "kind": "screenshot_followup_model_optimization", "data_version": "v6.0",
                "target": TARGET, "data_root": root, "n_jobs": args.n_jobs,
                "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in
                             ["lightgbm", "pandas", "numpy", "pyarrow", "numerai-tools", "scipy"]},
                "code_sha256": {name: sha256_file(HERE / name) for name in
                                [Path(__file__).name, "optimized_models.py", "paper_protocol.py",
                                 "paper_statistics.py", "run_experiments.py", "generate_xinchen_live.py"]},
                "interpretation": "Historical CV and reused development only; screenshot +3.4% is not a training target or evidence of improvement",
                "completed_folds": []}
    write_json(out / "run.json", run_info)
    pool, feature_sets, provenance = load_research_pool(root)
    plan = make_fold_plan(pool)
    booster, base_features, base_paths = verify_baseline(args.baseline.resolve(), args.n_jobs)
    live = pd.read_parquet(args.live_data, columns=KEYS + list(dict.fromkeys(feature_sets["medium"] + feature_sets["small"])))
    live["date"] = pd.to_datetime(live.date)
    check_live(live, feature_sets["medium"])
    if live.date.isna().any() or live.date.nunique() != 1:
        raise ValueError("Final fitting requires a single pinned live prediction date")
    live_date = live.date.iloc[0]
    final_cutoff = live_date - pd.offsets.BDay(65)
    write_json(out / "plan.json", {"frozen_at_utc": datetime.now(timezone.utc),
        "data_root": root, "source_provenance": provenance, "folds": plan,
        "candidates": candidate_specs(args.n_jobs), "selection_rule": "CV mean NCORR - 0.5*std(annual means)",
        "historical_development_previously_observed": True,
        "reference_model": {"id": REFERENCE, "directory": args.baseline.resolve(),
                            "model_sha256": sha256_file(base_paths[0]),
                            "scope": "Fixed previously submitted baseline; differs in refit schedule and features as well as sample size"},
        "final_fit": {"live_data": args.live_data.resolve(), "live_data_sha256": sha256_file(args.live_data),
                      "live_date": live_date, "train_cutoff_exclusive": final_cutoff,
                      "embargo_weekdays": 65, "candidate": "CV selection only"},
        "official_sources": ["https://docs.numer.ai/numerai-tournament/data",
                             "https://docs.numer.ai/numerai-tournament/scoring",
                             "https://docs.numer.ai/numerai-tournament/submissions"]})
    all_scores = []
    for phase in ["cv", "reused_development"]:
        for fold in [f for f in plan if f["phase"] == phase]:
            tick = time.perf_counter()
            train, evaluation = split_fold(pool, fold)
            fold_out = out / "folds" / fold["id"]
            print(f"{fold['id']}: train {len(train):,}, evaluate {len(evaluation):,}", flush=True)
            predictions, training = train_predict_candidates(train, evaluation, feature_sets,
                                                            fold_out / "models", n_jobs=args.n_jobs)
            predictions[REFERENCE] = booster.predict(evaluation[base_features], num_threads=args.n_jobs)
            predictions = predictions.reset_index(drop=True)
            del train, evaluation
            scores, coverage = score_predictions(predictions, root / "data/raw/v6.0/validation_neutralizer.parquet", fold)
            predictions.to_parquet(fold_out / "predictions.parquet", index=False)
            scores.to_csv(fold_out / "per_date_metrics.csv", index=False, date_format="%Y-%m-%d")
            write_json(fold_out / "training.json", {"fold": fold, "models": training,
                "neutralizer": coverage, "elapsed_seconds": time.perf_counter() - tick})
            all_scores.append(scores)
            pd.concat(all_scores, ignore_index=True).to_csv(out / "per_date_metrics.csv", index=False, date_format="%Y-%m-%d")
            run_info["completed_folds"].append(fold["id"])
            write_json(out / "run.json", run_info)
            print(f"{fold['id']} done in {time.perf_counter() - tick:.1f}s", flush=True)
            del predictions, scores
            gc.collect()
        if phase == "cv":
            selection = select_candidate(pd.concat(all_scores, ignore_index=True))
            write_json(out / "selection.json", selection)
            selected = selection["selected_model"]
            print(f"Frozen CV selection: {selected}", flush=True)
    combined = pd.concat(all_scores, ignore_index=True)
    metrics = summarize(combined)
    metrics.to_csv(out / "metrics.csv", index=False)
    annual = combined.assign(year=combined.date.dt.year).groupby(["phase", "year", "model"])[["CORR", "NCORR"]].agg(["mean", "std", "count"])
    annual.to_csv(out / "annual_metrics.csv")
    comparisons = {name: paired_block_bootstrap(combined.loc[combined.phase.eq("reused_development")],
                    selected, baseline=name) for name in [REFERENCE, "O0_small_200k", "O1_medium_600k_ensemble"]}
    write_json(out / "paired_comparison.json", comparisons)
    combined.loc[combined.date.gt(pd.Timestamp("2026-07-24"))].to_csv(out / "newly_matured_dates.csv", index=False)
    render_figures(combined, selected, out / "figures")
    # Refit the frozen winner only, using labels strictly mature before live.
    final_train = pool.loc[pool.date.lt(final_cutoff)].copy()
    if final_train.empty or final_train.date.max() + pd.offsets.BDay(65) >= live_date:
        raise ValueError("Final training violates the strict label maturity embargo")
    live_preds, final_records = train_predict_candidates(final_train, live, feature_sets,
        out / "final_model", n_jobs=args.n_jobs, candidate_ids=[selected])
    write_json(out / "final_training.json", {"selected_model": selected,
        "live_date": live_date, "train_cutoff_exclusive": final_cutoff,
        "full_training_rows": len(final_train), "full_training_date_max": final_train.date.max(),
        "models": final_records, "selection_sha256": sha256_file(out / "selection.json")})
    live_preds.to_parquet(out / "final_live_preview.parquet", index=False)
    # Keyed validation file supports official diagnostics; scores remain historical.
    diagnostics = []
    for fold in plan:
        part = pd.read_parquet(out / "folds" / fold["id"] / "predictions.parquet", columns=KEYS + [selected])
        part["prediction"] = part.groupby("date")[selected].rank(method="average", pct=True)
        diagnostics.append(part[KEYS + ["prediction"]])
    pd.concat(diagnostics, ignore_index=True).to_csv(out / "validation_predictions.csv", index=False, date_format="%Y-%m-%d")
    run_info.update(status="complete", selected_model=selected,
        completed_utc=datetime.now(timezone.utc), elapsed_seconds=time.perf_counter() - started,
        source_provenance=provenance, final_bundle=out / "final_model" / selected / "bundle.json",
        output_sha256={name: sha256_file(out / name) for name in
                       ["plan.json", "selection.json", "metrics.csv", "per_date_metrics.csv",
                        "paired_comparison.json", "final_training.json", "validation_predictions.csv"]})
    write_json(out / "run.json", run_info)
    print(metrics[["phase", "model", "metric", "mean", "sharpe"]].to_string(index=False), flush=True)
    print(f"Complete: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--live-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    monitor = PeakMemoryMonitor()
    failure = None
    try:
        with monitor, threadpool_limits(limits=args.n_jobs):
            run(args, monitor)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if monitor.run_started:
            path = args.output.resolve() / "run.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["memory"] = monitor.result()
            if failure:
                manifest.update(status="failed", error=failure)
            write_json(path, manifest)


if __name__ == "__main__":
    main()
