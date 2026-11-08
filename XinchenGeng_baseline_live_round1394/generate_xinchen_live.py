"""Generate a round-pinned Numerai v6 live CSV from the saved Xinchen baseline.

Uses public downloads only. Never retrains, authenticates, stakes, or submits.
The CSV follows https://docs.numer.ai/numerai-tournament/submissions .
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import shutil
import sys
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import lightgbm as lgb
from numerapi import NumerAPI
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE
VERSION = "v6.0"
MODEL = "B0_small_main"
LOCAL_ZONE = "America/Los_Angeles"
TRANSFORM = "pandas.Series.rank(method='average', pct=True)"
DATASET_QUERY = """
query ($filename: String!, $round: Int, $tournament: Int) {
  dataset(filename: $filename, round: $round, tournament: $tournament)
}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Expected an HTTPS public dataset URL")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def current_round(api: NumerAPI) -> int:
    number = api.get_current_round()
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        raise ValueError("No valid current Numerai round")
    return number


def download(api: NumerAPI, name: str, round_number: int, destination: Path) -> dict:
    dataset = f"{VERSION}/{name}"
    url = api.raw_query(DATASET_QUERY, {"filename": dataset, "round": round_number,
                                      "tournament": api.tournament_id})["data"]["dataset"]
    public_url = safe_url(url)
    temporary = destination.with_suffix(destination.suffix + ".part")
    retry = Retry(total=4, connect=4, read=4, status=4, backoff_factor=1,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods={"GET"})
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        with session.get(url, stream=True, timeout=(30, 180),
                         headers={"Accept-Encoding": "identity"}) as response:
            response.raise_for_status()
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Unexpected dataset content encoding")
            expected = response.headers.get("Content-Length")
            with temporary.open("xb") as stream:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        stream.write(chunk)
            if expected is not None and temporary.stat().st_size != int(expected):
                raise ValueError("Incomplete dataset download")
            metadata = {key: response.headers[key] for key in
                        ("Last-Modified", "ETag", "Content-Length") if key in response.headers}
    if name.endswith(".parquet"):
        with pq.ParquetFile(temporary) as parquet:
            rows = parquet.metadata.num_rows
        if rows < 2:
            raise ValueError("Live dataset must contain at least two observations")
    else:
        read_json(temporary)
        rows = None
    temporary.replace(destination)
    return {"dataset": dataset, "filename": name, "round": round_number,
            "url": public_url, "bytes": destination.stat().st_size,
            "sha256": sha256(destination), "rows": rows,
            "downloaded_at_utc": utc_now(), "remote_metadata": metadata}


def baseline_paths(directory: Path) -> tuple[Path, Path, Path]:
    if (directory / "baseline_run.json").is_file():
        return (directory / "B0_small_main.txt", directory / "baseline_run.json",
                directory / "baseline_protocol.json")
    return (directory / "models/B0_small_main.txt", directory / "run.json", directory / "protocol.json")


def verify_baseline(directory: Path, n_jobs: int) -> tuple[lgb.Booster, list[str], tuple[Path, Path, Path]]:
    model_path, run_path, protocol_path = baseline_paths(directory)
    run, protocol = read_json(run_path), read_json(protocol_path)
    if run.get("status") != "complete" or run.get("model") != MODEL:
        raise ValueError("Expected the completed fixed Xinchen baseline")
    if run.get("data_version") != VERSION or protocol.get("data_version") != VERSION:
        raise ValueError("Baseline must use Numerai v6.0")
    if run.get("target") != "target_jupiter_60" or protocol.get("target") != run["target"]:
        raise ValueError("Unexpected baseline target")
    if sha256(model_path) != run["model_sha256"] or sha256(protocol_path) != run["protocol_sha256"]:
        raise ValueError("Baseline model or protocol differs from the recorded run")
    booster = lgb.Booster(model_file=str(model_path))
    features = booster.feature_name()
    if features != run["features"] or features != protocol["features"]:
        raise ValueError("Saved-model feature order differs from the recorded baseline")
    if len(features) != 100 or "feature_country" in features or booster.num_trees() != 300:
        raise ValueError("Expected the fixed 100-numeric-feature, 300-tree baseline")
    return booster, features, (model_path, run_path, protocol_path)


def check_live(frame: pd.DataFrame, features: list[str]) -> None:
    tickers = frame["numerai_ticker"]
    if len(frame) < 2 or tickers.isna().any() or tickers.duplicated().any():
        raise ValueError("Live tickers must be nonempty, non-null and unique")
    if not tickers.map(lambda value: isinstance(value, str) and bool(value) and value == value.strip()).all():
        raise ValueError("Invalid live ticker strings")
    for name in features:
        if (not pd.api.types.is_numeric_dtype(frame[name])
                or pd.api.types.is_bool_dtype(frame[name])):
            raise ValueError(f"Nonnumeric live feature: {name}")
    # Match the trained model's native handling of missing numeric features.
    if np.isinf(frame[features].to_numpy(dtype=np.float64, na_value=np.nan)).any():
        raise ValueError("Infinite live feature values")


def generate(args: argparse.Namespace) -> None:
    api = NumerAPI(show_progress_bars=False)
    before = current_round(api)
    round_number = args.round_number if args.round_number is not None else before
    if round_number != before:
        raise ValueError("Requested round is no longer the current round; generate a fresh live CSV")
    booster, features, paths = verify_baseline(args.baseline.resolve(), args.n_jobs)
    model_path, run_path, protocol_path = paths
    round_entries = [entry for entry in api.list_rounds(number=round_number)
                     if entry["number"] == round_number and entry["tournament"] == api.tournament_id
                     and entry["target"] == "target_jupiter_60"]
    if len(round_entries) != 1:
        raise ValueError("Round metadata does not uniquely match the baseline tournament/target")
    round_info = {key: round_entries[0].get(key) for key in
                  ("number", "tournament", "target", "dataDatestamp", "openTime", "closeTime", "closeStakingTime")}
    round_info = {key: value.isoformat() if isinstance(value, datetime) else value
                  for key, value in round_info.items()}
    if datetime.fromisoformat(round_info["closeTime"].replace("Z", "+00:00")) <= datetime.now(timezone.utc):
        raise ValueError("Current round's general submission window has closed")
    required = {f"{VERSION}/live.parquet", f"{VERSION}/features.json"}
    if not required.issubset(api.list_datasets(round_num=round_number)):
        raise ValueError("Current round does not expose the required v6.0 live datasets")
    output = args.output.resolve()
    data_dir = (args.data_dir or (ROOT / f"data/live/v6.0/round{round_number}")).resolve()
    if output == data_dir or output.is_relative_to(data_dir) or data_dir.is_relative_to(output):
        raise ValueError("Keep the live dataset directory separate from the handoff output")
    for directory in (output, data_dir):
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise FileExistsError(f"Preserving existing files: {directory}; choose a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    record = {"status": "started", "round": round_number, "data_version": VERSION,
              "model_name": MODEL, "started_at_utc": utc_now(), "timezone": LOCAL_ZONE,
              "run_date_local": datetime.now(ZoneInfo(LOCAL_ZONE)).date().isoformat(),
              "current_round_before": before, "round_info": round_info,
              "baseline_source": str(args.baseline.resolve()), "live_data_directory": str(data_dir),
              "account_submission_performed": False, "training_performed": False}
    write_json(output / "live_generation.json", record)
    try:
        manifest = {"status": "started", "data_version": VERSION, "round": round_number,
                    "tournament": api.tournament_id, "round_info": round_info,
                    "download_policy": "Complete public objects; explicit same-round queries; no filtered downloads.",
                    "files": []}
        print(f"Pinned round {round_number}; downloading v6.0 live features", flush=True)
        for name in ("features.json", "live.parquet"):
            entry = download(api, name, round_number, data_dir / name)
            manifest["files"].append(entry)
            write_json(data_dir / "download_manifest.json", manifest)
            print(f"Downloaded {name}: {entry['bytes']:,} bytes", flush=True)
        manifest.update(status="complete", completed_at_utc=utc_now())
        write_json(data_dir / "download_manifest.json", manifest)
        metadata = read_json(data_dir / "features.json")
        small = [name for name in metadata["feature_sets"]["small"] if name != "feature_country"]
        if small != features:
            raise ValueError("Current small numeric feature definition differs from the saved baseline")
        live = pd.read_parquet(data_dir / "live.parquet", columns=["numerai_ticker"] + features)
        check_live(live, features)
        raw = np.asarray(booster.predict(live[features], num_threads=args.n_jobs, validate_features=True))
        if raw.shape != (len(live),) or not np.isfinite(raw).all() or np.unique(raw).size < 2:
            raise ValueError("Saved model produced invalid or constant live predictions")
        submission = pd.DataFrame({"numerai_ticker": live["numerai_ticker"].to_numpy(),
                                   "prediction": pd.Series(raw).rank(method="average", pct=True).to_numpy()})
        filename = f"XinchenGeng_baseline_live_round{round_number}.csv"
        temporary_csv = output / (filename + ".part")
        submission.to_csv(temporary_csv, index=False, float_format="%.17g", lineterminator="\n")
        roundtrip = pd.read_csv(temporary_csv, dtype={"numerai_ticker": str}, float_precision="round_trip")
        if not roundtrip["numerai_ticker"].equals(submission["numerai_ticker"]):
            raise ValueError("CSV ticker round-trip changed the submission identity")
        if not np.array_equal(roundtrip["prediction"].to_numpy(), submission["prediction"].to_numpy()):
            raise ValueError("CSV prediction round-trip lost numeric precision")
        after = current_round(api)
        if before != after:
            raise ValueError("Numerai round changed during generation; generate a fresh live CSV")
        temporary_csv.replace(output / filename)
        for source, target in ((model_path, "B0_small_main.txt"), (run_path, "baseline_run.json"),
                               (protocol_path, "baseline_protocol.json"), (data_dir / "features.json", "features.json"),
                               (data_dir / "download_manifest.json", "live_data_manifest.json")):
            shutil.copyfile(source, output / target)
        record.update(status="complete", generated_at_utc=utc_now(), current_round_after=after,
                      features=features, live_rows=len(live), csv_file=filename,
                      columns=list(submission.columns), prediction_transform=TRANSFORM,
                      prediction_min=float(submission.prediction.min()), prediction_max=float(submission.prediction.max()),
                      unique_predictions=int(submission.prediction.nunique()),
                      raw_prediction_min=float(raw.min()), raw_prediction_max=float(raw.max()),
                      model_sha256=sha256(output / "B0_small_main.txt"),
                      protocol_sha256=sha256(output / "baseline_protocol.json"),
                      baseline_run_sha256=sha256(output / "baseline_run.json"),
                      feature_metadata_sha256=sha256(output / "features.json"),
                      live_parquet_sha256=sha256(data_dir / "live.parquet"),
                      download_manifest_sha256=sha256(output / "live_data_manifest.json"),
                      csv_sha256=sha256(output / filename), generator_sha256=sha256(Path(__file__).resolve()),
                      environment={"python": platform.python_version(), "lightgbm": lgb.__version__,
                                   "pandas": pd.__version__, "numpy": np.__version__},
                      official_submission_spec="https://docs.numer.ai/numerai-tournament/submissions",
                      scope="Current-round live prediction handoff only. No Numerai account upload or submission receipt.")
        write_json(output / "live_generation.json", record)
        print(json.dumps({"status": "complete", "round": round_number, "rows": len(live),
                          "csv": str(output / filename), "model_sha256": record["model_sha256"]}), flush=True)
    except BaseException as exc:
        response = getattr(exc, "response", None)
        record.update(status="failed", failed_at_utc=utc_now(), error_type=type(exc).__name__,
                      http_status=getattr(response, "status_code", None))
        write_json(output / "live_generation.json", record)
        raise


def main() -> int:
    default_baseline = ROOT / "results/xinchen_baseline_20261105_latest"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=default_baseline if default_baseline.is_dir() else HERE,
                        help="Completed historical baseline directory, or a previously generated handoff directory")
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory for CSV and handoff artifacts")
    parser.add_argument("--data-dir", type=Path, help="Separate fresh directory for complete downloaded live data")
    parser.add_argument("--round", type=int, dest="round_number", help="Expected current round; fail if it has changed")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.n_jobs < 1 or (args.round_number is not None and args.round_number < 1):
        parser.error("n-jobs and round must be positive")
    # Signed URLs stay in memory, including on error; suppress library URL logs.
    logging.disable(logging.CRITICAL)
    try:
        generate(args)
        return 0
    except BaseException as exc:
        response = getattr(exc, "response", None)
        print(f"Live generation failed: {type(exc).__name__}, HTTP {getattr(response, 'status_code', None)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
