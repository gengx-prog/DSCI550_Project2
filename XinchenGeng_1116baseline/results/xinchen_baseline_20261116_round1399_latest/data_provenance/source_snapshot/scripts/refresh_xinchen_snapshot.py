"""Download a complete, round-pinned v6 data snapshot without changing prior data.

Creates an isolated source/data directory only. Preparation and training are
separate commands. Signed download URLs are kept in memory; saved URLs exclude
their query strings, and failures expose only exception types and HTTP status.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from numerapi import NumerAPI
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
VERSION = "v6.0"
FILES = ("features.json", "train.parquet", "validation.parquet",
         "validation_neutralizer.parquet", "validation_example_preds.parquet")
SCRIPT_NAMES = ("download_data.py", "prepare_data.py", "run_experiments.py",
                "run_xinchen_baseline.py", "validate_xinchen_baseline.py",
                "plot_xinchen_baseline.py", "summarize_xinchen_baseline.py")
COPY_FILES = tuple(f"scripts/{name}" for name in SCRIPT_NAMES) + (
    "references/official_sources.json", "references/v6_example_model.ipynb",
    "references/NUMERAI_EXAMPLE_SCRIPTS_LICENSE.txt", "requirements.txt",
)
LOCAL_TIMEZONE = "America/Los_Angeles"
DATASET_QUERY = """
query ($filename: String!, $round: Int, $tournament: Int) {
  dataset(filename: $filename, round: $round, tournament: $tournament)
}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, content: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sanitized_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("HTTPS dataset URL required")
    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def safe_failure(exc: BaseException) -> dict:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return {"exception_type": type(exc).__name__, "http_status": status}


def download_one(snapshot: Path, name: str, url: str, round_number: int) -> dict:
    destination = snapshot / "data/raw" / VERSION / name
    temporary = destination.with_suffix(destination.suffix + ".part")
    public_url = sanitized_url(url)
    started = time.perf_counter()
    retry = Retry(total=4, connect=4, read=4, status=4, backoff_factor=1,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET"}), respect_retry_after_header=True,
                  raise_on_status=False)
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        # Disable transparent content encoding so byte counts describe the
        # complete remote object and can be checked against Content-Length.
        with session.get(url, stream=True, timeout=(30, 180),
                         headers={"Accept-Encoding": "identity"}) as response:
            response.raise_for_status()
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Unexpected dataset content encoding")
            length_header = response.headers.get("Content-Length")
            expected = int(length_header) if length_header is not None else None
            metadata = {
                "LastModified": response.headers.get("Last-Modified"),
                "ETag": response.headers.get("ETag"), "ContentLength": expected,
            }
            for header in ("x-goog-generation", "x-goog-hash", "x-amz-version-id",
                           "x-amz-checksum-sha256"):
                if response.headers.get(header):
                    metadata[header] = response.headers[header]
            print(f"{name}: downloading", flush=True)
            digest = hashlib.sha256()
            byte_count = 0
            last_log = time.monotonic()
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    stream.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                    if time.monotonic() - last_log >= 20:
                        print(f"{name}: {byte_count / 1e6:.1f} MB downloaded", flush=True)
                        last_log = time.monotonic()
            if byte_count == 0 or (expected is not None and byte_count != expected):
                raise IOError("Incomplete dataset download")
    if name.endswith(".parquet"):
        parquet = pq.ParquetFile(temporary)
        parquet.close()
    else:
        value = json.loads(temporary.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Feature metadata must be an object")
    temporary.replace(destination)
    record = {
        "dataset": f"{VERSION}/{name}", "path": str(destination.relative_to(snapshot)),
        "round": round_number, "url": public_url, "remote_metadata": metadata,
        "bytes": byte_count, "sha256": digest.hexdigest(), "verified_at_utc": utc_now(),
        "status": "downloaded", "seconds": round(time.perf_counter() - started, 2),
    }
    print(f"{name}: complete, {byte_count / 1e6:.1f} MB", flush=True)
    return record


def refresh(snapshot: Path, requested_round: int | None, workers: int) -> int:
    # Check every local dependency/source before creating the destination. All
    # imported packages and the timezone database are resolved before mutation.
    local_zone = ZoneInfo(LOCAL_TIMEZONE)
    sources = list(COPY_FILES)
    proposal_name = "DSCI550_Numerai_Project2_Proposal.pdf"
    if (ROOT / proposal_name).is_file():
        sources.append(proposal_name)
    source_hashes = {}
    for name in sources:
        if not (ROOT / name).is_file():
            raise FileNotFoundError("Required snapshot source is missing")
        source_hashes[name] = sha256(ROOT / name)
    if snapshot == ROOT or ROOT.is_relative_to(snapshot):
        raise ValueError("Snapshot must not be the source root or its ancestor")
    if snapshot.exists() and (not snapshot.is_dir() or any(snapshot.iterdir())):
        raise FileExistsError("Snapshot destination must be absent or empty")
    snapshot.mkdir(parents=True, exist_ok=True)
    raw = snapshot / "data/raw" / VERSION
    raw.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot / "snapshot.json"
    manifest_path = raw / "manifest.json"
    started = datetime.now(timezone.utc)
    record = {
        "status": "started", "owner": "Xinchen Geng", "data_version": VERSION,
        "started_utc": started.isoformat(), "timezone": LOCAL_TIMEZONE,
        "run_date_local": started.astimezone(local_zone).date().isoformat(),
        "requested_round": requested_round, "round": None,
        "source_root": str(ROOT), "source_copy_sha256": source_hashes,
        "refresh_script_sha256": sha256(Path(__file__).resolve()),
        "download_policy": "Complete remote objects pinned to one Numerai round; no cached files, subsets, or hardlinks.",
        "scope": "Download and source snapshot only; preparation and training are separate.",
        "official_sources": [
            "https://docs.numer.ai/numerai-tournament/data",
            "https://numerapi.readthedocs.io/",
            "https://github.com/numerai/numerapi/blob/master/numerapi/base_api.py",
        ],
    }
    manifest = {"status": "started", "version": VERSION, "round": None,
                "available_files": [], "files": [], "failures": []}
    try:
        atomic_json(snapshot_path, record)
        atomic_json(manifest_path, manifest)
        for name in sources:
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
            if sha256(destination) != source_hashes[name]:
                raise IOError("Copied source hash mismatch")
        api = NumerAPI(show_progress_bars=False)
        round_number = requested_round if requested_round is not None else api.get_current_round()
        if not isinstance(round_number, int) or isinstance(round_number, bool) or round_number < 1:
            raise ValueError("No valid Numerai round available")
        record["round"] = manifest["round"] = round_number
        available = api.list_datasets(round_num=round_number)
        manifest["available_files"] = sorted(x for x in available if x.startswith(VERSION + "/"))
        required = {f"{VERSION}/{name}" for name in FILES}
        if not required.issubset(available):
            raise ValueError("Required datasets unavailable for the pinned round")
        # Query every signed URL with the exact same round and tournament. URLs
        # themselves are intentionally never saved or interpolated into logs.
        urls = {}
        for name in FILES:
            args = {"filename": f"{VERSION}/{name}", "round": round_number,
                    "tournament": api.tournament_id}
            url = api.raw_query(DATASET_QUERY, args)["data"]["dataset"]
            sanitized_url(url)
            urls[name] = url
        record["dataset_urls_resolved_utc"] = utc_now()
        atomic_json(snapshot_path, record)
        atomic_json(manifest_path, manifest)
        print(f"Pinned Numerai round {round_number}; downloading {len(FILES)} complete files", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_one, snapshot, name, urls[name], round_number): name
                       for name in FILES}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failure = {"dataset": f"{VERSION}/{name}", **safe_failure(exc)}
                    manifest["failures"].append(failure)
                    print(f"{name}: failed, {failure['exception_type']}, HTTP {failure['http_status']}",
                          flush=True)
                else:
                    manifest["files"].append(result)
                    manifest["files"].sort(key=lambda item: item["dataset"])
                # Only this main thread mutates and atomically writes manifests.
                atomic_json(manifest_path, manifest)
        if manifest["failures"]:
            record["status"] = manifest["status"] = "failed"
            record["failures"] = manifest["failures"]
        else:
            if len(manifest["files"]) != len(FILES):
                raise IOError("Snapshot file coverage incomplete")
            record["status"] = manifest["status"] = "complete"
        record["completed_utc"] = manifest["completed_utc"] = utc_now()
        atomic_json(manifest_path, manifest)
        record["raw_manifest_sha256"] = sha256(manifest_path)
        atomic_json(snapshot_path, record)
        print(f"Snapshot {record['status']}: {snapshot}", flush=True)
        return 0 if record["status"] == "complete" else 1
    except BaseException as exc:
        failure = safe_failure(exc)
        record.update(status="failed", failed_at_utc=utc_now(), error=failure)
        manifest.update(status="failed", failed_at_utc=utc_now(), error=failure)
        for path, content in ((manifest_path, manifest), (snapshot_path, record)):
            try:
                atomic_json(path, content)
            except Exception:
                pass
        print(f"Snapshot failed: {failure['exception_type']}, HTTP {failure['http_status']}",
              file=sys.stderr, flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--round", type=int, dest="round_number")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or (args.round_number is not None and args.round_number < 1):
        parser.error("workers and round must be positive")
    # Libraries may include request URLs in their own exception logs. All useful
    # progress/error information is emitted explicitly and sanitized above.
    logging.disable(logging.CRITICAL)
    try:
        return refresh(args.snapshot.resolve(), args.round_number, args.workers)
    except BaseException as exc:
        failure = safe_failure(exc)
        print(f"Snapshot not started: {failure['exception_type']}, HTTP {failure['http_status']}",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
