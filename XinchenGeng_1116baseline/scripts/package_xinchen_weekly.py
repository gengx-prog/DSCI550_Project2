"""Create a local, lightweight weekly baseline folder and verified ZIP.

The original evidence bytes and source hashes are preserved. This command does
not train a model, download data, publish to GitHub, or submit to Numerai.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = "ARCHIVE_MANIFEST.json"
RAW_FILES = {
    "features.json", "train.parquet", "validation.parquet",
    "validation_neutralizer.parquet", "validation_example_preds.parquet",
}
REQUIRED_RESULTS = {
    "run.json", "protocol.json", "validation.json", "freshness_validation.json",
    "metrics.csv", "per_date_metrics.csv", "annual_corr.csv",
    "failure_periods.csv", "new_dates_metrics.csv", "data_update_comparison.json",
}
EXTRA_SCRIPTS = (
    "refresh_xinchen_snapshot.py", "compare_xinchen_runs.py",
    "validate_latest_baseline_refresh.py", "build_latest_baseline_report.py",
    "generate_xinchen_live.py", "validate_xinchen_live.py",
    "package_xinchen_weekly.py",
)
TEST_FILES = (
    "test_cleaning.py", "test_metrics.py", "test_live_generation.py",
    "test_live_submission.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def relative_name(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts,
            f"Unsafe relative path: {value}")
    require(not any(":" in part for part in path.parts), f"Unsafe relative path: {value}")
    require(not {part.casefold() for part in path.parts}.intersection(
        {".git", ".venv", "venv", "__pycache__", "node_modules", "secrets", "credentials"}),
        f"Excluded path: {value}")
    require(not path.name.casefold().startswith(".env"), f"Excluded path: {value}")
    return path.as_posix()


def source_bytes(root: Path, relative: str) -> bytes:
    relative = relative_name(relative)
    source = root
    for part in PurePosixPath(relative).parts:
        source = source / part
        require(not source.is_symlink() and not source.is_junction(), f"Source links are excluded: {source}")
    require(source.is_file() and source.resolve().is_relative_to(root), f"Missing source file: {source}")
    return source.read_bytes()


def read_json(root: Path, relative: str) -> dict:
    value = json.loads(source_bytes(root, relative).decode("utf-8"))
    require(isinstance(value, dict), f"Expected an object: {relative}")
    return value


def check_hash(root: Path, relative: str, expected: str) -> None:
    require(digest(source_bytes(root, relative)) == expected, f"Evidence hash changed: {root / relative}")


def build_payload(args: argparse.Namespace) -> tuple[dict[str, bytes], dict]:
    root, result, previous = args.root, args.results, args.previous
    snapshot = read_json(root, "snapshot.json")
    run = read_json(result, "run.json")
    protocol = read_json(result, "protocol.json")
    audit = read_json(result, "validation.json")
    freshness = read_json(result, "freshness_validation.json")
    comparison = read_json(result, "data_update_comparison.json")
    prior = read_json(previous, "protocol.json")
    raw_relative = "data/raw/v6.0/manifest.json"
    raw = read_json(root, raw_relative)
    for label, value in (("snapshot", snapshot), ("run", run), ("raw manifest", raw),
                         ("comparison", comparison)):
        require(value.get("status") == "complete", f"A complete {label} is required")
    require(audit.get("status") == freshness.get("status") == "passed", "Both independent audits must pass")
    require(run.get("historical_holdout_previously_observed") is True, "Expected historical baseline evidence")
    require(run.get("data_version") == protocol.get("data_version") == "v6.0", "Expected fixed v6.0 baseline")
    round_number = snapshot.get("round")
    require(type(round_number) is int and round_number > 0, "Invalid snapshot round")
    require(raw.get("round") == freshness.get("round") == round_number, "Evidence rounds differ")
    require(all(protocol.get(key) == prior.get(key) for key in
                ("features", "params", "seed", "max_train_rows", "target", "data_version", "embargo_weekdays")),
            "Current and previous fixed protocols differ")
    require(protocol.get("seed") == 7 and protocol.get("max_train_rows") == 200000,
            "Expected fixed seed 7 and 200000-row baseline")
    check_hash(root, raw_relative, snapshot["raw_manifest_sha256"])
    require(len(raw.get("files", [])) == len(RAW_FILES) and
            {item.get("dataset") for item in raw["files"]} == {f"v6.0/{name}" for name in RAW_FILES},
            "Expected all five round-pinned raw objects")
    require(all(item.get("round") == round_number and item.get("status") == "downloaded"
                for item in raw["files"]), "Raw object round or completion differs")
    require({"run.json", "protocol.json", "validation.json", "metrics.csv", "per_date_metrics.csv"}
            .issubset(freshness.get("source_files_sha256", {})), "Refresh audit lacks required evidence hashes")
    for relative, expected in freshness["source_files_sha256"].items():
        check_hash(result, relative, expected)
    check_hash(root, "snapshot.json", freshness["snapshot_sha256"])
    check_hash(root, raw_relative, freshness["raw_manifest_sha256"])

    receipt = read_json(args.report_dir, "report_sources.json")
    require(receipt.get("round") == round_number and receipt.get("result_batch") == result.name,
            "Report refers to a different result batch or round")
    for relative, expected in receipt["results_sha256"].items():
        check_hash(result, relative, expected)
    check_hash(root, "snapshot.json", receipt["snapshot_json_sha256"])
    check_hash(root, raw_relative, receipt["raw_manifest_sha256"])
    check_hash(previous, "protocol.json", receipt["previous_protocol_sha256"])
    check_hash(result, "figures/per_date_corr.png", receipt["figure_sha256"])
    require(args.pdf.is_file() and not args.pdf.is_symlink(), "The final PDF must be a regular file")
    require(args.pdf.name in receipt["output_sha256"], "Final PDF missing from report receipt")
    require(digest(args.pdf.read_bytes()) == receipt["output_sha256"][args.pdf.name], "Final PDF hash changed")

    payload: dict[str, bytes] = {}

    def put(name: str, data: bytes) -> None:
        name = relative_name(name)
        require(name not in payload or payload[name] == data, f"Conflicting archive sources: {name}")
        payload[name] = data

    def copy(source_root: Path, relative: str, destination: str | None = None) -> None:
        put(destination or relative, source_bytes(source_root, relative))

    # These are the exact sources copied when the raw snapshot was created.
    copies = snapshot.get("source_copy_sha256")
    require(isinstance(copies, dict) and copies, "Missing frozen source inventory")
    provenance = f"results/{result.name}/data_provenance"
    for relative, expected in sorted(copies.items()):
        data = source_bytes(root, relative)
        require(digest(data) == expected, f"Frozen snapshot source changed: {relative}")
        put(relative, data)
        put(f"{provenance}/source_snapshot/{relative_name(relative)}", data)
    tool_hashes = {
        "refresh_xinchen_snapshot.py": snapshot["refresh_script_sha256"],
        "compare_xinchen_runs.py": comparison["comparison_script_sha256"],
        "validate_latest_baseline_refresh.py": freshness["audit_script_sha256"],
        "build_latest_baseline_report.py": receipt["report_builder_sha256"],
    }
    for name in EXTRA_SCRIPTS:
        relative = f"scripts/{name}"
        data = source_bytes(PROJECT, relative)
        if name in tool_hashes:
            require(digest(data) == tool_hashes[name], f"Workflow source differs from recorded evidence: {name}")
        put(relative, data)
        put(f"{provenance}/source_snapshot/{relative}", data)
    for name in TEST_FILES:
        copy(PROJECT, f"tests/{name}")
    copy(PROJECT, "requirements-report.txt")

    for name in sorted(REQUIRED_RESULTS):
        copy(result, name, f"results/{result.name}/{name}")
    # Include small supplemental records, but never models or row-level data.
    for path in sorted(result.iterdir()):
        if path.is_file() and path.suffix.casefold() in {".csv", ".json"}:
            copy(result, path.name, f"results/{result.name}/{path.name}")
    figure_dir = result / "figures"
    require(figure_dir.is_dir(), "Missing result figures")
    for path in sorted(figure_dir.iterdir()):
        if path.is_file() and path.suffix.casefold() in {".png", ".json"}:
            copy(result, f"figures/{path.name}", f"results/{result.name}/figures/{path.name}")

    for source, name in (
        ("snapshot.json", "snapshot.json"), (raw_relative, "manifest.json"),
        ("data/raw/v6.0/features.json", "features.json"),
        ("data/processed/v6.0/split_manifest.json", "split_manifest.json"),
        ("data/processed/v6.0/cleaning_report.json", "cleaning_report.json"),
        ("data/processed/v6.0/date_counts.csv", "date_counts.csv"),
    ):
        copy(root, source, f"{provenance}/{name}")
    for name in ("official_latest_check.json", "raw_validation_inspection.json"):
        if (root / name).is_file():
            copy(root, name, f"{provenance}/{name}")
    export = {
        "description": "Exact lightweight provenance exports; raw objects remain local.",
        "files_sha256": {name.removeprefix(provenance + "/"): digest(data)
                         for name, data in sorted(payload.items()) if name.startswith(provenance + "/")},
    }
    put(f"{provenance}/export_manifest.json", json_bytes(export))

    copy(args.report_dir, "report_sources.json", "report/report_sources.json")
    for name, expected in receipt["output_sha256"].items():
        relative_name(name)
        require(PurePosixPath(name).name == name, "Report receipt must use plain output filenames")
        data = args.pdf.read_bytes() if name == args.pdf.name else source_bytes(args.report_dir, name)
        require(digest(data) == expected, f"Report output hash changed: {name}")
        put(f"report/{name}", data)
    put(".gitignore", b".venv/\n__pycache__/\n*.pyc\ndata/\nmodels/\n*.part\n.env\nresults/**/predictions*.parquet\n")
    put(".gitattributes", b"# Preserve exact source and evidence hashes.\n* -text\n")
    put("README.md", make_readme(args, run, protocol, freshness, comparison, receipt, payload).encode("utf-8"))
    return payload, {"round": round_number, "result_batch": result.name,
                     "run_date_local": receipt["report_date_local"]}


def make_readme(args, run, protocol, freshness, comparison, receipt, payload) -> str:
    batch = args.results.name
    metrics = list(csv.DictReader(io.StringIO(payload[f"results/{batch}/metrics.csv"].decode("utf-8"))))
    lookup = {(row["split"], row["metric"]): row for row in metrics}
    require(set(lookup) == {(split, metric) for split in ("tuning", "holdout") for metric in ("CORR", "NCORR")},
            "Expected CORR and NCORR for tuning and holdout")
    rows = []
    for split in ("tuning", "holdout"):
        part = run["splits"][split]
        corr, ncorr = lookup[(split, "CORR")], lookup[(split, "NCORR")]
        rows.append(f"| {split} | {str(part['min_date'])[:10]} to {str(part['max_date'])[:10]} | "
                    f"{part['dates']} | {part['rows']:,} | {float(corr['mean']):.6f} | "
                    f"{float(corr['std_ddof0']):.6f} | {float(ncorr['mean']):.6f} |")
    new_dates = ", ".join(item["date"] for item in comparison.get("new_dates", [])) or "none"
    keys = comparison["prediction_keys"]
    model_identical = comparison.get("model_sha256", {}).get("identical")
    model_text = "unchanged" if model_identical else "different; inspect the recorded input and sample comparison"
    round_number = receipt["round"]
    return f"""# {args.output.name}

Xinchen Geng's Baseline & Validation contribution for **{receipt['report_date_local']}** (America/Los_Angeles), using complete official **v6.0 round {round_number}** data. This is a local upload-ready archive; creating it performs no GitHub upload or Numerai account submission.

## Fixed baseline and historical results

`{run['model']}` uses {len(protocol['features'])} small numeric features, `{protocol['target']}`, a fixed {run['train_sample_rows']:,}-row training sample and seed {protocol['seed']}. Parameters and the chronological validation protocol match the previous run. No validation-based model selection, tuning, early stopping or refitting was performed.

| Period | Actual dates | Dates | Rows | Mean CORR | CORR std (ddof=0) | Mean NCORR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

CORR and NCORR give equal weight to each observation date. The historical holdout includes dates viewed in prior experiments. These results are historical development and reproducibility evidence; they do not establish performance on a fresh unseen test set or in live rounds. Cumulative CORR is a score diagnostic, not an investment return.

## Changes and verification

Compared with `{args.previous.name}`, the snapshot adds {keys['added']:,} prediction rows and removes {keys['removed']:,}; {keys['common']:,} rows are shared. Newly scored dates: **{new_dates}**. Raw validation dates extend to **{freshness['raw_validation_max_date']}**; mature eligible labels end on **{freshness['mature_validation_max_date']}**. The saved model hash is {model_text}. Newly available dates alone do not establish stable improvement.

Both independent audits passed. See [all-date validation](results/{batch}/validation.json), [refresh validation](results/{batch}/freshness_validation.json), [frozen protocol](results/{batch}/protocol.json), [metrics](results/{batch}/metrics.csv), [data comparison](results/{batch}/data_update_comparison.json), [annual CORR](results/{batch}/annual_corr.csv) and [negative-score periods](results/{batch}/failure_periods.csv).

The [PDF report](report/{args.pdf.name}) and [report sources](report/report_sources.json) accompany the evidence. [Data provenance](results/{batch}/data_provenance/) contains the original raw-file manifest, feature metadata, cleaning/split records and exact frozen source copies. `ARCHIVE_MANIFEST.json` inventories the bytes of every packaged file except itself. Original audit paths and timestamps are preserved as provenance.

## Reproduce locally

Run these commands from this archive's root using Python 3.13 and a fresh virtual environment. Downloads are public and require roughly 8.3 GB plus room for processed data, models and predictions. A historical round must remain available from the official API; matching the recorded raw hashes is required for exact reproduction.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
$python = (Resolve-Path .\\.venv\\Scripts\\python.exe).Path
$snapshot = Join-Path (Get-Location).Path "snapshots/reproduction_round{round_number}"
$result = Join-Path (Get-Location).Path "results/reproduction_round{round_number}"
& $python scripts/refresh_xinchen_snapshot.py --snapshot $snapshot --round {round_number}
& $python "$snapshot/scripts/prepare_data.py"
& $python "$snapshot/scripts/run_xinchen_baseline.py" --root $snapshot --output $result
& $python "$snapshot/scripts/validate_xinchen_baseline.py" --root $snapshot --output $result
& $python "$snapshot/scripts/plot_xinchen_baseline.py" --output $result
& $python "$snapshot/scripts/summarize_xinchen_baseline.py" --output $result
& $python -m unittest discover -s tests -v
```

The included tests cover cleaning, scoring and live prediction artifacts (`test_cleaning.py`, `test_metrics.py`, `test_live_generation.py` and `test_live_submission.py`). The project's separate research-model experiments and their tests are outside this baseline archive.

Regenerating the PDF with `scripts/build_latest_baseline_report.py` additionally requires the pinned ReportLab dependency: install `requirements-report.txt`. The original frozen `requirements.txt` is preserved unchanged.

Raw/processed Parquet, trained models and row-level predictions are retained locally and excluded from this lightweight archive. Full independent validation needs the matching raw data, trained model and predictions generated locally. The recorded refresh comparison additionally needs the previous snapshot and previous full local results. Live CSV generation and current-round submission readiness are separate from this historical report.
"""


def publish(args: argparse.Namespace, payload: dict[str, bytes], metadata: dict) -> dict:
    inventory = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **metadata, "archive_folder": args.output.name,
        "description": "Lightweight historical baseline upload preparation; no remote upload performed.",
        "excluded": ["raw and processed datasets", "trained models", "row-level prediction Parquet", "execution environments and caches"],
        "file_count": len(payload), "total_uncompressed_bytes": sum(map(len, payload.values())),
        "files": [{"path": name, "bytes": len(data), "sha256": digest(data)}
                  for name, data in sorted(payload.items())],
    }
    payload = {**payload, MANIFEST: json_bytes(inventory)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    local = datetime.fromisoformat(metadata["run_date_local"])
    zip_time = (max(1980, local.year), local.month, local.day, 0, 0, 0)
    descriptor, temporary_zip_name = tempfile.mkstemp(prefix=".weekly-baseline-", suffix=".zip", dir=args.output_zip.parent)
    os.close(descriptor)
    temporary_zip = Path(temporary_zip_name)
    staging = Path(tempfile.mkdtemp(prefix=".weekly-baseline-", dir=args.output.parent))
    try:
        for name, data in sorted(payload.items()):
            target = staging.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        require({path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file()} == set(payload),
                "Folder inventory differs from payload")
        for name, data in payload.items():
            require(digest(source_bytes(staging, name)) == digest(data), f"Folder hash mismatch: {name}")
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in sorted(payload.items()):
                entry = zipfile.ZipInfo(f"{args.output.name}/{name}", date_time=zip_time)
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, data, compresslevel=9)
        with zipfile.ZipFile(temporary_zip) as archive:
            require(archive.testzip() is None, "ZIP CRC integrity failed")
            require(len(archive.namelist()) == len(payload) and set(archive.namelist()) ==
                    {f"{args.output.name}/{name}" for name in payload}, "ZIP inventory differs")
            for name, data in payload.items():
                archived = archive.read(f"{args.output.name}/{name}")
                require(len(archived) == len(data) and digest(archived) == digest(data), f"ZIP hash mismatch: {name}")
        require(not args.output.exists() and not args.output_zip.exists(), "Destinations appeared during packaging")
        # On Windows directory rename fails if the destination already exists.
        staging.rename(args.output)
        # Hard-link publication refuses to replace an existing archive.
        os.link(temporary_zip, args.output_zip)
    finally:
        temporary_zip.unlink(missing_ok=True)
        if staging.exists():
            require(staging.parent.resolve() == args.output.parent.resolve()
                    and staging.name.startswith(".weekly-baseline-"), "Unsafe temporary cleanup path")
            shutil.rmtree(staging)
    return {"status": "complete", **metadata, "folder": str(args.output), "zip": str(args.output_zip),
            "packaged_files": len(payload), "zip_bytes": args.output_zip.stat().st_size,
            "zip_sha256": digest(args.output_zip.read_bytes()), "remote_upload_performed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "results", "previous", "report-dir", "pdf", "output", "output-zip"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("root", "results", "previous", "report_dir"):
        path = getattr(args, name)
        require(not path.is_symlink() and not path.is_junction(), f"Input must not be a link: {path}")
        path = path.resolve(strict=True)
        require(path.is_dir(), f"Input directory missing: {path}")
        setattr(args, name, path)
    for name in ("pdf", "output", "output_zip"):
        path = getattr(args, name)
        require(not path.is_symlink() and not path.is_junction(), f"Path must not be a link: {path}")
        setattr(args, name, path.resolve())
    require(not args.output.exists() and not args.output_zip.exists(), "Choose fresh folder and ZIP destinations")
    require(args.output_zip.suffix.casefold() == ".zip", "Output ZIP requires a .zip extension")
    require(not args.output_zip.is_relative_to(args.output), "Place the ZIP outside the output folder")
    require(args.output != args.output_zip, "Output paths collide")
    require(args.output.name and args.output.name not in {".", ".."}, "Invalid output folder name")
    for source in (args.root, args.results, args.previous, args.report_dir):
        require(not source.is_relative_to(args.output), "Output folder must not contain an input directory")
    payload, metadata = build_payload(args)
    print(json.dumps(publish(args, payload, metadata), indent=2))


if __name__ == "__main__":
    main()
