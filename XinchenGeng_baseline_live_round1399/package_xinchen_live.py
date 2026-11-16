"""Package an independently validated Xinchen live bundle for local handoff.

Adds reproduction files and a byte-level inventory, then creates and verifies a
new ZIP. Existing bundle files and archives are never overwritten. No network,
Numerai account submission, or GitHub upload is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE
LOCAL_ZONE = "America/Los_Angeles"
SCRIPT_NAMES = ("generate_xinchen_live.py", "validate_xinchen_live.py",
                "package_xinchen_live.py")
BLOCKED_PARTS = {".git", ".venv", "venv", "__pycache__", "node_modules"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return digest_stream(stream)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path.name}")
    return value


def local_time(value: str | None) -> str:
    if value is None:
        return "not provided by the recorded API response"
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Recorded timestamp must include its timezone")
    return parsed.astimezone(ZoneInfo(LOCAL_ZONE)).strftime("%Y-%m-%d %H:%M:%S %Z")


def bundle_files(bundle: Path) -> list[Path]:
    files = []
    for path in sorted(bundle.rglob("*")):
        relative = path.relative_to(bundle)
        require(not path.is_symlink() and not path.is_junction(),
                f"Links cannot be packaged: {relative}")
        require(not (set(relative.parts) & BLOCKED_PARTS)
                and path.suffix not in {".pyc", ".part"},
                f"Unexpected temporary or environment content: {relative}")
        if path.is_file():
            files.append(path)
    return files


def copy_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            output_file.write(chunk)
    require(sha256(source) == sha256(destination), f"Copied bytes differ: {destination.name}")


def write_new(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def readme(generation: dict, validation: dict, run: dict) -> str:
    number = generation["round"]
    csv_name = generation["csv_file"]
    info = generation["round_info"]
    mode = generation["generation_mode"]
    description = ("Current-round prediction handoff" if mode == "current_round"
                   else "Historical closed-round prediction archive")
    recorded_window = "open" if generation["submission_window_open_at_generation"] else "closed"
    source_completed = local_time(run.get("completed_utc"))
    generated = local_time(generation["generated_at_utc"])
    general_deadline = local_time(info.get("closeTime"))
    staking_deadline = local_time(info.get("closeStakingTime"))
    checked = (str(validation["current_round_at_validation"])
               if validation.get("current_round_queried") else "not queried (offline validation)")
    rows = generation["live_rows"]
    source = generation["baseline_source"].replace("`", "")
    return f"""# Xinchen Geng baseline live predictions — round {number}

{description}. Pass [{csv_name}]({csv_name}) to the submission owner.
It contains **{rows:,}** predictions for Numerai **{generation['data_version']}**
round **{number}**, with columns `numerai_ticker,prediction` and average
percentile ranks in `[0,1]`. Every official live ticker is covered exactly once.

## Recorded round and deadlines

- Generation: **{generated}** ({LOCAL_ZONE}).
- API data datestamp: `{info.get('dataDatestamp')}`.
- General submission deadline (`closeTime`): **{general_deadline}** ({LOCAL_ZONE}).
- Staking deadline (`closeStakingTime`): **{staking_deadline}** ({LOCAL_ZONE}).
- General submission window recorded at generation: **{recorded_window}**.
- API current round before/after generation: **{generation['current_round_before']} / {generation['current_round_after']}**.
- API current round at validation: **{checked}**.

The two deadlines are separate API fields. This package does not extend either
deadline. Check the current round and submission window before uploading; this
CSV is specific to round {number}. Packaging makes no additional API request.

## Model source and checks

The saved `{generation['model_name']}` model comes from `{source}`. Its historical
training run completed **{source_completed}** ({LOCAL_ZONE}); training used
**{run['train_sample_rows']:,}** sampled rows and **{len(generation['features'])}**
features. The live generation stage reused that saved model and recorded
`training_performed=false`. The source run, frozen protocol and exact model
identity are included; the source run's actual timestamps describe when it was fitted.

Model SHA256: `{generation['model_sha256']}`.

Independent validation passed for all **{validation['full_model_and_rank_check']['rows']:,}**
predictions, including saved-model inference, independent average ranks, feature
order, exact ticker coverage and artifact hashes. Maximum absolute rank error:
`{validation['full_model_and_rank_check']['maximum_absolute_error']:.3g}`.

**No Numerai account submission or staking was performed. No submission receipt
is included.** This script creates local files only and does not upload to GitHub.

## Contents and reproduction

- `{csv_name}`: prediction CSV for this round.
- `B0_small_main.txt`, `baseline_run.json`, `baseline_protocol.json`: saved model and source identity.
- `features.json`, `live_data_manifest.json`, `live_generation.json`: exact official-data and generation records.
- `validation.json`: original independent audit, preserved unchanged.
- `audit_data/live.parquet`: exact downloaded live data for offline checking.
- `generate_xinchen_live.py`, `validate_xinchen_live.py`, `package_xinchen_live.py`, `requirements.txt`: reproduction tools.
- `HANDOFF_MANIFEST.json`: byte counts and SHA256 for every delivered file except the manifest itself.

Install the pinned requirements in a fresh Python environment. Then, from the
extracted handoff directory, independently check the complete CSV:

```powershell
python validate_xinchen_live.py --bundle . --live audit_data/live.parquet --report validation_recheck.json{' --allow-closed-round' if mode == 'historical_closed_round' else ''}
```

For a current-round API check, add `--check-current-round`. That check fails once
the round changes. Choose a fresh report path to preserve the original audit.

Generate a future current round from the included saved model using fresh paths:

```powershell
python generate_xinchen_live.py --baseline . --output ../baseline_live_next --data-dir ../baseline_live_next_data
```

Source machine paths retained in the JSON records are provenance. Offline checks
use the files bundled here. Official submission format:
https://docs.numer.ai/numerai-tournament/submissions
"""


def package(bundle: Path, live: Path, output_zip: Path) -> dict:
    bundle, live, output_zip = bundle.resolve(), live.resolve(), output_zip.resolve()
    require(bundle.is_dir(), "Bundle directory does not exist")
    require(live.is_file(), "Exact original live Parquet is required")
    require(not output_zip.exists(), "Archive already exists; choose a new --output-zip")
    require(not output_zip.is_relative_to(bundle), "Keep the ZIP outside its source bundle")
    summary_path = output_zip.with_suffix(output_zip.suffix + ".json")
    require(not summary_path.exists(), "Archive summary already exists; choose a new --output-zip")
    bundle_files(bundle)
    generation = load_json(bundle / "live_generation.json")
    validation = load_json(bundle / "validation.json")
    download = load_json(bundle / "live_data_manifest.json")
    run = load_json(bundle / "baseline_run.json")
    number = generation.get("round")
    require(type(number) is int and number > 0, "Invalid generated round")
    require(generation.get("status") == download.get("status") == run.get("status") == "complete"
            and validation.get("status") == "passed", "Generation, source run or audit is incomplete")
    require(number == validation.get("round") == download.get("round"), "Round records differ")
    require(generation.get("data_version") == validation.get("data_version")
            == download.get("data_version") == "v6.0", "Data versions differ")
    require(generation.get("generation_mode") in {"current_round", "historical_closed_round"}
            and generation["generation_mode"] == validation.get("generation_mode"), "Generation modes differ")
    require(generation.get("account_submission_performed") is False
            and generation.get("training_performed") is False,
            "Expected saved-model generation without an account submission")
    require(validation.get("exact_live_ticker_coverage") is True
            and validation["full_model_and_rank_check"]["rows"] == generation.get("live_rows"),
            "Full independent prediction coverage is required")
    if generation["generation_mode"] == "current_round":
        require(validation.get("current_round_queried") is True
                and validation.get("current_round_at_validation") == number,
                "Current-round handoff requires validation with --check-current-round")
    csv_name = generation.get("csv_file")
    require(csv_name == f"XinchenGeng_baseline_live_round{number}.csv", "Unexpected CSV filename")
    require(sha256(bundle / "live_generation.json") == validation.get("generation_sha256"),
            "Generation record changed after validation")
    require(sha256(bundle / csv_name) == generation.get("csv_sha256") == validation.get("csv_sha256"),
            "CSV bytes changed after validation")
    checks = ((bundle / "B0_small_main.txt", "model_sha256"),
              (bundle / "baseline_protocol.json", "protocol_sha256"),
              (bundle / "baseline_run.json", "baseline_run_sha256"),
              (bundle / "features.json", "feature_metadata_sha256"),
              (bundle / "live_data_manifest.json", "download_manifest_sha256"),
              (live, "live_parquet_sha256"))
    for path, key in checks:
        require(sha256(path) == generation.get(key) == validation["artifact_sha256"].get(key),
                f"Artifact changed after validation: {path.name}")
    require(run.get("model_sha256") == generation["model_sha256"]
            and run.get("protocol_sha256") == generation["protocol_sha256"],
            "Source run model/protocol identity differs")
    scripts = {name: HERE / name for name in SCRIPT_NAMES}
    require(sha256(scripts["generate_xinchen_live.py"]) == generation.get("generator_sha256"),
            "Current generator differs from the one recorded at generation")
    copies = [(source, bundle / name) for name, source in scripts.items()]
    copies.extend(((ROOT / "requirements.txt", bundle / "requirements.txt"),
                   (live, bundle / "audit_data/live.parquet")))
    texts = {bundle / "README.md": readme(generation, validation, run),
             bundle / ".gitattributes": "# Preserve the exact CSV, model and recorded evidence bytes.\n* -text\n"}
    manifest_path = bundle / "HANDOFF_MANIFEST.json"
    for source, destination in copies:
        require(source.is_file(), f"Missing reproduction source: {source.name}")
        require(not destination.exists(), f"Preserving existing file: {destination}")
    for destination in [*texts, manifest_path]:
        require(not destination.exists(), f"Preserving existing file: {destination}")

    for source, destination in copies:
        copy_new(source, destination)
    for path, content in texts.items():
        write_new(path, content)
    files = bundle_files(bundle)
    manifest = {"schema_version": 1, "kind": "Xinchen Geng baseline live handoff",
                "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
                "round": number, "data_version": generation["data_version"],
                "generation_mode": generation["generation_mode"],
                "prediction_rows": generation["live_rows"], "csv_file": csv_name,
                "model_sha256": generation["model_sha256"],
                "source_baseline_run_sha256": generation["baseline_run_sha256"],
                "source_baseline_completed_utc": run.get("completed_utc"),
                "independent_validation": "passed", "account_submission_performed": False,
                "manifest_scope": "All delivered files except HANDOFF_MANIFEST.json itself",
                "files": [{"path": path.relative_to(bundle).as_posix(),
                           "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]}
    write_new(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    prefix = f"XinchenGeng_baseline_live_round{number}/"
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in [*files, manifest_path]:
            archive.write(path, prefix + path.relative_to(bundle).as_posix())
    with zipfile.ZipFile(output_zip) as archive:
        require(archive.testzip() is None, "ZIP CRC check failed")
        expected_names = {prefix + entry["path"] for entry in manifest["files"]}
        expected_names.add(prefix + manifest_path.name)
        require(len(archive.namelist()) == len(expected_names)
                and set(archive.namelist()) == expected_names, "ZIP inventory differs")
        require(archive.read(prefix + manifest_path.name) == manifest_path.read_bytes(),
                "ZIP manifest bytes differ")
        for entry in manifest["files"]:
            name = prefix + entry["path"]
            require(archive.getinfo(name).file_size == entry["bytes"], f"ZIP file size differs: {name}")
            with archive.open(name) as stream:
                require(digest_stream(stream) == entry["sha256"], f"ZIP file hash differs: {name}")
    summary = {"status": "passed", "zip": str(output_zip), "zip_sha256": sha256(output_zip),
               "bytes": output_zip.stat().st_size, "files": len(files) + 1,
               "round": number, "prediction_rows": generation["live_rows"],
               "csv": str(bundle / csv_name), "independent_validation": "passed",
               "zip_inventory_and_hashes": "passed", "account_submission_performed": False}
    write_new(summary_path, json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Freshly generated and validated live bundle")
    parser.add_argument("--live", type=Path, required=True, help="Exact original downloaded live.parquet")
    parser.add_argument("--output-zip", type=Path, required=True, help="New ZIP outside the bundle")
    args = parser.parse_args()
    print(json.dumps(package(args.bundle, args.live, args.output_zip), indent=2))


if __name__ == "__main__":
    main()
