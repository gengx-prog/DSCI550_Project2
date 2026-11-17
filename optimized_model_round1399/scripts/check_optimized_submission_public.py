"""Read public Numerai submission evidence without authenticating or uploading.

The public API exposes submission IDs and accepted ticker counts, but does not
prove which local CSV was uploaded. Correlate a changed ID with an observed
website receipt before claiming that the optimized artifact was submitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

import requests

ENDPOINT = "https://api-tournament.numer.ai/"
QUERY = """
query PublicSubmissionEvidence($modelName: String!, $modelId: ID!, $round: Int!) {
  profile: v3UserProfile(modelName: $modelName, tournament: 8) {
    id username accountName tournament
  }
  performances: v2RoundModelPerformances(
    modelId: $modelId, tournament: 8, submittedOnly: true
  ) {
    roundNumber submissionId tickersSubmittedCount tickersAcceptedCount
    roundResolved roundTarget roundScoreTime roundResolveTime
    allSubmissionScores { version value submissionId roundNumber day date }
  }
  rounds(tournament: 8, number: $round) {
    number tournament target openTime closeTime closeStakingTime scoreTime resolveTime
  }
}
"""


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Preserving previous public snapshot: {output}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.model_name):
        raise ValueError("Expected an explicit Numerai model name")
    model_id = str(uuid.UUID(args.model_id))
    previous_id = (str(uuid.UUID(args.previous_submission_id))
                   if args.previous_submission_id else None)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"query": QUERY, "variables": {"modelName": args.model_name.lower(),
               "modelId": model_id, "round": args.round}}
    write_json(output / "request.json", payload)
    # No NumerAPI instance, environment API-key loading, browser cookies or
    # Authorization header is used. This is a public read-only GraphQL query.
    response = requests.post(ENDPOINT, json=payload, timeout=(30, 90))
    body = response.content
    (output / "response_body.json").write_bytes(body)
    write_json(output / "response_metadata.json", {
        "endpoint": ENDPOINT, "http_status": response.status_code,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_date": response.headers.get("Date"),
        "response_body_sha256": hashlib.sha256(body).hexdigest(),
        "response_body_preserved_without_json_reserialization": True,
        "authentication_used": False, "account_upload_performed": False,
    })
    response.raise_for_status()
    result = response.json()
    if result.get("errors"):
        raise ValueError("Public GraphQL returned errors; original response was saved")
    data = result.get("data") or {}
    profile = data.get("profile")
    if profile is None:
        raise ValueError("Requested public model profile was not found")
    identity_matches = (str(profile.get("id", "")).lower() == model_id
                        and str(profile.get("username", "")).lower() == args.model_name.lower()
                        and profile.get("tournament") == 8)
    if not identity_matches:
        raise ValueError("Requested model name/ID does not match the public profile")
    rows = [r for r in data.get("performances", [])
            if r.get("roundNumber") == args.round]
    rounds = [r for r in data.get("rounds", [])
              if r.get("number") == args.round and r.get("tournament") == 8]
    if len(rounds) != 1:
        raise ValueError("Public round metadata was not uniquely identified")
    observations = []
    for row in rows:
        submission_id = row.get("submissionId")
        submitted, accepted = row.get("tickersSubmittedCount"), row.get("tickersAcceptedCount")
        observations.append({
            "submission_id": submission_id,
            "tickers_submitted_count": submitted, "tickers_accepted_count": accepted,
            "all_submitted_tickers_accepted": (isinstance(submitted, int)
                and submitted > 0 and submitted == accepted),
            "expected_ticker_count_matches": (None if args.expected_tickers is None else
                submitted == accepted == args.expected_tickers),
            "previous_submission_id": previous_id,
            "submission_id_changed_from_previous": (None if previous_id is None else
                bool(submission_id) and submission_id != previous_id),
            "round_resolved": row.get("roundResolved"),
            "submission_scores": row.get("allSubmissionScores"),
        })
    summary = {
        "status": "observed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": ENDPOINT, "model_name": profile["username"], "model_id": model_id,
        "public_model_identity_verified": True, "round": args.round,
        "matching_performance_records": len(rows), "observations": observations,
        "round_schedule": rounds[0], "authentication_used": False,
        "account_upload_performed": False, "optimized_csv_identity_confirmed": False,
        "limitations": [
            "This script only reads public records; it never uploads or changes a submission.",
            "Matching accepted counts alone do not establish which CSV was uploaded.",
            "A previous submission ID must not be presented as a new upload receipt.",
            "The public fields do not verify filename, CSV hash, selected state or website Late status.",
            "A website receipt and local artifact hash are required to link an upload to the optimized CSV.",
            "Round matching is done locally; the API rejected its advertised roundNumberEq argument.",
        ],
        "response_body_sha256": hashlib.sha256(body).hexdigest(),
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-submission-id")
    parser.add_argument("--expected-tickers", type=int)
    args = parser.parse_args()
    if args.round < 1 or (args.expected_tickers is not None and args.expected_tickers < 1):
        parser.error("round and expected tickers must be positive")
    summary = check(args)
    print(json.dumps({"status": summary["status"], "round": summary["round"],
                      "model": summary["model_name"], "observations": summary["observations"],
                      "account_upload_performed": False,
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
