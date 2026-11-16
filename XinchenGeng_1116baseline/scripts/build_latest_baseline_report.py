"""Build a two-page baseline refresh report from completed, validated evidence.

This authoring tool does not download data, train a model, or calculate new
scores. Run the PDF skill's artifact marker before invoking it to create a PDF.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


TITLE = "Baseline on Refreshed Official Data"
REQUIRED_RESULTS = (
    "run.json", "protocol.json", "metrics.csv", "per_date_metrics.csv",
    "validation.json", "freshness_validation.json", "data_update_comparison.json",
    "annual_corr.csv", "failure_periods.csv",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(value):
    if value is None or str(value).strip().lower() in {"", "nan", "none"}:
        return "unavailable"
    return f"{float(value):.6f}"


def date_only(value):
    return str(value)[:10]


def assemble(snapshot_root, results, previous):
    run = read_json(results / "run.json")
    protocol = read_json(results / "protocol.json")
    audit = read_json(results / "validation.json")
    freshness = read_json(results / "freshness_validation.json")
    comparison = read_json(results / "data_update_comparison.json")
    snapshot = read_json(snapshot_root / "snapshot.json")
    version = run["data_version"]
    manifest = read_json(snapshot_root / "data/raw" / version / "manifest.json")
    prior_protocol = read_json(previous / "protocol.json")
    require(run.get("status") == snapshot.get("status") == comparison.get("status") == "complete",
            "Completed run, snapshot and comparison are required")
    require(audit.get("status") == freshness.get("status") == "passed", "Both audits must pass")
    require(run.get("historical_holdout_previously_observed") is True,
            "This report is for the previously observed historical baseline protocol")
    fixed_fields = ("features", "params", "seed", "max_train_rows", "target", "data_version")
    require(all(protocol.get(key) == prior_protocol.get(key) for key in fixed_fields),
            "Old and new fixed protocols differ; a data-only refresh report would be misleading")
    require(manifest.get("round") == snapshot.get("round"), "Raw manifest round differs from snapshot")

    metrics = {(r["split"], r["metric"]): r for r in read_csv(results / "metrics.csv")}
    require(set(metrics) == {(s, m) for s in ("tuning", "holdout") for m in ("CORR", "NCORR")},
            "Expected CORR and NCORR summaries for both periods")
    annual = read_csv(results / "annual_corr.csv")
    failures = [r for r in read_csv(results / "failure_periods.csv") if r["split"] == "holdout"]
    date_scores = read_csv(results / "per_date_metrics.csv")
    require(len(date_scores) == audit["all_date_corr"]["dates"], "Audit and date scores differ")
    require(sum(int(r["n_rows"]) for r in date_scores) == audit["all_date_corr"]["rows"],
            "Audit and prediction-row coverage differ")
    started = datetime.fromisoformat(run["started_utc"])
    require(started.tzinfo is not None, "Run timestamp requires a timezone")
    local_date = started.astimezone(ZoneInfo("America/Los_Angeles")).date().isoformat()
    round_number = snapshot["round"]
    holdout = run["splits"]["holdout"]
    raw_max = freshness.get("raw_validation_max_date")
    if raw_max is None:
        cleaning = read_json(snapshot_root / "data/processed" / version / "cleaning_report.json")
        raw_max = cleaning["raw"]["validation"]["date_max"]
    mature_max = freshness.get("mature_validation_max_date", date_only(holdout["max_date"]))
    download_date = snapshot.get("run_date_local", local_date)
    hcorr, hncorr = metrics[("holdout", "CORR")], metrics[("holdout", "NCORR")]
    params = protocol["params"]
    embargo = audit["embargo"]
    total_dates, total_rows = audit["all_date_corr"]["dates"], audit["all_date_corr"]["rows"]

    summary = (
        f"The fixed LightGBM baseline was retrained and evaluated on the official {run['data_version']} "
        f"round {round_number} snapshot downloaded on {download_date} (America/Los_Angeles). "
        f"Historical holdout mean CORR is {score(hcorr['mean'])} across {holdout['dates']} dates; "
        f"mean NCORR is {score(hncorr['mean'])}. The historical holdout was viewed in earlier "
        "experiments. These are reproducible development results and do not establish "
        "performance on a fresh unseen test set or in live rounds."
    )
    setup = (
        f"{run['model']} uses {len(protocol['features'])} small numeric features and "
        f"{protocol['target']}. A fixed {run['train_sample_rows']:,}-row sample (seed "
        f"{protocol['seed']}) is drawn from {run['train_input_rows']:,} eligible training rows "
        f"dated {date_only(run['splits']['train']['min_date'])} to "
        f"{date_only(run['splits']['train']['max_date'])}. LightGBM uses {params['n_estimators']} "
        f"trees, learning rate {params['learning_rate']}, depth {params['max_depth']}, "
        f"{params['num_leaves']} leaves and column sampling {params['colsample_bytree']}. "
        "The protocol matches the previous run; no validation-based tuning, early stopping, "
        "selection or refitting is performed. Missing target rows are excluded."
    )
    chronology = (
        f"The configured embargo is {embargo['configured_weekdays']} Monday-Friday weekdays. "
        f"Retained train-to-tuning and tuning-to-holdout boundaries are separated by "
        f"{embargo['train_to_tuning_weekdays']} and {embargo['tuning_to_holdout_weekdays']} "
        "weekdays, respectively. CORR is the proposal's primary metric; NCORR is reported "
        "separately. Each observation date has equal weight and standard deviation uses "
        "ddof=0. The period named tuning is not used to tune this fixed baseline."
    )

    keys = comparison["prediction_keys"]
    new_dates = comparison.get("new_dates", [])
    removed_dates = comparison.get("removed_dates", [])
    changes = comparison["common_prediction_changes"]
    new_phrase = f"{len(new_dates)} scored date{'s' if len(new_dates) != 1 else ''}"
    if 0 < len(new_dates) <= 3:
        new_phrase += " (" + ", ".join(item["date"] for item in new_dates) + ")"
    refresh = (
        f"Compared with {previous.name}, the refresh adds {new_phrase} and {keys['added']:,} "
        f"prediction rows; {keys['removed']:,} rows and {len(removed_dates)} dates are removed. "
        f"Among {keys['common']:,} common rows, {changes['target']['changed_rows']:,} targets "
        f"and {changes['prediction']['changed_rows']:,} predictions changed. "
        f"Raw validation dates extend to {raw_max}; mature eligible labels end on {mature_max}. "
    )
    if not new_dates:
        refresh += "No newly matured scored date was added. "
    else:
        refresh += "Newly available dates alone do not establish stable improvement. "
    if comparison.get("model_sha256", {}).get("identical"):
        refresh += "The saved model hash is unchanged."
    else:
        refresh += "The saved model hash differs; consult the recorded input and sample comparison."

    checked_dates = freshness.get("affected_date_checks", [])
    audit_text = (
        f"Independent validation passed for all {total_dates} CORR dates and {total_rows:,} "
        f"prediction rows (maximum CORR error {audit['all_date_corr']['maximum_absolute_error']:.2e}), "
        "including training-sample, boundary and source-hash checks. "
    )
    if audit.get("saved_model_prediction_check", {}).get("rows", 0):
        audit_text += "Saved-model predictions received an independent full-date check. "
    if audit.get("independent_ncorr_spot_check", {}).get("status") == "passed":
        audit_text += "NCORR received an independent full-date spot check. "
    if "affected_date_checks" in freshness:
        date_noun = "date" if len(checked_dates) == 1 else "dates"
        audit_text += f"The extra refresh audit checked {len(checked_dates)} added or revised full {date_noun}. "
    audit_text += (
        f"Training and scoring took {run['elapsed_seconds']:.2f} seconds, including input hashing; "
        f"fitting took {run['training']['training_seconds']:.2f} seconds, with peak process memory "
        f"{run['memory']['peak_rss_gib']:.2f} GiB. Download and preparation are excluded."
    )
    current_round = freshness.get("current_round_after_validation", freshness.get("current_round_after_download"))
    if current_round is not None:
        audit_text += f" The post-run API check returned round {current_round}; this snapshot is pinned to {round_number}."

    if failures:
        worst = min(failures, key=lambda row: float(row["total_corr"]))
        diagnostic = (
            f"The worst consecutive negative-CORR period in the historical holdout spans "
            f"{worst['start_date']} to {worst['end_date']}: {int(worst['n_dates'])} adjacent "
            f"observed dates, with mean CORR {score(worst['mean_corr'])}. "
        )
    else:
        diagnostic = "No consecutive negative-CORR period is recorded in the historical holdout. "
    recent_years = sorted([r for r in annual if r["split"] == "holdout"], key=lambda r: int(r["year"]))[-2:]
    diagnostic += " ".join(
        f"The {r['year']} mean CORR is {score(r['mean_corr'])} ({r['date_min']} to {r['date_max']})."
        for r in recent_years
    )
    diagnostic += " CORR and cumulative CORR are score diagnostics, not investment returns."
    next_steps = (
        "Preserve this fixed baseline and protocol for subsequent comparisons. Evaluate candidate "
        "models on identical dates using paired score differences and examine their behavior "
        "during negative-score periods. Coordinate the live-submission handoff separately; "
        "live submission receipts and live MMC/NMMC results are not documented by this historical evaluation."
    )
    evidence = (
        f"Evidence is in results/{results.name}/: run.json, protocol.json, metrics.csv, "
        "per_date_metrics.csv, validation.json, freshness_validation.json, data_update_comparison.json, "
        "annual_corr.csv and failure_periods.csv. Evidence hashes are listed in report_sources.json. "
        f"To reproduce, create a fresh snapshot pinned to round {round_number} with "
        "refresh_xinchen_snapshot.py, then run the copied prepare, baseline, validate, plot and "
        "summary scripts with one fresh absolute result directory. Full validation requires the "
        "matching raw data, trained model and row-level predictions generated locally."
    )
    rows = []
    for split, label in (("tuning", "Tuning period"), ("holdout", "Historical holdout*")):
        partition = run["splits"][split]
        corr, ncorr = metrics[(split, "CORR")], metrics[(split, "NCORR")]
        rows.append([
            label + "\n" + date_only(partition["min_date"]) + " to " + date_only(partition["max_date"]),
            str(partition["dates"]), f"{partition['rows']:,}", score(corr["mean"]),
            score(corr["std_ddof0"]), score(ncorr["mean"]),
        ])
    return {
        "local_date": local_date, "round": round_number, "owner": protocol.get("owner", "Xinchen Geng"),
        "summary": summary, "setup": setup, "chronology": chronology, "rows": rows,
        "refresh": refresh, "audit": audit_text, "diagnostic": diagnostic,
        "next_steps": next_steps, "evidence": evidence,
    }


def render_pdf(pdf, content, figure):
    styles = getSampleStyleSheet()
    specs = (
        ("ReportTitle", dict(fontName="Helvetica-Bold", fontSize=19, leading=23, spaceAfter=7)),
        ("Meta", dict(fontSize=9, leading=12, textColor=colors.HexColor("#4b5563"), spaceAfter=12)),
        ("Section", dict(fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=9,
                         spaceAfter=5, keepWithNext=True)),
        ("ReportBody", dict(fontSize=9.6, leading=12.8, spaceAfter=5)),
        ("SmallNote", dict(fontSize=8.2, leading=11, spaceAfter=5)),
        ("Cell", dict(fontSize=8.2, leading=11, alignment=TA_CENTER)),
    )
    for name, values in specs:
        settings = dict(fontName="Helvetica", textColor=colors.black)
        settings.update(values)
        styles.add(ParagraphStyle(name, **settings))
    styles.add(ParagraphStyle("CellLeft", parent=styles["Cell"], alignment=TA_LEFT))
    styles.add(ParagraphStyle("CellHeader", parent=styles["Cell"], fontName="Helvetica-Bold", textColor=colors.white))

    def p(text, style="ReportBody"):
        return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])

    def section(title, text, style="ReportBody"):
        return [p(title, "Section"), p(text, style)]

    meta = f"DSCI 550 Project 2 | {content['owner']} | {content['local_date']} | Round {content['round']}"
    story = [p(TITLE, "ReportTitle"), p(meta, "Meta"), p(content["summary"])]
    story += section("Fixed baseline and validation", content["setup"])
    story.append(p(content["chronology"]))
    headers = ["Period and date range", "Dates", "Rows", "CORR\nmean", "CORR\nstd", "NCORR\nmean"]
    cells = [[p(x, "CellHeader") for x in headers]]
    cells += [[p(x, "CellLeft" if i == 0 else "Cell") for i, x in enumerate(row)] for row in content["rows"]]
    table = Table(cells, colWidths=[170, 42, 64, 75, 75, 75], hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263445")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#f2f5f8")),
        ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#d9d9d9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [Spacer(1, 5), table, Spacer(1, 4),
              p("* The historical holdout includes previously observed dates.", "SmallNote")]
    story += section("Changes from the previous data snapshot", content["refresh"])
    story += section("Verification and runtime", content["audit"])
    story += [PageBreak(), p("Historical CORR and next steps", "ReportTitle")]
    width, height = ImageReader(str(figure)).getSize()
    story += [Image(str(figure), width=516, height=516 * height / width), Spacer(1, 3)]
    story += section("Failure periods and limits", content["diagnostic"])
    story += section("Next steps", content["next_steps"])
    story += section("Evidence and reproduction", content["evidence"], "SmallNote")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#4b5563"))
        canvas.drawString(48, 30, f"DSCI 550 | Baseline and Validation | {content['local_date']}")
        canvas.drawRightString(letter[0] - 48, 30, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(pdf), pagesize=letter, rightMargin=48, leftMargin=48,
                            topMargin=40, bottomMargin=46, title=TITLE, author=content["owner"])
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    require(doc.page == 2, f"Report rendered to {doc.page} pages; inspect layout before publication")


def build(args):
    root, results, previous = args.root.resolve(), args.results.resolve(), args.previous.resolve()
    out_dir, pdf = args.out_dir.resolve(), args.pdf.resolve()
    txt = out_dir / (pdf.stem + ".txt")
    sources = out_dir / "report_sources.json"
    require(len({pdf, txt, sources}) == 3, "Output paths collide")
    for destination in (pdf, txt, sources):
        require(args.force or not destination.exists(), f"Output exists; use a new path or --force: {destination}")
    content = assemble(root, results, previous)
    figure = results / "figures/per_date_corr.png"
    require(figure.is_file(), "The completed baseline CORR figure is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        ("Progress", "summary"), ("Fixed baseline and validation", "setup"),
        ("Chronological evaluation", "chronology"),
        ("Changes from the previous data snapshot", "refresh"),
        ("Verification and runtime", "audit"), ("Failure periods and limits", "diagnostic"),
        ("Next steps", "next_steps"), ("Evidence and reproduction", "evidence"),
    ]
    full_text = TITLE + "\n" + f"DSCI 550 Project 2 | {content['owner']} | {content['local_date']} | Round {content['round']}\n\n"
    for heading, key in sections:
        full_text += heading.upper() + "\n" + content[key] + "\n\n"
        if key == "chronology":
            full_text += "Period and dates | Dates | Rows | CORR mean | CORR std | NCORR mean\n"
            full_text += "\n".join(" | ".join(x.replace("\n", ": ") for x in row) for row in content["rows"])
            full_text += "\n* Historical holdout includes previously observed dates.\n\n"
    hashes = {name: sha256(results / name) for name in REQUIRED_RESULTS}
    receipt = {
        "report_date_local": content["local_date"], "timezone": "America/Los_Angeles",
        "round": content["round"], "result_batch": results.name,
        "results_sha256": hashes, "snapshot_json_sha256": sha256(root / "snapshot.json"),
        "raw_manifest_sha256": sha256(root / "data/raw" / read_json(results / "run.json")["data_version"] / "manifest.json"),
        "previous_protocol_sha256": sha256(previous / "protocol.json"),
        "figure_sha256": sha256(figure), "report_builder_sha256": sha256(Path(__file__)),
        "interpretation": "Historical reproducibility and development evidence; no new unseen-test or live-performance claim.",
    }
    # Render to a temporary sibling first so a layout failure leaves prior outputs intact.
    descriptor, temporary_name = tempfile.mkstemp(prefix=".baseline-report-", suffix=".pdf", dir=pdf.parent)
    import os
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        render_pdf(temporary, content, figure)
        require(hashes == {name: sha256(results / name) for name in REQUIRED_RESULTS},
                "Evidence changed while authoring the report")
        temporary.replace(pdf)
        txt.write_text(full_text, encoding="utf-8")
        receipt["output_sha256"] = {pdf.name: sha256(pdf), txt.name: sha256(txt)}
        sources.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"pdf": str(pdf), "text": str(txt), "sources": str(sources), "pages": 2}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Completed isolated data snapshot")
    parser.add_argument("--results", type=Path, required=True, help="Completed and validated current result directory")
    parser.add_argument("--previous", type=Path, required=True, help="Previous result directory used in the comparison")
    parser.add_argument("--out-dir", type=Path, required=True, help="Text report and source-receipt destination")
    parser.add_argument("--pdf", type=Path, required=True, help="Final PDF path, normally under output/pdf")
    parser.add_argument("--force", action="store_true", help="Permit replacing this report's existing outputs")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
