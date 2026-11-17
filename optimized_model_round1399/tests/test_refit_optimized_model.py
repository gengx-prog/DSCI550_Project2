"""Weekly refit keeps selection frozen and enforces the live label boundary."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import refit_optimized_model as refit
from run_experiments import sha256_file, write_json


class WeeklyRefitTests(unittest.TestCase):
    def fixture(self, folder):
        root, source = Path(folder) / "snapshot", Path(folder) / "experiment"
        raw = root / "data/raw/v6.0"
        raw.mkdir(parents=True)
        source.mkdir()
        metadata = {"feature_sets": {"small": ["feature_a"], "medium": ["feature_a", "feature_b"]}}
        write_json(raw / "features.json", metadata)
        selected = refit.MODEL_IDS[2]
        write_json(source / "selection.json", {"selected_model": selected,
            "selection_phase": "cv", "development_used_for_selection": False,
            "candidates": {name: {"selection_score": float(index == 2)}
                           for index, name in enumerate(refit.MODEL_IDS)}})
        write_json(source / "plan.json", {"candidates": refit.candidate_specs(1)})
        self.refresh_source(source, selected)
        live = pd.DataFrame({"numerai_ticker": ["A", "B"],
            "date": pd.Timestamp("2026-11-13"), "feature_a": [0.0, 1.0], "feature_b": [1.0, 0.0]})
        live_path = Path(folder) / "live.parquet"
        live.to_parquet(live_path, index=False)
        pool = pd.DataFrame({"numerai_ticker": ["A", "A", "A", "A"],
            "date": pd.to_datetime(["2026-08-07", "2026-08-13", "2026-08-14", "2026-12-01"]),
            "feature_a": [0.0, 1.0, 2.0, 3.0], "feature_b": [3.0, 2.0, 1.0, 0.0],
            refit.TARGET: [0.0, 0.5, 1.0, 0.5]})
        args = argparse.Namespace(data_root=root, selection=source / "selection.json",
            live_data=live_path, output=Path(folder) / "refit", n_jobs=1)
        return args, source, pool, metadata["feature_sets"], selected

    def refresh_source(self, source, selected):
        write_json(source / "run.json", {"status": "complete", "data_version": "v6.0",
            "target": refit.TARGET, "selected_model": selected, "n_jobs": 1,
            "output_sha256": {name: sha256_file(source / name) for name in ["selection.json", "plan.json"]}})

    def test_only_frozen_winner_refits_with_strict_cutoff_and_reloads(self):
        with tempfile.TemporaryDirectory() as folder:
            args, source, pool, feature_sets, selected = self.fixture(folder)
            captured = {}

            def train_only(train, evaluation, supplied_features, out_dir, n_jobs, candidate_ids):
                captured.update(train=train.copy(), candidate_ids=candidate_ids, n_jobs=n_jobs)
                self.assertEqual(supplied_features, feature_sets)
                model_dir = out_dir / selected
                model_dir.mkdir(parents=True)
                write_json(model_dir / "bundle.json", {"model_id": selected})
                result = evaluation[refit.KEYS].copy()
                result[selected] = [0.5, 1.0]
                return result, [{"model": selected, "sample_keys_sha256": "saved-sample-hash"}]

            with patch.object(refit, "load_research_pool", return_value=(pool, feature_sets, {"rows": 4})), \
                    patch.object(refit, "train_predict_candidates", side_effect=train_only) as fitting, \
                    patch.object(refit, "predict_bundle", return_value=np.array([0.5, 1.0])) as reload:
                result = refit.run(args, SimpleNamespace(run_started=False))
            self.assertEqual(captured["candidate_ids"], [selected])
            self.assertEqual(captured["train"].date.tolist(),
                             list(pd.to_datetime(["2026-08-07", "2026-08-13"])))
            self.assertEqual(result["excluded_at_or_after_maturity_cutoff"], 2)
            self.assertEqual(result["train_cutoff_exclusive"], pd.Timestamp("2026-08-14"))
            self.assertFalse(result["candidate_selection_performed"])
            self.assertEqual(result["status"], "complete")
            fitting.assert_called_once()
            reload.assert_called_once()
            saved = json.loads((args.output / "final_training.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["selection_sha256"], sha256_file(source / "selection.json"))
            self.assertEqual(saved["models"][0]["sample_keys_sha256"], "saved-sample-hash")
            self.assertTrue((args.output / "final_live_preview.parquet").is_file())

    def test_existing_output_is_preserved_without_loading_or_training(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, _, _, _ = self.fixture(folder)
            args.output.mkdir()
            original = args.output / "keep.txt"
            original.write_text("existing result", encoding="utf-8")
            with patch.object(refit, "load_research_pool") as loader, \
                    patch.object(refit, "train_predict_candidates") as fitting:
                with self.assertRaises(FileExistsError):
                    refit.run(args, SimpleNamespace(run_started=False))
            self.assertEqual(original.read_text(encoding="utf-8"), "existing result")
            loader.assert_not_called()
            fitting.assert_not_called()

    def test_tampered_selection_incomplete_run_and_development_selection_fail(self):
        for corruption in ["hash", "incomplete", "development"]:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as folder:
                args, source, _, _, selected = self.fixture(folder)
                if corruption == "incomplete":
                    record = json.loads((source / "run.json").read_text(encoding="utf-8"))
                    record["status"] = "started"
                    write_json(source / "run.json", record)
                else:
                    selection = json.loads(args.selection.read_text(encoding="utf-8"))
                    selection["development_used_for_selection"] = True
                    write_json(args.selection, selection)
                    if corruption == "development":
                        self.refresh_source(source, selected)
                with self.assertRaises(ValueError):
                    refit.checked_selection(args.selection)

    def test_changed_fixed_plan_and_inconsistent_saved_winner_fail(self):
        for corruption in ["plan", "winner"]:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as folder:
                args, source, _, _, selected = self.fixture(folder)
                if corruption == "plan":
                    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
                    plan["candidates"][selected]["max_rows"] = 123
                    write_json(source / "plan.json", plan)
                else:
                    selection = json.loads(args.selection.read_text(encoding="utf-8"))
                    selection["candidates"][refit.MODEL_IDS[0]]["selection_score"] = 99.0
                    write_json(args.selection, selection)
                self.refresh_source(source, selected)
                with self.assertRaises(ValueError):
                    refit.checked_selection(args.selection)

    def test_invalid_live_dates_and_tickers_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, _, _, _ = self.fixture(folder)
            live = pd.read_parquet(args.live_data)
            live.loc[1, "date"] = pd.Timestamp("2026-11-14")
            live.to_parquet(args.live_data, index=False)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                refit.checked_live(args.live_data, ["feature_a"])
            live.loc[1, "date"] = live.loc[0, "date"]
            live.loc[1, "numerai_ticker"] = live.loc[0, "numerai_ticker"]
            live.to_parquet(args.live_data, index=False)
            with self.assertRaises(ValueError):
                refit.checked_live(args.live_data, ["feature_a"])


if __name__ == "__main__":
    unittest.main()
