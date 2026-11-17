"""Guard live identities, label maturity, round metadata and final rank output."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_optimized_live as generation


class OptimizedLiveTests(unittest.TestCase):
    def setUp(self):
        self.features = ["feature_a", "feature_b"]
        self.bundle = {"features": self.features, "feature_set": "medium",
                       "full_training_date_max": "2026-08-07"}
        self.metadata = {"feature_sets": {"medium": self.features + ["feature_country"]}}
        self.info = {"dataDatestamp": 20261113}
        self.frame = pd.DataFrame({"numerai_ticker": ["AAA US", "BBB US", "CCC US"],
                                   "date": ["2026-11-13"] * 3,
                                   "feature_a": [0, 1, np.nan], "feature_b": [3, 2, 1]})

    def test_live_actual_date_and_native_nan_are_preserved(self):
        result = generation.checked_live(self.frame, self.bundle, self.metadata, self.info)
        self.assertTrue(result.feature_a.isna().iloc[2])
        self.assertEqual(result.date.iloc[0], pd.Timestamp("2026-11-13"))

    def test_boundary_mature_label_is_excluded_by_strict_embargo(self):
        self.bundle["full_training_date_max"] = "2026-08-14"
        with self.assertRaisesRegex(ValueError, "65-weekday"):
            generation.checked_live(self.frame, self.bundle, self.metadata, self.info)

    def test_live_date_must_match_pinned_round(self):
        self.info["dataDatestamp"] = 20261112
        with self.assertRaisesRegex(ValueError, "dataDatestamp"):
            generation.checked_live(self.frame, self.bundle, self.metadata, self.info)

    def test_duplicate_live_tickers_are_rejected(self):
        self.frame.loc[1, "numerai_ticker"] = "AAA US"
        with self.assertRaisesRegex(ValueError, "unique"):
            generation.checked_live(self.frame, self.bundle, self.metadata, self.info)

    def test_metadata_order_must_match_saved_model(self):
        self.metadata["feature_sets"]["medium"] = list(reversed(self.features))
        with self.assertRaisesRegex(ValueError, "feature-set order"):
            generation.checked_live(self.frame, self.bundle, self.metadata, self.info)

    def test_average_tie_ranks_match_independent_implementation(self):
        result = generation.checked_submission(self.frame, np.array([.1, .4, .4]))
        np.testing.assert_array_equal(result.prediction, [1 / 3, 2.5 / 3, 2.5 / 3])
        generation.validate_submission(result, self.frame, np.array([.1, .4, .4]))

    def test_corrupt_predictions_and_ticker_order_are_rejected(self):
        raw = np.array([.1, .4, .4])
        result = generation.checked_submission(self.frame, raw)
        for value in (np.nan, np.inf, -0.1, 1.01, .5):
            with self.subTest(value=value):
                altered = result.copy()
                altered.loc[0, "prediction"] = value
                with self.assertRaises(ValueError):
                    generation.validate_submission(altered, self.frame, raw)
        with self.assertRaisesRegex(ValueError, "ticker coverage"):
            generation.validate_submission(result.iloc[::-1], self.frame, raw)

    def test_constant_or_nonfinite_model_output_is_rejected(self):
        for values in ([1, 1, 1], [1, np.nan, 2], [1, np.inf, 2], [1, 2]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "Invalid or constant"):
                    generation.checked_submission(self.frame, np.array(values))

    def api(self):
        now = datetime.now(timezone.utc)
        api = MagicMock(tournament_id=8)
        api.get_current_round.return_value = 1399
        api.list_rounds.return_value = [{"number": 1399, "tournament": 8,
            "target": "target_jupiter_60", "dataDatestamp": 20261113,
            "openTime": now - timedelta(hours=1), "closeTime": now + timedelta(days=1),
            "closeStakingTime": now - timedelta(minutes=1)}]
        api.list_datasets.return_value = ["v6.0/features.json", "v6.0/live.parquet"]
        return api

    def test_stale_round_rejected_before_datasets(self):
        api = self.api()
        with self.assertRaisesRegex(ValueError, "no longer the current round"):
            generation.checked_round(api, 1398)
        api.list_datasets.assert_not_called()
        api.list_rounds.assert_not_called()

    def test_round_target_mismatch_or_duplicate_metadata_is_rejected(self):
        for duplicate in (False, True):
            api = self.api()
            if duplicate:
                api.list_rounds.return_value *= 2
            else:
                api.list_rounds.return_value[0]["target"] = "target_jupiter_20"
            with self.assertRaisesRegex(ValueError, "uniquely match"):
                generation.checked_round(api, 1399)

    def test_datetime_round_metadata_and_closed_staking_are_valid(self):
        number, info = generation.checked_round(self.api(), 1399)
        self.assertEqual(number, 1399)
        self.assertIsInstance(info["closeTime"], str)
        self.assertEqual(info["dataDatestamp"], 20261113)

    def test_closed_submission_window_is_rejected(self):
        api = self.api()
        api.list_rounds.return_value[0]["closeTime"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "window has closed"):
            generation.checked_round(api, 1399)

    def run_fixture_generation(self, directory, rollover=False):
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        bundle = dict(self.bundle, schema_version=1, target="target_jupiter_60",
                      model_id="O2_medium_1200k_ensemble", members=[{"file": "member.txt"}])
        (source / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
        (source / "member.txt").write_text("native model fixture", encoding="utf-8")
        args = argparse.Namespace(bundle=source / "bundle.json", output=root / "handoff",
                                  round_number=1399, n_jobs=1)
        api = self.api()
        api.get_current_round.side_effect = [1399, 1400 if rollover else 1399]

        def fake_download(api, name, number, destination):
            if name == "features.json":
                destination.write_text(json.dumps(self.metadata), encoding="utf-8")
            else:
                self.frame.to_parquet(destination, index=False)
            return {"filename": name, "round": number,
                    "bytes": destination.stat().st_size, "sha256": generation.sha256(destination)}

        with patch.object(generation, "NumerAPI", return_value=api), \
             patch.object(generation, "download", side_effect=fake_download), \
             patch.object(generation, "predict_bundle", return_value=np.array([.1, .4, .4])) as predict:
            record = generation.generate(args)
        self.assertEqual(predict.call_count, 2)
        self.assertTrue(all(call.args[0] == args.output / "model/bundle.json"
                            for call in predict.call_args_list))
        return args, record

    def test_complete_generation_reloads_copied_bundle_and_checks_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            args, record = self.run_fixture_generation(directory)
            output = args.output
            self.assertEqual(record["status"], "complete")
            self.assertFalse(record["account_submission_performed"])
            self.assertEqual(record["live_date"], "2026-11-13")
            self.assertEqual(record["current_round_before"], record["current_round_after"])
            csv = pd.read_csv(output / record["csv_file"], float_precision="round_trip")
            generation.validate_submission(csv, self.frame, np.array([.1, .4, .4]))
            validation = json.loads((output / "validation.json").read_text("utf-8"))
            self.assertEqual(validation["status"], "passed")
            self.assertEqual(validation["csv_sha256"], generation.sha256(output / record["csv_file"]))
            self.assertTrue((output / "model/member.txt").is_file())

    def test_rollover_preserves_failed_record_without_publishing_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "round changed"):
                self.run_fixture_generation(directory, rollover=True)
            output = Path(directory) / "handoff"
            self.assertEqual(list(output.glob("*.csv")), [])
            record = json.loads((output / "live_generation.json").read_text("utf-8"))
            self.assertEqual(record["status"], "failed")
            self.assertFalse(record["account_submission_performed"])


if __name__ == "__main__":
    unittest.main()
