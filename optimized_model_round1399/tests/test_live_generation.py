"""Guard current-round generation, preserved artifacts and download file lifetime."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_xinchen_live as generation


class GenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "handoff"
        self.data = self.root / "data"
        self.baseline = self.root / "baseline"
        self.baseline.mkdir()
        self.paths = tuple(self.baseline / name for name in
                           ("B0_small_main.txt", "baseline_run.json", "baseline_protocol.json"))
        for path in self.paths:
            path.write_text("fixture", encoding="utf-8")
        self.features = [f"feature_{index}" for index in range(100)]
        self.live = pd.DataFrame({name: [0, 1, 4] for name in self.features})
        self.live.insert(0, "numerai_ticker", ["AAA US", "BBB US", "CCC US"])
        self.booster = MagicMock()
        self.booster.predict.return_value = np.array([.1, .4, .4])
        now = datetime.now(timezone.utc)
        self.api = MagicMock(tournament_id=8)
        self.api.get_current_round.side_effect = [1394, 1394]
        self.api.list_rounds.return_value = [{
            "number": 1394, "tournament": 8, "target": "target_jupiter_60",
            "dataDatestamp": 20261106, "openTime": now - timedelta(hours=1),
            "closeStakingTime": now + timedelta(hours=1), "closeTime": now + timedelta(days=1),
        }]
        self.api.list_datasets.return_value = ["v6.0/features.json", "v6.0/live.parquet"]
        self.args = argparse.Namespace(baseline=self.baseline, output=self.output,
                                       data_dir=self.data, round_number=1394, n_jobs=1)

    def fake_download(self, api, name, round_number, destination):
        if name == "features.json":
            destination.write_text(json.dumps({"feature_sets": {
                "small": self.features + ["feature_country"]}}), encoding="utf-8")
        else:
            self.live.to_parquet(destination, index=False)
        return {"filename": name, "round": round_number,
                "sha256": generation.sha256(destination), "bytes": destination.stat().st_size}

    def generate(self):
        with patch.object(generation, "NumerAPI", return_value=self.api), \
             patch.object(generation, "verify_baseline", return_value=(self.booster, self.features, self.paths)), \
             patch.object(generation, "download", side_effect=self.fake_download):
            generation.generate(self.args)

    def test_stale_requested_round_stops_before_baseline_or_files(self):
        self.args.round_number = 1393
        with patch.object(generation, "NumerAPI", return_value=self.api), \
             patch.object(generation, "verify_baseline") as verify, \
             patch.object(generation, "download") as download:
            with self.assertRaisesRegex(ValueError, "no longer the current round"):
                generation.generate(self.args)
        verify.assert_not_called()
        download.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.data.exists())

    def test_existing_output_is_preserved_before_download_or_data_creation(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_bytes(b"previous submission")
        with self.assertRaisesRegex(FileExistsError, "Preserving existing files"):
            self.generate()
        self.assertEqual(sentinel.read_bytes(), b"previous submission")
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        self.assertFalse(self.data.exists())

    def test_existing_data_is_preserved_before_any_output_creation(self):
        self.data.mkdir()
        sentinel = self.data / "keep.txt"
        sentinel.write_bytes(b"archived live data")
        with self.assertRaisesRegex(FileExistsError, "Preserving existing files"):
            self.generate()
        self.assertEqual(sentinel.read_bytes(), b"archived live data")
        self.assertFalse(self.output.exists())

    def test_round_change_does_not_publish_a_final_csv(self):
        self.api.get_current_round.side_effect = [1394, 1395]
        with self.assertRaisesRegex(ValueError, "round changed"):
            self.generate()
        self.assertEqual(list(self.output.glob("*.csv")), [])
        record = json.loads((self.output / "live_generation.json").read_text("utf-8"))
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["account_submission_performed"])
        self.assertFalse(record["training_performed"])

    def test_datetime_metadata_and_average_rank_generate_reusable_handoff(self):
        self.generate()
        csv = self.output / "XinchenGeng_baseline_live_round1394.csv"
        frame = pd.read_csv(csv, float_precision="round_trip")
        self.assertEqual(frame.columns.tolist(), ["numerai_ticker", "prediction"])
        self.assertEqual(frame.numerai_ticker.tolist(), self.live.numerai_ticker.tolist())
        np.testing.assert_array_equal(frame.prediction, [1 / 3, 2.5 / 3, 2.5 / 3])
        record = json.loads((self.output / "live_generation.json").read_text("utf-8"))
        self.assertEqual(record["status"], "complete")
        self.assertIsInstance(record["round_info"]["openTime"], str)
        self.assertEqual(record["round_info"]["dataDatestamp"], 20261106)
        self.assertEqual(record["current_round_before"], record["current_round_after"])
        self.booster.predict.assert_called_once()
        prediction_input = self.booster.predict.call_args.args[0]
        self.assertEqual(prediction_input.columns.tolist(), self.features)
        self.assertTrue(self.booster.predict.call_args.kwargs["validate_features"])

    def test_closed_historical_generation_pins_requested_data_and_records_actual_current_round(self):
        self.args.allow_closed_round = True
        self.api.get_current_round.side_effect = [1395, 1395]
        self.api.list_rounds.return_value[0]["closeTime"] = datetime.now(timezone.utc) - timedelta(days=1)
        with patch.object(generation, "NumerAPI", return_value=self.api), \
             patch.object(generation, "verify_baseline", return_value=(self.booster, self.features, self.paths)), \
             patch.object(generation, "download", side_effect=self.fake_download) as download:
            generation.generate(self.args)
        self.api.list_rounds.assert_called_once_with(number=1394)
        self.api.list_datasets.assert_called_once_with(round_num=1394)
        self.assertEqual([(call.args[1], call.args[2]) for call in download.call_args_list],
                         [("features.json", 1394), ("live.parquet", 1394)])
        record = json.loads((self.output / "live_generation.json").read_text("utf-8"))
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["round"], 1394)
        self.assertEqual(record["current_round_before"], 1395)
        self.assertEqual(record["current_round_after"], 1395)
        self.assertEqual(record["generation_mode"], "historical_closed_round")
        self.assertIs(record["is_current_round_at_generation"], False)
        self.assertIs(record["submission_window_open_at_generation"], False)
        self.assertFalse(record["training_performed"])
        self.assertFalse(record["account_submission_performed"])
        self.assertTrue((self.output / "XinchenGeng_baseline_live_round1394.csv").is_file())
        self.assertFalse((self.output / "XinchenGeng_baseline_live_round1395.csv").exists())

    def test_closed_round_opt_in_requires_explicit_past_round(self):
        self.args.allow_closed_round = True
        for requested in (None, 1394, 1395):
            with self.subTest(requested=requested):
                self.args.round_number = requested
                self.api.get_current_round.side_effect = [1394, 1394]
                with patch.object(generation, "NumerAPI", return_value=self.api), \
                     patch.object(generation, "verify_baseline") as verify, \
                     patch.object(generation, "download") as download:
                    with self.assertRaises(ValueError):
                        generation.generate(self.args)
                verify.assert_not_called()
                download.assert_not_called()
                self.assertFalse(self.output.exists())
                self.assertFalse(self.data.exists())

    def test_closed_round_opt_in_rejects_past_round_with_open_submission_window(self):
        self.args.allow_closed_round = True
        self.api.get_current_round.side_effect = [1395, 1395]
        with patch.object(generation, "NumerAPI", return_value=self.api), \
             patch.object(generation, "verify_baseline", return_value=(self.booster, self.features, self.paths)), \
             patch.object(generation, "download") as download:
            with self.assertRaises(ValueError):
                generation.generate(self.args)
        download.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.data.exists())

    def test_closed_current_round_remains_rejected_by_default(self):
        self.api.list_rounds.return_value[0]["closeTime"] = datetime.now(timezone.utc) - timedelta(days=1)
        with patch.object(generation, "NumerAPI", return_value=self.api), \
             patch.object(generation, "verify_baseline", return_value=(self.booster, self.features, self.paths)), \
             patch.object(generation, "download") as download:
            with self.assertRaisesRegex(ValueError, "submission window has closed"):
                generation.generate(self.args)
        download.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.data.exists())

    def test_boolean_live_features_are_not_treated_as_numeric_bins(self):
        frame = self.live.copy()
        frame[self.features[0]] = [True, False, True]
        with self.assertRaisesRegex(ValueError, "Nonnumeric|Boolean"):
            generation.check_live(frame, self.features)


class BaselineIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features = [f"feature_{index}" for index in range(100)]
        self.model = self.root / "B0_small_main.txt"
        self.model.write_bytes(b"saved model fixture")
        self.protocol = {"model": "B0_small_main", "data_version": "v6.0",
                         "target": "target_jupiter_60", "features": self.features}
        self.run = {"status": "complete", "model": "B0_small_main", "data_version": "v6.0",
                    "target": "target_jupiter_60", "features": self.features,
                    "model_sha256": generation.sha256(self.model)}
        self.save()
        self.booster = MagicMock()
        self.booster.feature_name.return_value = self.features
        self.booster.num_trees.return_value = 300

    def save(self):
        protocol_path = self.root / "baseline_protocol.json"
        protocol_path.write_text(json.dumps(self.protocol), encoding="utf-8")
        self.run["protocol_sha256"] = generation.sha256(protocol_path)
        (self.root / "baseline_run.json").write_text(json.dumps(self.run), encoding="utf-8")

    def test_changed_model_file_rejected_before_loading(self):
        self.model.write_bytes(b"different saved model")
        with patch.object(generation.lgb, "Booster") as load_model:
            with self.assertRaisesRegex(ValueError, "model or protocol differs"):
                generation.verify_baseline(self.root, 1)
        load_model.assert_not_called()

    def test_protocol_feature_order_mismatch_rejected_even_with_matching_hash(self):
        self.protocol["features"] = list(reversed(self.features))
        self.save()
        with patch.object(generation.lgb, "Booster", return_value=self.booster):
            with self.assertRaisesRegex(ValueError, "feature order differs"):
                generation.verify_baseline(self.root, 1)

    def test_changed_target_or_version_is_rejected(self):
        for field, value in [("target", "target_jupiter_20"), ("data_version", "v5.0")]:
            with self.subTest(field=field):
                original = self.protocol[field]
                self.protocol[field] = value
                self.save()
                with self.assertRaisesRegex(ValueError, "target|v6.0"):
                    generation.verify_baseline(self.root, 1)
                self.protocol[field] = original


class DownloadLifetimeTests(unittest.TestCase):
    def test_parquet_reader_is_closed_before_windows_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.parquet"
            pd.DataFrame({"numerai_ticker": ["a", "b"]}).to_parquet(source, index=False)
            payload = source.read_bytes()
            destination = root / "live.parquet"
            api = MagicMock(tournament_id=8)
            api.raw_query.return_value = {"data": {"dataset":
                "https://example.invalid/20261106/v6.0/live.parquet?signature=never-record-this"}}
            response = MagicMock()
            response.__enter__.return_value = response
            response.headers = {"Content-Length": str(len(payload))}
            response.iter_content.return_value = [payload]
            session = MagicMock()
            session.__enter__.return_value = session
            session.get.return_value = response
            reader_closed = []
            actual_parquet = pq.ParquetFile
            actual_replace = Path.replace

            class TrackedParquet:
                def __init__(self, path):
                    self.parquet = actual_parquet(path)

                def __enter__(self):
                    return self.parquet

                def __exit__(self, *args):
                    self.parquet.close()
                    reader_closed.append(True)

            def rename_only_closed(path, target):
                self.assertTrue(reader_closed, "Parquet handle must be closed before rename on Windows")
                return actual_replace(path, target)

            with patch.object(generation.requests, "Session", return_value=session), \
                 patch.object(generation.pq, "ParquetFile", TrackedParquet), \
                 patch.object(Path, "replace", rename_only_closed):
                record = generation.download(api, "live.parquet", 1394, destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(record["rows"], 2)
            self.assertNotIn("signature", record["url"])
            self.assertEqual(api.raw_query.call_args.args[1]["round"], 1394)


if __name__ == "__main__":
    unittest.main()
