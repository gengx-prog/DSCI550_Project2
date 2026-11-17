"""Numerical and identity failure cases for the independent live handoff audit."""
import sys
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_xinchen_live import compare_predictions, validate_bundle, validate_features, validate_tickers


class LiveSubmissionTests(unittest.TestCase):
    def test_average_ties_and_permuted_ticker_alignment(self):
        frame = pd.DataFrame({"numerai_ticker": ["d", "b", "a", "c"],
                              "prediction": [1.0, .625, .25, .625]})
        report = compare_predictions(frame, ["a", "b", "c", "d"], np.array([1, 3, 3, 8]))
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["maximum_absolute_error"], 0)

    def test_wrong_tie_handling_is_rejected(self):
        frame = pd.DataFrame({"numerai_ticker": ["a", "b", "c", "d"],
                              "prediction": [.25, .5, .75, 1.]})
        with self.assertRaisesRegex(ValueError, "independent"):
            compare_predictions(frame, ["a", "b", "c", "d"], np.array([1, 3, 3, 8]))

    def test_constant_csv_and_model_predictions_are_rejected(self):
        for values, raw in [([.5, .5], [1, 2]), ([.5, 1.], [1, 1])]:
            frame = pd.DataFrame({"numerai_ticker": ["a", "b"], "prediction": values})
            with self.subTest(values=values, raw=raw), self.assertRaisesRegex(ValueError, "constant"):
                compare_predictions(frame, ["a", "b"], np.array(raw))

    def test_changed_ticker_and_duplicate_ticker_are_rejected(self):
        for tickers in [["a", "x"], ["a", "a"]]:
            frame = pd.DataFrame({"numerai_ticker": tickers, "prediction": [.5, 1.]})
            with self.subTest(tickers=tickers), self.assertRaises(ValueError):
                compare_predictions(frame, ["a", "b"], np.array([1, 2]))

    def test_null_empty_numeric_and_whitespace_ids_are_rejected(self):
        for values in [["a", None], ["a", ""], ["a", " b"], ["a", "b "], ["a", 2]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_tickers(pd.Series(values), "fixture")

    def test_nonfinite_and_out_of_bounds_predictions_are_rejected(self):
        for value in [np.nan, np.inf, -np.inf, -.1, 1.1]:
            frame = pd.DataFrame({"numerai_ticker": ["a", "b"], "prediction": [value, 1.]})
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare_predictions(frame, ["a", "b"], np.array([1, 2]))

    def test_extra_or_reordered_columns_are_rejected(self):
        frame = pd.DataFrame({"prediction": [.5, 1.], "numerai_ticker": ["a", "b"]})
        with self.assertRaisesRegex(ValueError, "columns"):
            compare_predictions(frame, ["a", "b"], np.array([1, 2]))

    def test_native_feature_nan_is_allowed_but_infinity_is_rejected(self):
        frame = pd.DataFrame({"feature_a": [0., np.nan], "feature_b": [1, 4]})
        self.assertEqual(validate_features(frame, ["feature_a", "feature_b"]), 1)
        frame.loc[0, "feature_a"] = np.inf
        with self.assertRaisesRegex(ValueError, "Infinite"):
            validate_features(frame, ["feature_a", "feature_b"])

    def test_strings_boolean_country_and_missing_features_are_rejected(self):
        fixtures = [(pd.DataFrame({"feature_a": ["1", "2"]}), ["feature_a"]),
                    (pd.DataFrame({"feature_a": [True, False]}), ["feature_a"]),
                    (pd.DataFrame({"feature_country": [1, 2]}), ["feature_country"]),
                    (pd.DataFrame({"feature_a": [1, 2]}), ["feature_b"])]
        for frame, features in fixtures:
            with self.subTest(features=features), self.assertRaises(ValueError):
                validate_features(frame, features)


class LiveBundleTests(unittest.TestCase):
    """Exercise the public audit with a tiny real saved LightGBM model."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.live = self.root / "live.parquet"
        self.features = [f"feature_{index}" for index in range(100)]
        rng = np.random.default_rng(27)
        train = pd.DataFrame(rng.integers(0, 5, size=(32, 100)), columns=self.features)
        target = train.iloc[:, 0] + train.iloc[:, 1] * .4
        model = lgb.train({"objective": "regression", "verbosity": -1, "num_threads": 1, "learning_rate": .01,
                           "num_leaves": 4, "min_data_in_leaf": 1, "seed": 7},
                          lgb.Dataset(train, label=target), num_boost_round=300)
        model.save_model(str(self.bundle / "B0_small_main.txt"))
        live = train.iloc[:8].copy()
        live.insert(0, "numerai_ticker", [f"T{index}" for index in range(8)])
        live.to_parquet(self.live, index=False)
        self.write_json("baseline_protocol.json", {"model": "B0_small_main", "data_version": "v6.0",
                                                   "target": "target_jupiter_60", "features": self.features})
        self.write_json("baseline_run.json", {
            "status": "complete", "model": "B0_small_main", "data_version": "v6.0",
            "target": "target_jupiter_60",
            "features": self.features, "model_sha256": self.digest(self.bundle / "B0_small_main.txt"),
            "protocol_sha256": self.digest(self.bundle / "baseline_protocol.json")})
        self.write_json("features.json", {"feature_sets": {"small": self.features + ["feature_country"]}})
        files = []
        for path in [self.live, self.bundle / "features.json"]:
            files.append({"filename": path.name, "dataset": f"v6.0/{path.name}",
                          "sha256": self.digest(path), "bytes": path.stat().st_size,
                          "url": f"https://example.invalid/20261106/v6.0/{path.name}", "round": 1394})
        self.write_json("live_data_manifest.json", {"status": "complete", "data_version": "v6.0",
                                                    "round": 1394, "files": files, "tournament": 8,
                                                    "round_info": {"number": 1394, "tournament": 8,
                                                                   "target": "target_jupiter_60"}})
        self.csv = self.bundle / "XinchenGeng_baseline_live_round1394.csv"
        prediction = pd.Series(model.predict(live[self.features], num_threads=1)).rank(pct=True)
        pd.DataFrame({"numerai_ticker": live.numerai_ticker, "prediction": prediction}).to_csv(self.csv, index=False)
        generation = {"status": "complete", "round": 1394, "data_version": "v6.0",
                      "features": self.features, "live_rows": len(live), "csv_file": self.csv.name,
                      "csv_sha256": self.digest(self.csv), "columns": ["numerai_ticker", "prediction"],
                      "prediction_transform": "pandas.Series.rank(method='average', pct=True)",
                      "current_round_before": 1394, "current_round_after": 1394,
                      "round_info": {"number": 1394, "tournament": 8, "target": "target_jupiter_60"}}
        for field, path in {"model_sha256": self.bundle / "B0_small_main.txt",
                            "protocol_sha256": self.bundle / "baseline_protocol.json",
                            "baseline_run_sha256": self.bundle / "baseline_run.json",
                            "feature_metadata_sha256": self.bundle / "features.json",
                            "live_parquet_sha256": self.live,
                            "download_manifest_sha256": self.bundle / "live_data_manifest.json"}.items():
            generation[field] = self.digest(path)
        self.write_json("live_generation.json", generation)

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_json(self, name, value):
        (self.bundle / name).write_text(json.dumps(value), encoding="utf-8")

    def make_closed_historical_bundle(self):
        started = datetime.now(timezone.utc)
        generation = json.loads((self.bundle / "live_generation.json").read_text())
        generation.update(generation_mode="historical_closed_round",
                          is_current_round_at_generation=False,
                          submission_window_open_at_generation=False,
                          current_round_before=1395, current_round_after=1395,
                          started_at_utc=started.isoformat())
        generation["round_info"]["closeTime"] = (started - timedelta(days=1)).isoformat()
        manifest = json.loads((self.bundle / "live_data_manifest.json").read_text())
        manifest["round_info"] = generation["round_info"]
        self.write_json("live_data_manifest.json", manifest)
        generation["download_manifest_sha256"] = self.digest(self.bundle / "live_data_manifest.json")
        self.write_json("live_generation.json", generation)
        return generation

    def test_complete_saved_model_bundle_validates_offline(self):
        report = validate_bundle(self.bundle, self.live)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["full_model_and_rank_check"]["rows"], 8)
        self.assertFalse(report["current_round_queried"])
        self.assertFalse((self.bundle / "validation.json").exists())

    def test_changed_source_live_bytes_are_rejected(self):
        frame = pd.read_parquet(self.live)
        frame.loc[0, "feature_0"] = 4 - frame.loc[0, "feature_0"]
        frame.to_parquet(self.live, index=False)
        with self.assertRaisesRegex(ValueError, "live_parquet_sha256"):
            validate_bundle(self.bundle, self.live)

    def test_round_rollover_is_rejected(self):
        generation = json.loads((self.bundle / "live_generation.json").read_text())
        generation["current_round_after"] = 1395
        self.write_json("live_generation.json", generation)
        with self.assertRaisesRegex(ValueError, "Round changed"):
            validate_bundle(self.bundle, self.live)

    def test_closed_historical_bundle_requires_explicit_audit_opt_in(self):
        self.make_closed_historical_bundle()
        with self.assertRaises(ValueError):
            validate_bundle(self.bundle, self.live)
        report = validate_bundle(self.bundle, self.live, allow_closed_round=True)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["round"], 1394)
        self.assertEqual(report["generation_mode"], "historical_closed_round")
        self.assertFalse(report["current_round_queried"])
        self.assertEqual(report["full_model_and_rank_check"]["rows"], 8)

    def test_historical_audit_rejects_submission_window_open_at_generation(self):
        generation = self.make_closed_historical_bundle()
        started = datetime.fromisoformat(generation["started_at_utc"])
        generation["round_info"]["closeTime"] = (started + timedelta(hours=1)).isoformat()
        manifest = json.loads((self.bundle / "live_data_manifest.json").read_text())
        manifest["round_info"] = generation["round_info"]
        self.write_json("live_data_manifest.json", manifest)
        generation["download_manifest_sha256"] = self.digest(self.bundle / "live_data_manifest.json")
        self.write_json("live_generation.json", generation)
        with self.assertRaises(ValueError):
            validate_bundle(self.bundle, self.live, allow_closed_round=True)

    def test_historical_audit_rejects_current_round_and_rollover_claims(self):
        generation = self.make_closed_historical_bundle()
        for before, after in ((1394, 1394), (1395, 1396)):
            with self.subTest(before=before, after=after):
                generation.update(current_round_before=before, current_round_after=after)
                self.write_json("live_generation.json", generation)
                with self.assertRaises(ValueError):
                    validate_bundle(self.bundle, self.live, allow_closed_round=True)

    def test_current_round_api_check_still_rejects_closed_historical_round(self):
        self.make_closed_historical_bundle()
        with patch("numerapi.NumerAPI") as api:
            api.return_value.get_current_round.return_value = 1395
            with self.assertRaisesRegex(ValueError, "no longer current"):
                validate_bundle(self.bundle, self.live, check_current_round=True, allow_closed_round=True)

    def test_wrong_csv_is_rejected_even_with_matching_recorded_hash(self):
        frame = pd.read_csv(self.csv)
        frame.prediction = frame.prediction.iloc[::-1].to_numpy()
        frame.to_csv(self.csv, index=False)
        generation = json.loads((self.bundle / "live_generation.json").read_text())
        generation["csv_sha256"] = self.digest(self.csv)
        self.write_json("live_generation.json", generation)
        with self.assertRaisesRegex(ValueError, "independent saved-model"):
            validate_bundle(self.bundle, self.live)

    def test_inconsistent_download_tournament_is_rejected(self):
        manifest = json.loads((self.bundle / "live_data_manifest.json").read_text())
        manifest["tournament"] = 9
        self.write_json("live_data_manifest.json", manifest)
        generation = json.loads((self.bundle / "live_generation.json").read_text())
        generation["download_manifest_sha256"] = self.digest(self.bundle / "live_data_manifest.json")
        self.write_json("live_generation.json", generation)
        with self.assertRaisesRegex(ValueError, "tournament differs"):
            validate_bundle(self.bundle, self.live)


if __name__ == "__main__":
    unittest.main()
