"""Regression checks for chronology, rank ensembles and portable model hashes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import optimized_models as om


def frames():
    rng = np.random.default_rng(18)
    train = pd.DataFrame({"numerai_ticker": [f"ticker_{i}" for i in range(80)],
        "date": pd.Timestamp("2017-01-06"), "feature_a": rng.normal(size=80),
        "feature_b": rng.normal(size=80)})
    train[om.TARGET] = (train.feature_a > 0).astype(float)
    evaluation = train.drop(columns=[om.TARGET]).iloc[:12].copy()
    evaluation["date"] = pd.to_datetime(["2020-01-03"] * 6 + ["2020-01-10"] * 6)
    evaluation.index = np.arange(12) * 3 + 11
    return train, evaluation


def tiny_specs(n_jobs=1):
    specs = ORIGINAL_SPECS(n_jobs)
    for index, spec in enumerate(specs.values()):
        spec["max_rows"] = [20, 40, 60, 60][index]
        spec["params"].update(n_estimators=5, min_child_samples=2, colsample_bytree=1.0,
                              subsample=1.0, subsample_freq=0)
    return specs


ORIGINAL_SPECS = om.candidate_specs


class OptimizedModelTests(unittest.TestCase):
    def test_fixed_matrix_and_fresh_parameter_objects(self):
        specs = om.candidate_specs(2)
        self.assertEqual(list(specs), om.MODEL_IDS)
        self.assertEqual([s["max_rows"] for s in specs.values()], [200000, 600000, 1200000, 1200000])
        self.assertEqual(specs[om.MODEL_IDS[2]]["seeds"], [7, 19, 43])
        self.assertEqual(specs[om.MODEL_IDS[3]]["params"]["reg_lambda"], 10)
        specs[om.MODEL_IDS[0]]["params"]["n_estimators"] = 1
        self.assertEqual(om.candidate_specs()[om.MODEL_IDS[0]]["params"]["n_estimators"], 300)

    def test_rank_mean_preserves_ties_date_groups_and_row_order(self):
        dates = ["2020-01-10", "2020-01-03", "2020-01-10", "2020-01-03", "2020-01-03"]
        values = [[4, 9], [5, 5], [1, 9], [5, 1], [1, 3]]
        expected = [0.875, (5 / 6 + 1) / 2, 0.625, (5 / 6 + 1 / 3) / 2, 0.5]
        np.testing.assert_allclose(om.rank_mean(dates, values), expected)
        order = [3, 1, 4, 0, 2]
        np.testing.assert_allclose(om.rank_mean(np.asarray(dates)[order], np.asarray(values)[order]),
                                   np.asarray(expected)[order])

    def test_rank_rejects_missing_or_nonfinite_members(self):
        with self.assertRaises(ValueError):
            om.rank_mean(["2020-01-03"], [[np.nan]])
        with self.assertRaises(ValueError):
            om.rank_mean([None], [[1]])

    def test_nested_samples_and_loaded_model_reproduce_predictions(self):
        train, evaluation = frames()
        features = {"small": ["feature_a"], "medium": ["feature_a", "feature_b"]}
        with tempfile.TemporaryDirectory() as folder, patch.object(om, "candidate_specs", tiny_specs):
            predictions, records = om.train_predict_candidates(train, evaluation, features, Path(folder), n_jobs=1)
            permutation = train.sample(n=len(train), random_state=7).reset_index(drop=True)
            for record in records:
                self.assertEqual(record["sample_keys_sha256"],
                                 om._sample_hash(permutation.iloc[:record["training_rows"]]))
                loaded = om.predict_bundle(Path(record["bundle_file"]), evaluation, n_jobs=1)
                np.testing.assert_allclose(loaded, predictions[record["model"]], rtol=0, atol=0)
            self.assertEqual(records[2]["sample_keys_sha256"], records[3]["sample_keys_sha256"])
            with self.assertRaises(FileExistsError):
                om.train_predict_candidates(train, evaluation, features, Path(folder), n_jobs=1)

    def test_evaluation_labels_are_not_read_and_hash_tampering_fails(self):
        train, evaluation = frames()
        # Deliberately invalid labels must not affect model fitting or inference.
        evaluation[om.TARGET] = "labels must not enter fitting"
        with tempfile.TemporaryDirectory() as folder, patch.object(om, "candidate_specs", tiny_specs):
            predictions, records = om.train_predict_candidates(train, evaluation,
                {"small": ["feature_a"]}, Path(folder), n_jobs=1, candidate_ids=[om.MODEL_IDS[0]])
            bundle_path = Path(records[0]["bundle_file"])
            actual = om.predict_bundle(bundle_path, evaluation.drop(columns=om.TARGET), n_jobs=1)
            np.testing.assert_array_equal(actual, predictions[om.MODEL_IDS[0]])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            model_path = bundle_path.parent / bundle["members"][0]["file"]
            with model_path.open("ab") as stream:
                stream.write(b"\n# tampered\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                om.predict_bundle(bundle_path, evaluation, n_jobs=1)

    def test_future_training_duplicates_and_infinity_are_rejected(self):
        train, evaluation = frames()
        features = {"small": ["feature_a"]}
        with tempfile.TemporaryDirectory() as folder:
            for corrupt in ["future", "duplicate", "infinity"]:
                bad = train.copy()
                if corrupt == "future":
                    bad.loc[0, "date"] = pd.Timestamp("2020-01-03")
                elif corrupt == "duplicate":
                    bad.loc[1, "numerai_ticker"] = bad.loc[0, "numerai_ticker"]
                else:
                    bad.loc[0, "feature_a"] = np.inf
                with self.assertRaises(ValueError):
                    om.train_predict_candidates(bad, evaluation, features,
                        Path(folder), n_jobs=1, candidate_ids=[om.MODEL_IDS[0]])


if __name__ == "__main__":
    unittest.main()
