"""Tests of source eligibility and horizon-aware chronological research blocks."""
import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper_protocol import TARGET, _eligible_batch, load_research_pool, make_fold_plan, split_fold


def frame(dates):
    return pd.DataFrame({"numerai_ticker": [f"S{i} US" for i in range(len(dates))],
                         "date": pd.to_datetime(dates), "feature_a": 2,
                         TARGET: .5})


class ResearchProtocolTests(unittest.TestCase):
    def test_raw_pool_restores_mature_dates_and_preserves_pilot_files(self):
        with tempfile.TemporaryDirectory(prefix="numerai_protocol_test_") as directory:
            root = Path(directory)
            raw = root / "data/raw/v6.0"
            processed = root / "data/processed/v6.0"
            raw.mkdir(parents=True)
            processed.mkdir(parents=True)
            sentinel = processed / "train.parquet"
            sentinel.write_bytes(b"original pilot file must not change")
            metadata = {"feature_sets": {"small": ["feature_a", "feature_country"],
                                          "medium": ["feature_a", "feature_country"], "signals": []}}
            (raw / "features.json").write_text(json.dumps(metadata), encoding="utf-8")
            train = frame(["2018-12-21", "2018-12-28"])
            train["data_type"] = "train"
            train["feature_a"] = pd.Series([2, None], dtype="Int8")
            train.to_parquet(raw / "train.parquet", index=False)
            validation = frame(["2019-01-04", "2026-07-24", "2026-10-16", "2026-07-17"])
            validation["data_type"] = ["validation", "validation", "test", "validation"]
            validation["feature_a"] = validation.feature_a.astype("Int8")
            validation.loc[3, TARGET] = np.nan
            validation.to_parquet(raw / "validation.parquet", index=False)
            pool, sets, provenance = load_research_pool(root)
            self.assertEqual(len(pool), 4)
            self.assertTrue((pool.date == pd.Timestamp("2019-01-04")).any())
            self.assertFalse((pool.date == pd.Timestamp("2026-10-16")).any())
            self.assertEqual(sets["small"], ["feature_a"])
            self.assertEqual(provenance["source_reports"]["validation.parquet"]["excluded_null_target"], 1)
            self.assertEqual(sentinel.read_bytes(), b"original pilot file must not change")
            cached, _, cached_provenance = load_research_pool(root)
            pd.testing.assert_frame_equal(pool, cached)
            self.assertEqual(provenance["cache_sha256"], cached_provenance["cache_sha256"])

    def test_cutoff_uses_actual_available_scoring_date(self):
        pool = frame(["2018-12-28", "2020-01-03", "2020-01-10", "2023-01-06"])
        folds = {x["id"]: x for x in make_fold_plan(pool)}
        fold = folds["cv_2020"]
        self.assertEqual(fold["eval_start"], "2020-01-03")
        expected = pd.Timestamp("2020-01-03") - pd.offsets.BDay(65)
        self.assertEqual(pd.Timestamp(fold["train_cutoff_exclusive"]), expected)
        self.assertNotEqual(expected, pd.Timestamp("2020-01-01") - pd.offsets.BDay(65))

    def test_strict_label_maturity_boundary_and_no_future_data(self):
        scoring = pd.Timestamp("2020-01-03")
        cutoff = scoring - pd.offsets.BDay(65)
        pool = frame([cutoff - pd.offsets.BDay(1), cutoff, scoring, "2020-06-01", "2025-01-03"])
        fold = {"id": "cv_2020", "phase": "cv", "eval_start": "2020-01-03",
                "eval_end": "2020-06-01", "train_cutoff_exclusive": str(cutoff.date()),
                "embargo_weekdays": 65}
        train, evaluate = split_fold(pool, fold)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(evaluate), 2)
        self.assertLess(train.date.max() + pd.offsets.BDay(65), evaluate.date.min())
        self.assertFalse((train.date >= cutoff).any())
        self.assertFalse((evaluate.date.dt.year == 2025).any())

    def test_rejects_a_weakened_or_inconsistent_cutoff(self):
        pool = frame(["2018-01-05", "2020-01-03"])
        fold = make_fold_plan(pool)[0]
        for change in ({"embargo_weekdays": 60}, {"train_cutoff_exclusive": "2019-12-31"}):
            with self.assertRaises(ValueError):
                split_fold(pool, {**fold, **change})

    def test_source_specific_data_type_and_target_maturity(self):
        source = frame(["2019-01-04", "2019-01-11", "2019-01-18", "2019-01-25"])
        source["data_type"] = ["validation", "test", "unknown", "validation"]
        source.loc[3, TARGET] = np.nan
        result = _eligible_batch(source, "validation", ["feature_a"], set())
        self.assertEqual(result.numerai_ticker.tolist(), ["S0 US"])

    def test_missing_data_type_cannot_default_to_include_all(self):
        with self.assertRaises(ValueError):
            _eligible_batch(frame(["2019-01-04"]), "validation", ["feature_a"], set())

    def test_invalid_feature_is_rejected_and_native_nan_is_retained(self):
        source = frame(["2019-01-04", "2019-01-11"])
        source["data_type"] = "validation"
        source["feature_a"] = [2.0, np.nan]
        result = _eligible_batch(source, "validation", ["feature_a"], set())
        self.assertEqual(len(result), 2)
        self.assertTrue(pd.isna(result.iloc[1].feature_a))
        for bad in (5.0, np.inf):
            source.loc[0, "feature_a"] = bad
            with self.assertRaises(ValueError):
                _eligible_batch(source, "validation", ["feature_a"], set())

    def test_observed_later_period_is_named_reused_development(self):
        pool = frame(["2018-12-28", "2019-04-05", "2022-09-30", "2022-10-07",
                      "2023-01-06", "2026-07-24"])
        plan = make_fold_plan(pool)
        self.assertEqual({f["phase"] for f in plan if f["id"].startswith("dev_")},
                         {"reused_development"})
        self.assertEqual(next(f for f in plan if f["id"] == "cv_2022")["eval_end"], "2022-09-30")


if __name__ == "__main__":
    unittest.main()
