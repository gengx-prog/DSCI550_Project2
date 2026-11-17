"""Independent checks for chronological model selection and paired resampling."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper_statistics import CANDIDATES, _circular_block_indices, paired_block_bootstrap, select_candidate, summarize


def cv_frame():
    records = []
    # Unequal folds distinguish equal-date averages from equal-fold averages.
    for fold, n in enumerate([2, 3, 4, 5]):
        for i in range(n):
            date = pd.Timestamp(f"{2019 + fold}-04-01") + pd.Timedelta(weeks=i)
            for j, name in enumerate(CANDIDATES):
                score = 0.01 + j * .001 + fold * .002 + i * .0001
                records.append(dict(phase="cv", fold=fold, date=date, model=name,
                                    NCORR=score, CORR=score / 2, n_rows=100))
    return pd.DataFrame(records)


def development_frame(n=40):
    records = []
    for i, date in enumerate(pd.date_range("2023-01-06", periods=n, freq="W-FRI")):
        for name, delta in [("P1_medium_200k", 0), ("P2_medium_600k", .002 + np.sin(i) * .004)]:
            records.append(dict(phase="reused_development", fold=str(date.year), date=date, model=name,
                                NCORR=.01 + delta, CORR=.005 + delta, n_rows=6000 + i))
    return pd.DataFrame(records)


class PaperStatisticsTests(unittest.TestCase):
    def test_summary_matches_independent_statistics(self):
        frame = cv_frame()
        result = summarize(frame)
        row = result[(result.model == CANDIDATES[0]) & (result.metric == "NCORR")].iloc[0]
        scores = frame[frame.model == CANDIDATES[0]].sort_values("date").NCORR.to_numpy()
        cumulative = np.r_[0, np.cumsum(scores)]
        expected_drawdown = np.max(np.maximum.accumulate(cumulative) - cumulative)
        self.assertAlmostEqual(row["mean"], np.mean(scores), places=14)
        self.assertAlmostEqual(row["std_ddof0"], np.std(scores), places=14)
        self.assertAlmostEqual(row["max_drawdown"], expected_drawdown, places=14)
        folds = frame[frame.model == CANDIDATES[0]].groupby("fold").NCORR.mean().to_numpy()
        self.assertAlmostEqual(row.fold_mean_std_ddof0, np.std(folds), places=14)
        self.assertEqual(row.n_folds, 4)

    def test_selection_uses_equal_date_mean_and_fixed_fold_penalty(self):
        frame = cv_frame()
        result = select_candidate(frame)
        self.assertEqual(result["selected_model"], CANDIDATES[-1])
        rows = frame[frame.model == CANDIDATES[0]]
        expected = rows.NCORR.mean() - .5 * rows.groupby("fold").NCORR.mean().std(ddof=0)
        self.assertAlmostEqual(result["candidates"][CANDIDATES[0]]["selection_score"], expected, places=14)
        self.assertNotAlmostEqual(rows.NCORR.mean(), rows.groupby("fold").NCORR.mean().mean(), places=8)

    def test_selection_rejects_development_and_incomplete_candidate_folds(self):
        frame = cv_frame()
        with self.assertRaises(ValueError):
            select_candidate(frame.assign(phase="development"))
        with self.assertRaises(ValueError):
            select_candidate(frame.iloc[1:])
        with self.assertRaises(ValueError):
            select_candidate(frame[frame.fold != 0])

    def test_selection_ties_break_by_identifier(self):
        result = select_candidate(cv_frame().assign(NCORR=.01, CORR=.01))
        self.assertEqual(result["selected_model"], sorted(CANDIDATES)[0])

    def test_circular_indices_match_independent_known_rng_and_contiguity(self):
        actual = _circular_block_indices(11, 4, 3, np.random.default_rng(91))
        starts = np.random.default_rng(91).integers(0, 11, size=(3, 3))
        expected = np.array([[(int(start) + j) % 11 for start in row for j in range(4)][:11]
                             for row in starts])
        np.testing.assert_array_equal(actual, expected)
        for start in [0, 4, 8]:
            np.testing.assert_array_equal(np.diff(actual[:, start:min(start + 4, 11)], axis=1) % 11,
                                          np.ones((3, min(4, 11 - start) - 1), dtype=int))

    def test_bootstrap_is_paired_deterministic_and_independently_recomputed(self):
        frame = development_frame()
        actual = paired_block_bootstrap(frame.sample(frac=1, random_state=1), "P2_medium_600k", reps=100, seed=23)
        repeated = paired_block_bootstrap(frame, "P2_medium_600k", reps=100, seed=23)
        self.assertEqual(actual, repeated)
        pivot = frame.pivot(index="date", columns="model", values="NCORR").sort_index()
        delta = (pivot.P2_medium_600k - pivot.P1_medium_200k).to_numpy()
        rng = np.random.default_rng(23)
        starts = rng.integers(0, 40, size=(100, 4))
        means = []
        for row in starts:
            indices = [(int(s) + j) % 40 for s in row for j in range(13)][:40]
            means.append(np.mean(delta[indices]))
        lower, upper = np.percentile(means, [2.5, 97.5])
        self.assertAlmostEqual(actual["observed_mean_NCORR_delta"], np.mean(delta), places=14)
        self.assertAlmostEqual(actual["block_bootstrap"]["13"]["ci95_low"], lower, places=14)
        self.assertAlmostEqual(actual["block_bootstrap"]["13"]["ci95_high"], upper, places=14)

    def test_bootstrap_same_baseline_yields_zero_interval(self):
        result = paired_block_bootstrap(development_frame(), "P1_medium_200k", reps=10)
        self.assertEqual(result["observed_mean_NCORR_delta"], 0)
        for record in result["block_bootstrap"].values():
            self.assertEqual(record["ci95_low"], 0)
            self.assertEqual(record["ci95_high"], 0)

    def test_bootstrap_rejects_cv_and_mismatched_keys_or_populations(self):
        frame = development_frame()
        with self.assertRaises(ValueError):
            paired_block_bootstrap(frame.assign(phase="cv"), "P2_medium_600k")
        with self.assertRaises(ValueError):
            paired_block_bootstrap(frame.iloc[1:], "P2_medium_600k")
        frame.loc[0, "n_rows"] += 1
        with self.assertRaises(ValueError):
            paired_block_bootstrap(frame, "P2_medium_600k")


if __name__ == "__main__":
    unittest.main()
