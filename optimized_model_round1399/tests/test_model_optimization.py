"""Guardrails around selection and delayed-label model choice."""
import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_model_optimization import MODEL_IDS, select_candidate


class SelectionTests(unittest.TestCase):
    def scores(self):
        return pd.DataFrame([
            {"phase": "cv", "fold": f"cv_{year}", "date": pd.Timestamp(f"{year}-06-01"),
             "model": model, "NCORR": .01 + i * .01, "n_rows": 100}
            for year in range(2019, 2023) for i, model in enumerate(MODEL_IDS)])

    def test_complete_cv_selects_without_development(self):
        chosen = select_candidate(self.scores())
        self.assertEqual(chosen["selected_model"], MODEL_IDS[-1])
        self.assertFalse(chosen["development_used_for_selection"])

    def test_refuses_development_scores(self):
        scores = self.scores()
        scores.loc[0, "phase"] = "reused_development"
        with self.assertRaises(ValueError):
            select_candidate(scores)

    def test_refuses_missing_dates_or_changed_population(self):
        with self.assertRaises(ValueError):
            select_candidate(self.scores().iloc[1:])
        scores = self.scores()
        scores.loc[0, "n_rows"] = 99
        with self.assertRaises(ValueError):
            select_candidate(scores)

    def test_instability_penalty_changes_choice(self):
        scores = self.scores()
        mask = scores.model.eq(MODEL_IDS[-1])
        scores.loc[mask, "NCORR"] = [-.2, .2, .2, .2]
        self.assertEqual(select_candidate(scores)["selected_model"], MODEL_IDS[-2])


if __name__ == "__main__":
    unittest.main()
