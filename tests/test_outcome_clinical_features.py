"""Outcome routing checks using synthetic inputs only."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_models/clinical_genomic_models/scripts"))
import modeling_contract as contract
import build_nested_information_models as nested
import compare_model_families as families
import build_compact_model as compact


class OutcomeClinicalFeaturesTests(unittest.TestCase):
    def test_table_s2_panels_and_distinguishing_predictors(self):
        death = contract.selected_clinical_features("death_30d")
        invasive = contract.selected_clinical_features("metastatic")
        self.assertEqual(len(death), 16)
        self.assertEqual(len(invasive), 17)
        self.assertEqual(len(set(death)), 16)
        self.assertEqual(len(set(invasive)), 17)
        self.assertIn("SOFA评分", death)
        self.assertIn("CRP-1", death)
        self.assertNotIn("年龄", death)
        self.assertIn("年龄", invasive)
        self.assertIn("器官移植状态（1是0否）", invasive)
        self.assertNotIn("CRP-1", invasive)

    def test_every_nested_tier_uses_its_outcome_panel(self):
        for outcome in contract.OUTCOME_LABELS:
            clinical = contract.selected_clinical_features(outcome)
            tiers = nested.tiers_for_panel(["synthetic_gene"], outcome)
            self.assertEqual(tiers, contract.outcome_tiers(outcome, {outcome: ["synthetic_gene"]}))
            for features in tiers.values():
                self.assertEqual(features[:len(clinical)], clinical)
            self.assertEqual(tiers["clinical_ast_markers"], nested.marker_model_features(outcome))
            table = contract.feature_panel_table({outcome: tiers}, {outcome: ["synthetic_gene"]})
            self.assertEqual(set(table.loc[table.feature.isin(clinical), "feature_block"]), {"clinical_baseline"})
        invasive_tiers = nested.tiers_for_panel(["synthetic_gene"], "metastatic")
        self.assertEqual(families.tier_features(["synthetic_gene"]), invasive_tiers)
        self.assertEqual(compact.full_model_features(["synthetic_gene"]), invasive_tiers["clinical_ast_markers_gwas"])

    def test_panels_cannot_be_mutated_through_returned_tiers(self):
        tiers = nested.tiers_for_panel([], "death_30d")
        tiers["clinical"].clear()
        self.assertEqual(len(contract.selected_clinical_features("death_30d")), 16)
        with self.assertRaises(ValueError):
            contract.selected_clinical_features("unknown")

    def test_validation_topk_uses_outcome_specific_adjustment(self):
        for outcome in contract.OUTCOME_LABELS:
            calls = []

            def evaluate(train, validation, received_outcome, features, *args, **kwargs):
                self.assertEqual(received_outcome, outcome)
                calls.append(features)
                return {"AUROC": .7, "AUPRC": .5, "Brier": .2, "LogLoss": .6}, None, None

            with patch.object(nested, "C_CANDIDATES", [.1]), patch.object(nested, "evaluate_model", side_effect=evaluate):
                nested.topk_panel_comparison(pd.DataFrame(), pd.DataFrame(), ["synthetic_gene"], outcome)
            self.assertEqual(calls, [nested.marker_model_features(outcome) + ["synthetic_gene"]])

    def test_fit_and_predict_need_only_the_selected_outcome_columns(self):
        rng = np.random.default_rng(17)
        for outcome in contract.OUTCOME_LABELS:
            features = contract.selected_clinical_features(outcome)
            train = pd.DataFrame(rng.normal(size=(40, len(features))), columns=features)
            train[outcome] = [0, 1] * 20
            fitted = contract.fit_model(train, outcome, features, .1)
            evaluation = train[features].iloc[:8].copy()
            probabilities = contract.predict_proba(fitted, evaluation)
            self.assertEqual(fitted.preprocessor.transformed_names, features)
            self.assertEqual(probabilities.shape, (8,))
            self.assertTrue(np.isfinite(probabilities).all())
            with self.assertRaises(ValueError):
                contract.predict_proba(fitted, evaluation.drop(columns=features[0]))


if __name__ == "__main__":
    unittest.main()
