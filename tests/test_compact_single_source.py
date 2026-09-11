"""Guard that the exported compact-model coefficients come from the same pooled fit as the test predictions."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_models/clinical_genomic_models/scripts"))
import build_compact_model as compact


def _frame(rng, years, n, features):
    rows = []
    for i, year in enumerate(years):
        for j in range(n):
            row = {"strain": f"{year}_{j}", "年份": year}
            for k, feature in enumerate(features):
                row[feature] = rng.normal(loc=k * 0.3, scale=1.0)
            row[compact.OUTCOME] = int((row[features[0]] + rng.normal(scale=0.5)) > 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def _splits(rng, features):
    discovery = _frame(rng, [2013, 2014, 2021], 40, features)
    validation = _frame(rng, [2022, 2023], 40, features)
    test = _frame(rng, [2024, 2025], 40, features)
    config = compact.SplitConfig(
        discovery_label="discovery_2013_2021",
        validation_label="validation_2022_2023",
        test_label="test_2024_2025",
        discovery_years=[2013, 2014, 2021],
        validation_years=[2022, 2023],
        test_years=[2024, 2025],
    )
    splits = {
        config.discovery_label: discovery,
        config.validation_label: validation,
        config.test_label: test,
    }
    return splits, config


class CompactSingleSourceTests(unittest.TestCase):
    def test_exported_coefficients_match_pooled_fit_that_produced_test_predictions(self):
        rng = np.random.default_rng(7)
        selected = ["feat_a", "feat_b"]
        full_features = selected + ["feat_c"]
        splits, config = _splits(rng, full_features)
        chosen = compact.PanelChoice(
            source="synthetic",
            selector_c=None,
            l1_ratio=None,
            rank_scope="synthetic",
            n_features=len(selected),
            model_c=0.01,
            features=tuple(selected),
            validation_metrics={"AUROC": 0.7, "Brier": 0.2},
            validation_threshold=0.5,
        )
        compressed, reference, predictions, final_coefficients = compact.prediction_tables(
            chosen, splits, config, full_features
        )

        discovery = splits[config.discovery_label]
        validation = splits[config.validation_label]
        test = splits[config.test_label]
        pooled = pd.concat([discovery, validation], ignore_index=True)

        pooled_pre, pooled_model = compact.fit_model(pooled, selected, chosen.model_c)
        pooled_coef = compact.model_coefficients(pooled_pre, pooled_model, selected)
        pooled_prob = compact.predict_with(pooled_pre, pooled_model, test, selected)

        discovery_pre, discovery_model = compact.fit_model(discovery, selected, chosen.model_c)
        discovery_coef = compact.model_coefficients(discovery_pre, discovery_model, selected)

        for feature in selected:
            self.assertAlmostEqual(final_coefficients[feature], pooled_coef[feature], places=10)
        np.testing.assert_allclose(
            predictions[f"{compact.MODEL_ID}_raw_prob"].to_numpy(), pooled_prob, rtol=0, atol=1e-10
        )
        self.assertFalse(
            np.allclose(
                [final_coefficients[f] for f in selected],
                [discovery_coef[f] for f in selected],
                atol=1e-8,
            ),
            "Exported coefficients must come from the pooled development fit, not the discovery-only fit.",
        )


if __name__ == "__main__":
    unittest.main()
