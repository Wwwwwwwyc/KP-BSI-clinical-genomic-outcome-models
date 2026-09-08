"""Evaluate saved probabilities: percentile intervals, Wilson intervals and DCA."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "03_models/clinical_genomic_models/scripts"))
from modeling_contract import calibration_intercept_slope, decision_curve_table
from build_compact_model import fit_platt_recalibration, apply_platt_recalibration

METRICS = ("AUROC", "AUPRC", "Brier", "LogLoss")
TIERS = ("G0_clinical", "G1_clinical_AST", "G2_clinical_AST_5markers", "G3_clinical_AST_5markers_GWAS")
COMPACT, FULL = "compact_logistic_l2", "full_g3_logistic_l2"
GRID = np.round(np.arange(0.05, 0.51, 0.01), 2)


def check_predictions(frame: pd.DataFrame, models: list[str]) -> None:
    if frame.empty or frame.strain.isna().any() or not frame.strain.is_unique:
        raise ValueError("Prediction records require unique, nonmissing episode identifiers")
    if set(frame.observed.unique()) != {0, 1}:
        raise ValueError("Observed outcomes must contain both binary classes without missing values")
    p = frame[[f"{m}_prob" for m in models]].to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Predicted probabilities must be finite and between zero and one")


def metrics(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_score(y, p), average_precision_score(y, p),
                     brier_score_loss(y, p), log_loss(y, p, labels=[0, 1])])


def interval(point: float, draws: np.ndarray) -> dict:
    bounds = np.percentile(draws, [2.5, 97.5])
    return {"estimate": float(point), "ci_low": float(bounds[0]),
            "ci_high": float(bounds[1]), "resamples": len(draws)}


def bootstrap_group(frame: pd.DataFrame, models: list[str], repeats: int, seed: int,
                    calibration: bool = False) -> tuple[dict, dict, dict]:
    check_predictions(frame, models)
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    y = frame.observed.to_numpy(int)
    p = {m: frame[f"{m}_prob"].to_numpy(float) for m in models}
    point = {m: metrics(y, prob) for m, prob in p.items()}
    draws, cal_draws = {m: [] for m in models}, {m: [] for m in models}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if np.unique(y[idx]).size != 2:
            continue
        for m, prob in p.items():
            draws[m].append(metrics(y[idx], prob[idx]))
            if calibration:
                # Diagnostic calibration regressions evaluate saved probabilities;
                # their coefficients never replace the validation Platt calibrator.
                cal_draws[m].append(calibration_intercept_slope(y[idx], prob[idx]))
    if not draws[models[0]]:
        raise ValueError("No bootstrap samples contained both outcome classes")
    draws = {m: np.asarray(v) for m, v in draws.items()}
    result = {m: {name: interval(point[m][i], draws[m][:, i])
                  for i, name in enumerate(METRICS)} for m in models}
    if calibration:
        for m in models:
            cal_point = calibration_intercept_slope(y, p[m])
            for i, name in enumerate(("Calibration intercept", "Calibration slope")):
                result[m][name] = interval(cal_point[i], np.asarray(cal_draws[m])[:, i])
    return result, point, draws


def paired_intervals(point: dict, draws: dict, reference: str, comparator: str) -> dict:
    return {name: interval(point[comparator][i] - point[reference][i],
                           draws[comparator][:, i] - draws[reference][:, i])
            for i, name in enumerate(METRICS)}


def operating_intervals(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("A finite validation-selected probability threshold is required")
    positive = p >= threshold
    tp, fn = int(np.sum(positive & (y == 1))), int(np.sum(~positive & (y == 1)))
    tn, fp = int(np.sum(~positive & (y == 0))), int(np.sum(positive & (y == 0)))
    result = {"threshold": threshold, "TP": tp, "FN": fn, "TN": tn, "FP": fp}
    for name, successes, total in [("Sensitivity", tp, tp+fn), ("Specificity", tn, tn+fp),
                                    ("PPV", tp, tp+fp), ("NPV", tn, tn+fn)]:
        if total == 0:
            result[name] = {"estimate": None, "ci_low": None, "ci_high": None}
        else:
            lo, hi = proportion_confint(successes, total, alpha=0.05, method="wilson")
            result[name] = {"estimate": successes/total, "ci_low": float(lo), "ci_high": float(hi)}
    return result


def decision_curves(ladder: pd.DataFrame, compact: pd.DataFrame,
                    validation_split: str, test_split: str) -> tuple[pd.DataFrame, dict]:
    outcome = compact.outcome.unique()
    if len(outcome) != 1:
        raise ValueError("Compact predictions must contain one outcome")
    selected = ladder.loc[ladder.outcome.eq(outcome[0])]
    validation = selected.loc[selected.split.eq(validation_split)].copy()
    test = selected.loc[selected.split.eq(test_split)].copy()
    check_predictions(validation, list(TIERS))
    check_predictions(test, list(TIERS))
    check_predictions(compact, [COMPACT, FULL])
    if set(validation.strain) & set(test.strain) or set(test.strain) != set(compact.strain):
        raise ValueError("Validation/test overlap or unmatched test episodes")
    test = test.set_index("strain").loc[compact.strain].reset_index()
    if not np.array_equal(test.observed.to_numpy(), compact.observed.to_numpy()):
        raise ValueError("Outcome labels differ across prediction files")
    # Compact/full probabilities already carry their own locked validation Platt
    # transformations. Fit only the three additional reference calibrators here.
    probabilities = {"Compact": compact[f"{COMPACT}_prob"].to_numpy(float),
                     "Full": compact[f"{FULL}_prob"].to_numpy(float)}
    calibrators = {}
    for name, tier in zip(("Clinical", "Clinical + AST", "Clinical + AST + markers"), TIERS[:3]):
        calibrators[name] = fit_platt_recalibration(validation.observed.to_numpy(int),
                                                    validation[f"{tier}_prob"].to_numpy(float))
        probabilities[name] = apply_platt_recalibration(test[f"{tier}_prob"].to_numpy(float), calibrators[name])
    curves = decision_curve_table(compact.observed.to_numpy(int), probabilities, GRID)
    curves["probability_scale"] = "validation_calibrated"
    return curves, calibrators


def run(args) -> None:
    ladder_path = args.ladder_predictions
    compact_path = args.compact_model_dir / "test_prediction_comparison.csv"
    ladder = pd.read_csv(ladder_path, dtype={"strain": str})
    compact = pd.read_csv(compact_path, dtype={"strain": str})
    if not compact.split.eq(args.test_split).all():
        raise ValueError("Compact prediction split differs from --test-split")
    result = {"ladder": {}, "compact": {}, "operating": {}, "paired": {}}
    for (outcome, split), frame in ladder.groupby(["outcome", "split"], sort=True):
        if split not in (args.validation_split, args.test_split):
            continue
        group, point, draws = bootstrap_group(frame, list(TIERS), args.ladder_bootstrap, args.seed)
        if split == args.test_split:
            for reference, comparator in zip(TIERS[:-1], TIERS[1:]):
                group[f"{comparator}_minus_{reference}"] = paired_intervals(point, draws, reference, comparator)
        result["ladder"][f"{outcome}|{split}"] = group
    result["compact"], point, draws = bootstrap_group(compact, [COMPACT, FULL], args.compact_bootstrap, args.seed, calibration=True)
    result["paired"] = paired_intervals(point, draws, FULL, COMPACT)
    inputs = [ladder_path, compact_path]
    for model, filename in [(COMPACT, "compact_model_performance.csv"), (FULL, "full_model_performance.csv")]:
        path = args.compact_model_dir / filename
        inputs.append(path)
        perf = pd.read_csv(path)
        row = perf.loc[perf.split.eq(args.test_split)]
        if len(row) != 1:
            raise ValueError(f"Expected one temporal-test performance row in {path.name}")
        result["operating"][model] = operating_intervals(compact.observed.to_numpy(int),
            compact[f"{model}_prob"].to_numpy(float), float(row.iloc[0].locked_threshold))
    curves, calibrators = decision_curves(ladder, compact, args.validation_split, args.test_split)
    result["manifest"] = {
        "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "seed": args.seed, "ladder_resamples": args.ladder_bootstrap, "compact_resamples": args.compact_bootstrap,
        "interval_method": "Episode-level paired bootstrap; percentile 95% CI; original-sample point estimates and direct differences",
        "operating_method": "Wilson binomial 95% CI at fixed validation-selected thresholds",
        "prediction_model_refitting": False,
        "diagnostic_calibration": "Evaluation-only intercept and slope; no test-set probability recalibration",
        "dca": "Saved calibrated compact/full probabilities; three additional reference Platt calibrators fitted on validation only",
        "reference_calibrators": calibrators,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "locked_performance_intervals.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    curves.to_csv(args.output_dir / "decision_curves.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Function\nEvaluate saved probabilities with bootstrap/Wilson intervals and five-model decision curves.\n\nOutput\nA new private directory containing locked_performance_intervals.json and decision_curves.csv; no plots or uploads.", formatter_class=argparse.RawTextHelpFormatter)
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("--ladder-predictions", type=Path, required=True)
    required.add_argument("--compact-model-dir", type=Path, required=True)
    required.add_argument("--output-dir", type=Path, required=True)
    optional = parser.add_argument_group("Optional Arguments")
    optional.add_argument("--validation-split", default="validation_2022_2023")
    optional.add_argument("--test-split", default="test_2024_2025")
    optional.add_argument("--ladder-bootstrap", type=int, default=2000)
    optional.add_argument("--compact-bootstrap", type=int, default=1000)
    optional.add_argument("--seed", type=int, default=42)
    optional.add_argument("--run", action="store_true")
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    if not args.run or min(args.ladder_bootstrap, args.compact_bootstrap) < 1:
        parser.error("--run and positive bootstrap counts are required")
    run(args)


if __name__ == "__main__":
    main()
