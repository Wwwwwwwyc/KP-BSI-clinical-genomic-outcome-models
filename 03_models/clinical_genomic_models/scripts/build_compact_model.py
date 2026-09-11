"""Build a compact clinical-genomic model for invasive phenotype prediction."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from modeling_contract import (
    AST_RESISTANCE,
    C_CANDIDATES,
    outcome_tiers,
    load_clinical_selection,
    validate_recorded_clinical_panel,
    FIVE_MARKERS,
    OUTCOME_LABELS,
    SelectorSetting,
    add_derived_predictors,
    calibration_curve_table,
    classification_metrics,
    decision_curve_table,
    feature_block,
    paired_bootstrap_metric_delta,
    prepare_features,
    pr_curve_table,
    project_root,
    require_columns,
    roc_curve_table,
    stratified_sample_indices,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


warnings.filterwarnings("ignore", category=FutureWarning)

HELP = """Function
  Build a compact clinical-genomic model for liver abscess or metastatic infection. Candidate predictors include clinical variables, antimicrobial susceptibility, hypervirulence-associated markers, and validation-selected GWAS features. The script applies training-only elastic-net stability selection, chooses the compact logistic model on 2022-2023 validation data, calibrates probabilities on that validation set, refits the final compact model on the pooled discovery and validation development cohort, and evaluates it on the combined 2024-2025 temporal test set.

Required Arguments
  --run
      Explicitly confirms execution. Running with no arguments prints this help and exits without doing work.

Optional Arguments
  --input PATH
      Model-ready feature table. Default: 02_features/analysis_datasets/merged_full.csv.
  --information-model-dir PATH
      Upstream nested-model GWAS output directory. Default: 03_models/clinical_genomic_models/model_outputs/nested_information_models.
  --output-dir PATH
      Primary output directory. Default: 03_models/clinical_genomic_models/compact_model.
  --validation-years CSV
      Calendar years used for model and threshold selection. Discovery uses all earlier years from 2013 onward. Default: 2022,2023.
  --test-years CSV
      Calendar years held out for final testing. Default: 2024,2025.
  --resamples N
      Discovery-only stability-selection resamples per elastic-net setting. Default: 80.
  --sample-fraction VALUE
      Stratified discovery fraction per stability-selection resample. Default: 0.8.
  --min-features N
      Minimum compact panel size. Default: 3.
  --max-features N
      Maximum compact panel size. Default: 12.
  --auc-tolerance VALUE
      Validation AUROC tolerance from the best eligible candidate before preferring smaller panels. Default: 0.01.
  --brier-tolerance VALUE
      Validation Brier tolerance from the best eligible candidate. Default: 0.005.
  --bootstrap N
      Paired bootstrap resamples for compact-vs-full model deltas on the test set. Default: 1000.
  --seed N
      Random seed. Default: 42.
  --overwrite
      Allow replacing files inside the output directory.
  -h, --help
      Show this help text.

Output
  Primary output path: --output-dir. The script writes selected_features.csv, selection_frequency.csv, candidate_panels.csv, compact_model_performance.csv, full_model_performance.csv, test_prediction_comparison.csv, paired_bootstrap_delta.csv, threshold_metrics.csv, roc_curve_source.csv, precision_recall_curve_source.csv, calibration_curve_source.csv, decision_curve_source.csv, feature_contract_audit.csv, run_manifest.json, and README.md.
"""


OUTCOME = "metastatic"
PANEL_ID = "full_clinical_genomic_model"
MODEL_ID = "compact_logistic_l2"
REFERENCE_ID = "full_clinical_genomic_logistic_l2"


@dataclass(frozen=True)
class Paths:
    input_path: Path
    information_model_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class SplitConfig:
    discovery_label: str
    validation_label: str
    test_label: str
    discovery_years: list[int]
    validation_years: list[int]
    test_years: list[int]


@dataclass(frozen=True)
class PanelChoice:
    source: str
    selector_c: float | None
    l1_ratio: float | None
    rank_scope: str
    n_features: int
    model_c: float
    features: tuple[str, ...]
    validation_metrics: dict[str, object]
    validation_threshold: float


def default_paths() -> Paths:
    root = project_root()
    return Paths(
        input_path=root / "02_features" / "analysis_datasets" / "merged_full.csv",
        information_model_dir=root
        / "03_models"
        / "clinical_genomic_models"
        / "model_outputs"
        / "nested_information_models",
        output_dir=root / "03_models" / "clinical_genomic_models" / "compact_model",
    )


def parse_year_csv(value: str, name: str) -> list[int]:
    try:
        years = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated year list") from exc
    if not years or any(year < 2000 or year > 2100 for year in years):
        raise argparse.ArgumentTypeError(f"{name} contains implausible calendar years")
    return sorted(set(years))


def parse_args(argv: list[str]) -> argparse.Namespace | None:
    defaults = default_paths()
    if not argv:
        print(HELP)
        return None

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--input", type=Path, default=defaults.input_path)
    parser.add_argument("--information-model-dir", type=Path, default=defaults.information_model_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--validation-years", type=lambda x: parse_year_csv(x, "validation-years"), default=[2022, 2023])
    parser.add_argument("--test-years", type=lambda x: parse_year_csv(x, "test-years"), default=[2024, 2025])
    parser.add_argument("--resamples", type=int, default=80)
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--min-features", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--auc-tolerance", type=float, default=0.01)
    parser.add_argument("--brier-tolerance", type=float, default=0.005)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    args, unknown = parser.parse_known_args(argv)

    if args.help:
        print(HELP)
        return None
    if unknown or not args.run:
        print(HELP, file=sys.stderr)
        if unknown:
            print(f"Unknown arguments: {' '.join(unknown)}", file=sys.stderr)
        if not args.run:
            print("Missing required argument: --run", file=sys.stderr)
        raise SystemExit(2)
    if set(args.validation_years) & set(args.test_years):
        raise SystemExit("--validation-years and --test-years must not overlap")
    if args.resamples <= 0:
        raise SystemExit("--resamples must be positive")
    if not 0 < args.sample_fraction <= 1:
        raise SystemExit("--sample-fraction must be in (0, 1]")
    if args.min_features <= 0 or args.max_features < args.min_features:
        raise SystemExit("--max-features must be >= --min-features > 0")
    if args.auc_tolerance < 0 or args.brier_tolerance < 0:
        raise SystemExit("--auc-tolerance and --brier-tolerance must be non-negative")
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else project_root() / path


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing_files = [p for p in path.iterdir() if p.is_file()]
    if existing_files and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite or choose a new --output-dir.")
    if overwrite:
        for file in existing_files:
            file.unlink()


def split_label(prefix: str, years: list[int]) -> str:
    return f"{prefix}_{'_'.join(str(year) for year in years)}"


def temporal_splits(df: pd.DataFrame, validation_years: list[int], test_years: list[int]) -> tuple[dict[str, pd.DataFrame], SplitConfig]:
    require_columns(df, ["年份"], "merged feature table")
    discovery_years = sorted(int(year) for year in df.loc[df["年份"].between(2013, min(validation_years) - 1), "年份"].unique())
    config = SplitConfig(
        discovery_label=split_label("discovery", discovery_years),
        validation_label=split_label("validation", validation_years),
        test_label=split_label("test", test_years),
        discovery_years=discovery_years,
        validation_years=validation_years,
        test_years=test_years,
    )
    splits = {
        config.discovery_label: df[df["年份"].isin(discovery_years)].copy(),
        config.validation_label: df[df["年份"].isin(validation_years)].copy(),
        config.test_label: df[df["年份"].isin(test_years)].copy(),
    }
    empty = [name for name, table in splits.items() if table.empty]
    if empty:
        raise ValueError(f"Temporal split is empty: {empty}")
    return splits, config


def read_locked_gwas_panel(information_model_dir: Path) -> list[str]:
    load_clinical_selection(information_model_dir / "clinical_selection.json")
    panel_path = information_model_dir / "gwas_feature_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Upstream nested-model feature panel does not exist: {panel_path}")
    panel = pd.read_csv(panel_path)
    require_columns(panel, ["outcome", "model", "feature_order", "feature", "feature_block"], "upstream nested-model feature panel")
    validate_recorded_clinical_panel(panel, OUTCOME)
    selected = panel[
        panel["outcome"].eq(OUTCOME_LABELS[OUTCOME])
        & panel["model"].eq("clinical_ast_markers_gwas")
        & panel["feature_block"].eq("gwas_features")
    ].sort_values("feature_order", kind="mergesort")
    features = selected["feature"].astype(str).tolist()
    if not features:
        raise ValueError(f"Upstream nested-model feature panel does not contain locked GWAS features for {OUTCOME_LABELS[OUTCOME]}")
    return features


def full_model_features(gwas_features: list[str]) -> list[str]:
    return outcome_tiers(OUTCOME, {OUTCOME: gwas_features})["clinical_ast_markers_gwas"]


def selection_settings() -> list[SelectorSetting]:
    return [
        SelectorSetting(selector_c=c_value, l1_ratio=l1_ratio)
        for c_value in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        for l1_ratio in [0.3, 0.5, 0.7, 0.9]
    ]


def stability_selection_frequency(
    discovery: pd.DataFrame,
    features: list[str],
    gwas_features: list[str],
    *,
    settings: list[SelectorSetting],
    resamples: int,
    sample_fraction: float,
    seed: int,
) -> pd.DataFrame:
    y_all = discovery[OUTCOME].to_numpy()
    rows = []
    for setting_index, setting in enumerate(settings):
        counts = dict.fromkeys(features, 0)
        for i in range(resamples):
            rng = np.random.default_rng(seed + setting_index * 10000 + i)
            idx = stratified_sample_indices(y_all, sample_fraction, rng)
            x, preprocessor = prepare_features(discovery.iloc[idx], features, fit=True)
            model = LogisticRegression(
                C=setting.selector_c,
                penalty="elasticnet",
                l1_ratio=setting.l1_ratio,
                solver="saga",
                max_iter=5000,
                random_state=seed + setting_index * 10000 + i,
            )
            model.fit(x, y_all[idx])
            if preprocessor.transformed_to_feature is None:
                raise ValueError("Fitted preprocessor did not expose transformed feature ownership")
            selected_columns = np.flatnonzero(np.abs(model.coef_[0]) > 1e-8)
            selected_features = {preprocessor.transformed_to_feature[col] for col in selected_columns}
            for feature in features:
                counts[feature] += int(feature in selected_features)
        for order, feature in enumerate(features, start=1):
            rows.append(
                {
                    "selector_c": setting.selector_c,
                    "l1_ratio": setting.l1_ratio,
                    "feature_order": order,
                    "feature": feature,
                    "feature_block": feature_block(feature, OUTCOME, {"metastatic": gwas_features}),
                    "selection_count": counts[feature],
                    "selection_frequency": counts[feature] / resamples,
                }
            )
    return pd.DataFrame(rows)


def rank_table(frequencies: pd.DataFrame) -> pd.DataFrame:
    ranked = (
        frequencies.groupby(["feature", "feature_order", "feature_block"], as_index=False)["selection_frequency"]
        .max()
        .rename(columns={"selection_frequency": "max_selection_frequency"})
        .sort_values(["max_selection_frequency", "feature_order"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    ranked["global_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def candidate_feature_sets(
    frequencies: pd.DataFrame,
    ranked: pd.DataFrame,
    min_features: int,
    max_features: int,
    gwas_features: list[str],
) -> list[dict[str, object]]:
    candidates: dict[tuple[str, ...], dict[str, object]] = {}

    def add(features: list[str], source: str, selector_c: float | None, l1_ratio: float | None, rank_scope: str) -> None:
        if not min_features <= len(features) <= max_features:
            return
        if not any(feature in gwas_features for feature in features):
            return
        key = tuple(features)
        candidates.setdefault(
            key,
            {
                "features": key,
                "source": source,
                "selector_c": selector_c,
                "l1_ratio": l1_ratio,
                "rank_scope": rank_scope,
            },
        )

    global_order = ranked["feature"].tolist()
    for k in range(min_features, max_features + 1):
        add(global_order[:k], "global_max_frequency_rank_prefix", None, None, "all_selector_settings")

    for (selector_c, l1_ratio), table in frequencies.groupby(["selector_c", "l1_ratio"], sort=False):
        order = table.sort_values(["selection_frequency", "feature_order"], ascending=[False, True], kind="mergesort")["feature"].tolist()
        for k in range(min_features, max_features + 1):
            add(order[:k], "setting_frequency_rank_prefix", float(selector_c), float(l1_ratio), "single_selector_setting")

    return list(candidates.values())


def fit_model(train_df: pd.DataFrame, features: list[str], c_value: float) -> tuple[object, LogisticRegression]:
    x_train, preprocessor = prepare_features(train_df, features, fit=True)
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=10000)
    model.fit(x_train, train_df[OUTCOME].to_numpy())
    if preprocessor.transformed_to_feature is None:
        raise ValueError("Fitted preprocessor did not expose transformed feature ownership")
    return preprocessor, model


def predict_with(preprocessor: object, model: LogisticRegression, eval_df: pd.DataFrame, features: list[str]) -> np.ndarray:
    x_eval, _ = prepare_features(eval_df, features, preprocessor=preprocessor)
    return np.asarray(model.predict_proba(x_eval)[:, 1], dtype=float)


def model_coefficients(preprocessor: object, model: LogisticRegression, features: list[str]) -> dict[str, float]:
    coefficients: dict[str, list[float]] = {feature: [] for feature in features}
    for owner, coef in zip(preprocessor.transformed_to_feature, model.coef_[0]):
        coefficients[owner].append(float(coef))
    return {
        feature: float(np.asarray(values)[np.argmax(np.abs(values))]) if values else 0.0
        for feature, values in coefficients.items()
    }


def fit_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame, features: list[str], c_value: float) -> np.ndarray:
    preprocessor, model = fit_model(train_df, features, c_value)
    return predict_with(preprocessor, model, eval_df, features)


def logit_prob(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def expit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-value))


def fit_platt_recalibration(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    model.fit(logit_prob(prob).reshape(-1, 1), y_true)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0][0]),
    }


def apply_platt_recalibration(prob: np.ndarray, calibrator: dict[str, float]) -> np.ndarray:
    return expit(calibrator["intercept"] + calibrator["slope"] * logit_prob(prob))


def metric_row(model: str, split: str, y_true: np.ndarray, prob: np.ndarray, features: list[str], c_value: float) -> dict[str, object]:
    return {
        "model": model,
        "outcome": OUTCOME_LABELS[OUTCOME],
        "split": split,
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "n_features": len(features),
        "C": float(c_value),
        **classification_metrics(y_true, prob),
    }


def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    roc = roc_curve_table(y_true, prob).replace([np.inf, -np.inf], np.nan).dropna(subset=["threshold"])
    roc["youden"] = roc["tpr"] - roc["fpr"]
    return float(roc.sort_values(["youden", "threshold"], ascending=[False, False], kind="mergesort")["threshold"].iloc[0])


def threshold_metrics_at(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = prob >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "sensitivity": float(recall),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "PPV": float(precision),
        "NPV": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "F1": float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
    }


def evaluate_candidates(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    candidates: list[dict[str, object]],
    validation_label: str,
) -> list[PanelChoice]:
    y_validation = validation[OUTCOME].to_numpy()
    choices = []
    for candidate in candidates:
        features = list(candidate["features"])
        for c_value in C_CANDIDATES:
            prob = fit_predict(discovery, validation, features, c_value)
            metrics = metric_row("compressed_candidate", validation_label, y_validation, prob, features, c_value)
            threshold = youden_threshold(y_validation, prob)
            choices.append(
                PanelChoice(
                    source=str(candidate["source"]),
                    selector_c=candidate["selector_c"],
                    l1_ratio=candidate["l1_ratio"],
                    rank_scope=str(candidate["rank_scope"]),
                    n_features=len(features),
                    model_c=float(c_value),
                    features=tuple(features),
                    validation_metrics=metrics,
                    validation_threshold=threshold,
                )
            )
    return choices


def choose_panel(choices: list[PanelChoice], auc_tolerance: float, brier_tolerance: float) -> tuple[PanelChoice, dict[str, object]]:
    if not choices:
        raise ValueError("No compact candidate panel satisfied the feature-count and GWAS-presence constraints")
    event_rate = float(choices[0].validation_metrics["event_rate"])
    no_information_brier = event_rate * (1 - event_rate)
    eligible = [choice for choice in choices if float(choice.validation_metrics["Brier"]) <= no_information_brier]
    if not eligible:
        eligible = choices
    best_auc = max(float(choice.validation_metrics["AUROC"]) for choice in eligible)
    best_brier = min(float(choice.validation_metrics["Brier"]) for choice in eligible)
    near = [
        choice
        for choice in eligible
        if float(choice.validation_metrics["AUROC"]) >= best_auc - auc_tolerance
        and float(choice.validation_metrics["Brier"]) <= best_brier + brier_tolerance
    ]
    if not near:
        near = eligible
    chosen = sorted(
        near,
        key=lambda choice: (
            choice.n_features,
            float(choice.validation_metrics["Brier"]),
            -float(choice.validation_metrics["AUROC"]),
            choice.source,
            choice.model_c,
        ),
    )[0]
    return chosen, {
        "validation_event_rate": event_rate,
        "validation_no_information_brier": no_information_brier,
        "total_candidate_count": len(choices),
        "eligible_candidate_count": len(eligible),
        "near_candidate_count": len(near),
        "auc_tolerance": auc_tolerance,
        "brier_tolerance": brier_tolerance,
        "eligibility": "requires at least one locked upstream GWAS feature and validation Brier no worse than no-information risk",
        "tie_break": "fewest features, then lower validation Brier, then higher validation AUROC",
    }


def candidate_table(choices: list[PanelChoice], chosen: PanelChoice) -> pd.DataFrame:
    rows = []
    for choice in choices:
        rows.append(
            {
                "selected": choice == chosen,
                "source": choice.source,
                "selector_c": choice.selector_c,
                "l1_ratio": choice.l1_ratio,
                "rank_scope": choice.rank_scope,
                "model_c": choice.model_c,
                "n_features": choice.n_features,
                "validation_threshold_youden": choice.validation_threshold,
                "AUROC": choice.validation_metrics["AUROC"],
                "AUPRC": choice.validation_metrics["AUPRC"],
                "Brier": choice.validation_metrics["Brier"],
                "LogLoss": choice.validation_metrics["LogLoss"],
                "features": "|".join(choice.features),
            }
        )
    return pd.DataFrame(rows).sort_values(["selected", "n_features", "Brier", "AUROC"], ascending=[False, True, True, False])


def selected_feature_table(features: list[str], ranked: pd.DataFrame, coef_lookup: dict[str, float], gwas_features: list[str]) -> pd.DataFrame:
    rank_lookup = ranked.set_index("feature")
    rows = []
    for order, feature in enumerate(features, start=1):
        rows.append(
            {
                "feature_order": order,
                "feature": feature,
                "feature_block": feature_block(feature, OUTCOME, {"metastatic": gwas_features}),
                "global_stability_rank": int(rank_lookup.loc[feature, "global_rank"]),
                "max_selection_frequency": float(rank_lookup.loc[feature, "max_selection_frequency"]),
                "final_l2_coefficient": coef_lookup[feature],
            }
        )
    return pd.DataFrame(rows)


def prediction_tables(
    chosen: PanelChoice,
    splits: dict[str, pd.DataFrame],
    config: SplitConfig,
    full_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    discovery = splits[config.discovery_label]
    validation = splits[config.validation_label]
    test = splits[config.test_label]
    selected_features = list(chosen.features)
    y_validation = validation[OUTCOME].to_numpy()
    y_test = test[OUTCOME].to_numpy()

    validation_prob_raw = fit_predict(discovery, validation, selected_features, chosen.model_c)
    compact_calibrator = fit_platt_recalibration(y_validation, validation_prob_raw)
    validation_prob = apply_platt_recalibration(validation_prob_raw, compact_calibrator)
    validation_threshold = youden_threshold(y_validation, validation_prob)
    final_development = pd.concat([discovery, validation], ignore_index=True)
    final_preprocessor, final_model = fit_model(final_development, selected_features, chosen.model_c)
    test_prob_raw = predict_with(final_preprocessor, final_model, test, selected_features)
    final_coefficients = model_coefficients(final_preprocessor, final_model, selected_features)
    test_prob = apply_platt_recalibration(test_prob_raw, compact_calibrator)

    full_validation_prob_raw = fit_predict(discovery, validation, full_features, 0.01)
    full_calibrator = fit_platt_recalibration(y_validation, full_validation_prob_raw)
    full_validation_prob = apply_platt_recalibration(full_validation_prob_raw, full_calibrator)
    reference_threshold = youden_threshold(y_validation, full_validation_prob)
    full_prob_raw = fit_predict(discovery, test, full_features, 0.01)
    full_prob = apply_platt_recalibration(full_prob_raw, full_calibrator)

    compressed_rows = [
        metric_row(MODEL_ID, config.validation_label, y_validation, validation_prob, selected_features, chosen.model_c),
        metric_row(MODEL_ID, config.test_label, y_test, test_prob, selected_features, chosen.model_c),
    ]
    reference_rows = [
        metric_row(REFERENCE_ID, config.test_label, y_test, full_prob, full_features, 0.01),
    ]
    for row in compressed_rows:
        row.update(
            {
                "probability_scale": "validation_recalibrated_platt",
                "validation_recalibration_intercept": compact_calibrator["intercept"],
                "validation_recalibration_slope": compact_calibrator["slope"],
            }
        )
    reference_rows[0].update(
        {
            "probability_scale": "validation_recalibrated_platt",
            "validation_recalibration_intercept": full_calibrator["intercept"],
            "validation_recalibration_slope": full_calibrator["slope"],
        }
    )
    compressed_rows[0].update({f"locked_{key}": value for key, value in threshold_metrics_at(y_validation, validation_prob, validation_threshold).items()})
    compressed_rows[1].update({f"locked_{key}": value for key, value in threshold_metrics_at(y_test, test_prob, validation_threshold).items()})
    reference_rows[0].update({f"locked_{key}": value for key, value in threshold_metrics_at(y_test, full_prob, reference_threshold).items()})

    predictions = test[["strain", "年份"]].copy() if "strain" in test.columns else test[["年份"]].copy()
    predictions["outcome"] = OUTCOME_LABELS[OUTCOME]
    predictions["split"] = config.test_label
    predictions["observed"] = y_test
    predictions[f"{MODEL_ID}_raw_prob"] = test_prob_raw
    predictions[f"{MODEL_ID}_prob"] = test_prob
    predictions[f"{REFERENCE_ID}_raw_prob"] = full_prob_raw
    predictions[f"{REFERENCE_ID}_prob"] = full_prob
    return pd.DataFrame(compressed_rows), pd.DataFrame(reference_rows), predictions, final_coefficients


def source_data_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = predictions["observed"].to_numpy()
    prob_by_model = {
        MODEL_ID: predictions[f"{MODEL_ID}_prob"].to_numpy(),
        REFERENCE_ID: predictions[f"{REFERENCE_ID}_prob"].to_numpy(),
    }
    roc_rows = []
    pr_rows = []
    cal_rows = []
    threshold_rows = []
    for model, prob in prob_by_model.items():
        roc = roc_curve_table(y, prob)
        roc.insert(0, "model", model)
        roc_rows.append(roc)
        pr = pr_curve_table(y, prob)
        pr.insert(0, "model", model)
        pr_rows.append(pr)
        cal = calibration_curve_table(y, prob)
        cal.insert(0, "model", model)
        cal_rows.append(cal)
        threshold_rows.append({"model": model, "threshold_source": "test_youden_descriptive", **threshold_metrics_at(y, prob, youden_threshold(y, prob))})
    return (
        pd.concat(roc_rows, ignore_index=True),
        pd.concat(pr_rows, ignore_index=True),
        pd.concat(cal_rows, ignore_index=True),
        decision_curve_table(y, prob_by_model),
        pd.DataFrame(threshold_rows),
    )


def feature_contract_audit(all_features: list[str], selected_features: list[str], gwas_features: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_order": order,
            "feature": feature,
            "feature_block": feature_block(feature, OUTCOME, {"metastatic": gwas_features}),
            "in_full_model_candidate_pool": True,
            "selected_in_compact_model": feature in selected_features,
        }
        for order, feature in enumerate(all_features, start=1)
    )


def write_readme(output_dir: Path, compressed: pd.DataFrame, reference: pd.DataFrame, selected: list[str], config: SplitConfig) -> None:
    test_row = compressed[compressed["split"].str.startswith("test_")].iloc[0]
    reference_row = reference.iloc[0]
    discovery_span = f"{min(config.discovery_years)}-{max(config.discovery_years)}"
    validation_span = f"{min(config.validation_years)}-{max(config.validation_years)}"
    test_span = f"{min(config.test_years)}-{max(config.test_years)}"
    text = f"""# Compact Metastatic-Infection Model

This directory contains the compact model selected from the full clinical-genomic predictor set.

## Selected Features

{chr(10).join(f'- `{feature}`' for feature in selected)}

## Test Performance

| model | n_features | AUROC | AUPRC | Brier | sensitivity | specificity | PPV | NPV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Compact model | {int(test_row['n_features'])} | {float(test_row['AUROC']):.3f} | {float(test_row['AUPRC']):.3f} | {float(test_row['Brier']):.3f} | {float(test_row['locked_sensitivity']):.3f} | {float(test_row['locked_specificity']):.3f} | {float(test_row['locked_PPV']):.3f} | {float(test_row['locked_NPV']):.3f} |
| Full model | {int(reference_row['n_features'])} | {float(reference_row['AUROC']):.3f} | {float(reference_row['AUPRC']):.3f} | {float(reference_row['Brier']):.3f} | {float(reference_row['locked_sensitivity']):.3f} | {float(reference_row['locked_specificity']):.3f} | {float(reference_row['locked_PPV']):.3f} | {float(reference_row['locked_NPV']):.3f} |

Predicted probabilities are generated by the final model refitted on the pooled {discovery_span} development cohort. They are recalibrated by a Platt intercept and slope fitted on the {validation_span} validation predictions of the {discovery_span} selection model and applied unchanged to the combined {test_span} test split. Threshold-dependent metrics use thresholds selected on the recalibrated validation predictions and applied unchanged to the test split.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run(paths: Paths, args: argparse.Namespace) -> None:
    raw = pd.read_csv(paths.input_path, low_memory=False)
    require_columns(raw, ["年份", OUTCOME], "merged feature table")
    merged = add_derived_predictors(raw)
    gwas_features = read_locked_gwas_panel(paths.information_model_dir)
    features = full_model_features(gwas_features)
    require_columns(merged, features, "full clinical-genomic feature pool")

    splits, config = temporal_splits(merged, args.validation_years, args.test_years)
    discovery = splits[config.discovery_label]
    validation = splits[config.validation_label]
    frequencies = stability_selection_frequency(
        discovery,
        features,
        gwas_features,
        settings=selection_settings(),
        resamples=args.resamples,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
    ranked = rank_table(frequencies)
    candidates = candidate_feature_sets(frequencies, ranked, args.min_features, args.max_features, gwas_features)
    choices = evaluate_candidates(discovery, validation, candidates, config.validation_label)
    chosen, selection_audit = choose_panel(choices, args.auc_tolerance, args.brier_tolerance)
    selected_features = list(chosen.features)

    compressed, reference, predictions, final_coefficients = prediction_tables(chosen, splits, config, features)
    roc, pr, cal, dca, threshold = source_data_tables(predictions)
    paired = paired_bootstrap_metric_delta(
        predictions,
        split_col="split",
        outcome_col="observed",
        reference_col=f"{REFERENCE_ID}_prob",
        comparator_col=f"{MODEL_ID}_prob",
        metric_label="compact_vs_full_model",
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    selected_features_df = selected_feature_table(selected_features, ranked, final_coefficients, gwas_features)

    selected_features_df.to_csv(paths.output_dir / "selected_features.csv", index=False)
    frequencies.to_csv(paths.output_dir / "selection_frequency.csv", index=False)
    candidate_table(choices, chosen).to_csv(paths.output_dir / "candidate_panels.csv", index=False)
    compressed.to_csv(paths.output_dir / "compact_model_performance.csv", index=False)
    reference.to_csv(paths.output_dir / "full_model_performance.csv", index=False)
    predictions.to_csv(paths.output_dir / "test_prediction_comparison.csv", index=False)
    paired.to_csv(paths.output_dir / "paired_bootstrap_delta.csv", index=False)
    threshold.to_csv(paths.output_dir / "threshold_metrics.csv", index=False)
    roc.to_csv(paths.output_dir / "roc_curve_source.csv", index=False)
    pr.to_csv(paths.output_dir / "precision_recall_curve_source.csv", index=False)
    cal.to_csv(paths.output_dir / "calibration_curve_source.csv", index=False)
    dca.to_csv(paths.output_dir / "decision_curve_source.csv", index=False)
    feature_contract_audit(features, selected_features, gwas_features).to_csv(paths.output_dir / "feature_contract_audit.csv", index=False)

    manifest = {
        "input_path": str(paths.input_path),
        "information_model_dir": str(paths.information_model_dir),
        "output_dir": str(paths.output_dir),
        "outcome": OUTCOME_LABELS[OUTCOME],
        "candidate_pool": "clinical predictors + antimicrobial susceptibility + hypervirulence-associated markers + validation-selected GWAS features",
        "gwas_features": gwas_features,
        "selected_features": selected_features,
        "selected_model_c": chosen.model_c,
        "selection_audit": selection_audit,
        "split_config": {
            "discovery_split": config.discovery_label,
            "validation_split": config.validation_label,
            "test_split": config.test_label,
            "discovery_years": config.discovery_years,
            "validation_years": config.validation_years,
            "test_years": config.test_years,
        },
        "final_model_training_split": f"development_pooled_{config.discovery_label}_{config.validation_label}",
        "final_model_training_years": sorted(set(config.discovery_years) | set(config.validation_years)),
        "probability_calibration": f"fit Platt intercept and slope on {config.validation_label} predictions generated by the {config.discovery_label} selection model; apply unchanged to temporal-test probabilities from the pooled development final model",
        "threshold_policy": "select Youden threshold on validation-recalibrated predictions; apply unchanged to the test set",
        "bootstrap_resamples": args.bootstrap,
        "generated_files": sorted({path.name for path in paths.output_dir.iterdir() if path.is_file()} | {"run_manifest.json", "README.md"}),
    }
    with (paths.output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    write_readme(paths.output_dir, compressed, reference, selected_features, config)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args is None:
        return 0
    paths = Paths(
        input_path=resolve_path(args.input).resolve(),
        information_model_dir=resolve_path(args.information_model_dir).resolve(),
        output_dir=resolve_path(args.output_dir).resolve(),
    )
    if not paths.input_path.exists():
        raise FileNotFoundError(f"Input feature table does not exist: {paths.input_path}")
    if not paths.information_model_dir.exists():
        raise FileNotFoundError(f"Upstream nested-model directory does not exist: {paths.information_model_dir}")
    prepare_output_dir(paths.output_dir, args.overwrite)
    run(paths, args)
    print(f"Wrote compact model outputs to: {paths.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
