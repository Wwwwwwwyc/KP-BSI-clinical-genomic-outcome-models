"""Shared feature and evaluation contract for KP-BSI outcome models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="'penalty' was deprecated.*",
)

OUTCOME_LABELS = {
    "death_30d": "mortality_30d",
    "metastatic": "metastatic_infection",
}

EXCLUDED_MAIN_PREDICTORS = ["年份", "是否入ICU（1是0否）", "感染来源"]

# Frozen outcome-specific clinical predictors from Supplementary Table 2.
CLINICAL_FEATURES_BY_OUTCOME = {
    "death_30d": [
        "2-3月前是否入院（1是0否）",
        "是否激素（1是0否）",
        "WBC-1",
        "N-1",
        "HB-1",
        "PLT-1",
        "CRP-1",
        "PCT-1",
        "ALT-1",
        "CHE-1",
        "Tbil-1",
        "Cr-1",
        "INR-1",
        "APACHEⅡ评分",
        "pitt_cont",
        "SOFA评分"
    ],
    "metastatic": [
        "男1女2",
        "年龄",
        "2-3月前是否入院（1是0否）",
        "手术史（1是0否）",
        "是否有糖尿病（1是0否）",
        "是否有肝炎（1是0否）",
        "是否有癌症（1是0否）",
        "高血压（1是0否）",
        "冠心病（1是0否）",
        "脑梗死脑出血（1是0否）",
        "肾功能不全（1是0否）",
        "器官移植状态（1是0否）",
        "是否放化疗（1是0否）",
        "是否激素（1是0否）",
        "是否免疫抑制剂（1是0否）",
        "APACHEⅡ评分",
        "pitt_cont"
    ]
}

AST_RESISTANCE = ["res_CRKP", "res_ESBL"]
FIVE_MARKERS = ["iucA", "iroB", "peg344", "rmpA", "rmpA2"]
FIVE_MARKER_BINARY = ["five_marker_all_positive", "five_marker_partial_positive"]
CATEGORICAL_LEVELS: dict[str, list[str]] = {}

C_CANDIDATES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
VALIDATION_SPLIT = "validation_2022_2023"
TEST_SPLITS = ["test_2024_2025"]
EVAL_SPLITS = [VALIDATION_SPLIT, *TEST_SPLITS]


@dataclass(frozen=True)
class FittedModel:
    model: LogisticRegression
    preprocessor: "FeaturePreprocessor"
    features: list[str]


@dataclass
class FeaturePreprocessor:
    """Fit numeric scaling and categorical one-hot expansion without changing feature-level semantics."""

    features: list[str]
    scaler: StandardScaler | None = None
    numeric_medians: dict[str, float] | None = None
    transformed_names: list[str] | None = None
    transformed_to_feature: list[str] | None = None

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self._raw_matrix(df, fit=True)
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(matrix)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("FeaturePreprocessor must be fitted before transform")
        matrix = self._raw_matrix(df, fit=False)
        return self.scaler.transform(matrix)

    def _raw_matrix(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        require_columns(df, self.features, "feature matrix")
        arrays = []
        names = []
        owners = []
        if fit:
            self.numeric_medians = {}

        for feature in self.features:
            if feature in CATEGORICAL_LEVELS:
                values = df[feature].astype("string").fillna("other")
                mapped = values.where(values.isin(CATEGORICAL_LEVELS[feature]), "other")
                for level in CATEGORICAL_LEVELS[feature]:
                    arrays.append((mapped == level).astype(float).to_numpy().reshape(-1, 1))
                    names.append(f"{feature}={level}")
                    owners.append(feature)
            else:
                series = pd.to_numeric(df[feature], errors="coerce")
                if fit:
                    median = float(series.median()) if not series.dropna().empty else 0.0
                    self.numeric_medians[feature] = median
                if self.numeric_medians is None or feature not in self.numeric_medians:
                    raise ValueError(f"Missing fitted median for numeric feature: {feature}")
                arrays.append(series.fillna(self.numeric_medians[feature]).to_numpy(dtype=float).reshape(-1, 1))
                names.append(feature)
                owners.append(feature)

        if fit:
            self.transformed_names = names
            self.transformed_to_feature = owners
        return np.hstack(arrays) if arrays else np.empty((len(df), 0))


@dataclass(frozen=True)
class SelectorSetting:
    selector_c: float
    l1_ratio: float


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def add_derived_predictors(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["pitt_cont", "resistance", *FIVE_MARKERS], "merged feature table")
    out = df.copy()

    if not out["resistance"].isin(["SKPN", "CRKP", "ESBL"]).all():
        raise ValueError("AST predictors require an observed SKPN, CRKP or ESBL phenotype")
    out["res_CRKP"] = (out["resistance"] == "CRKP").astype(int)
    out["res_ESBL"] = (out["resistance"] == "ESBL").astype(int)

    marker_matrix = out[FIVE_MARKERS].apply(pd.to_numeric, errors="raise").fillna(0)
    if not marker_matrix.isin([0, 1]).all().all():
        raise ValueError("Hypervirulence-marker calls must be binary or absent")
    out[FIVE_MARKERS] = marker_matrix
    marker_count = marker_matrix.sum(axis=1)
    out["five_marker_all_positive"] = (marker_count == len(FIVE_MARKERS)).astype(int)
    out["five_marker_partial_positive"] = ((marker_count > 0) & (marker_count < len(FIVE_MARKERS))).astype(int)

    return out


def split_by_year(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    require_columns(df, ["年份"], "merged feature table")
    return {
        "discovery_2013_2021": df[df["年份"].between(2013, 2021)].copy(),
        VALIDATION_SPLIT: df[df["年份"].between(2022, 2023)].copy(),
        "test_2024_2025": df[df["年份"].between(2024, 2025)].copy(),
    }


def resolve_gwas_panels(gwas_panels: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    return gwas_panels if gwas_panels is not None else {}


def gene_base(feature: str) -> str:
    parts = str(feature).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and parts[0] != "group":
        return parts[0]
    return str(feature)


ACTIVE_CLINICAL_PANELS = {o: list(p) for o, p in CLINICAL_FEATURES_BY_OUTCOME.items()}


def load_clinical_selection(path: Path) -> None:
    """Install the explicit training-only selection record for this process."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("training_years") != list(range(2013, 2022)):
        raise ValueError("Clinical selection must use 2013-2021 training records only")
    panels = record["selected_panels"]
    if set(panels) != set(OUTCOME_LABELS):
        raise ValueError("Clinical selection must include both outcomes")
    for outcome, features in panels.items():
        if (not isinstance(features, list) or not features or
                not all(isinstance(f, str) and f for f in features) or
                len(features) != len(set(features)) or
                set(features) & set([*OUTCOME_LABELS, *EXCLUDED_MAIN_PREDICTORS, "strain"])):
            raise ValueError(f"Invalid selected clinical panel for {outcome}")
    ACTIVE_CLINICAL_PANELS.update({o: list(p) for o, p in panels.items()})


def selected_clinical_features(outcome: str) -> list[str]:
    """Return a fresh copy so one model cannot mutate another model's panel."""
    if outcome not in CLINICAL_FEATURES_BY_OUTCOME:
        raise ValueError(f"Unsupported outcome: {outcome}")
    return list(ACTIVE_CLINICAL_PANELS[outcome])


def validate_recorded_clinical_panel(panel: pd.DataFrame, outcome: str) -> None:
    recorded = panel.loc[panel["outcome"].eq(OUTCOME_LABELS[outcome]) & panel["model"].eq("clinical")]
    if recorded.sort_values("feature_order")["feature"].tolist() != selected_clinical_features(outcome):
        raise ValueError("Upstream model panel disagrees with clinical_selection.json")


def outcome_tiers(outcome: str, gwas_panels: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    panels = resolve_gwas_panels(gwas_panels)
    if outcome not in panels:
        raise ValueError(f"Unsupported outcome: {outcome}")
    clinical = selected_clinical_features(outcome)
    return {
        "clinical": list(clinical),
        "clinical_ast": clinical + AST_RESISTANCE,
        "clinical_ast_markers": clinical + AST_RESISTANCE + FIVE_MARKERS,
        "clinical_ast_markers_gwas": clinical
        + AST_RESISTANCE
        + FIVE_MARKERS
        + panels[outcome],
    }


def check_no_excluded_predictors(tiers: dict[str, list[str]]) -> None:
    violations = {
        tier: [feature for feature in features if feature in EXCLUDED_MAIN_PREDICTORS]
        for tier, features in tiers.items()
    }
    violations = {tier: bad for tier, bad in violations.items() if bad}
    if violations:
        raise ValueError(f"Excluded predictors entered model panels: {violations}")


def validate_feature_panel(df: pd.DataFrame, tiers: dict[str, list[str]]) -> None:
    check_no_excluded_predictors(tiers)
    for tier, features in tiers.items():
        require_columns(df, features, f"{tier} feature panel")


def prepare_features(
    df: pd.DataFrame,
    features: list[str],
    *,
    preprocessor: FeaturePreprocessor | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, FeaturePreprocessor]:
    preprocessor = preprocessor or FeaturePreprocessor(features=list(features))
    x = preprocessor.fit_transform(df) if fit else preprocessor.transform(df)
    return x, preprocessor


def fit_model(train_df: pd.DataFrame, outcome: str, features: list[str], c_value: float) -> FittedModel:
    x_train, preprocessor = prepare_features(train_df, features, fit=True)
    y_train = train_df[outcome].to_numpy()
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=10000)
    model.fit(x_train, y_train)
    return FittedModel(model=model, preprocessor=preprocessor, features=features)


def predict_proba(fitted: FittedModel, eval_df: pd.DataFrame) -> np.ndarray:
    x_eval, _ = prepare_features(eval_df, fitted.features, preprocessor=fitted.preprocessor)
    return fitted.model.predict_proba(x_eval)[:, 1]


def stratified_sample_indices(y: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    sampled = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        n_take = max(1, int(round(len(indices) * fraction)))
        sampled.append(rng.choice(indices, size=n_take, replace=False))
    return np.sort(np.concatenate(sampled))


def tune_c(discovery: pd.DataFrame, outcome: str, features: list[str]) -> float:
    y = discovery[outcome].to_numpy()
    best_c = C_CANDIDATES[0]
    best_brier = float("inf")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for c_value in C_CANDIDATES:
        fold_briers = []
        for train_idx, val_idx in splitter.split(np.zeros(len(y)), y):
            train_df = discovery.iloc[train_idx]
            val_df = discovery.iloc[val_idx]
            fitted = fit_model(train_df, outcome, features, c_value)
            prob = predict_proba(fitted, val_df)
            fold_briers.append(brier_score_loss(val_df[outcome].to_numpy(), prob))
        mean_brier = float(np.mean(fold_briers))
        if mean_brier < best_brier:
            best_brier = mean_brier
            best_c = c_value
    return best_c


def calibration_intercept_slope(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(prob, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    model.fit(logit, y_true)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_intercept_slope(y_true, prob)
    return {
        "AUROC": float(roc_auc_score(y_true, prob)),
        "AUPRC": float(average_precision_score(y_true, prob)),
        "Brier": float(brier_score_loss(y_true, prob)),
        "LogLoss": float(log_loss(y_true, prob)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def threshold_metrics(y_true: np.ndarray, prob: np.ndarray, *, threshold: float | None = None) -> dict[str, float]:
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, prob)
        finite = np.isfinite(thresholds)
        threshold = float(thresholds[finite][int(np.argmax(tpr[finite] - fpr[finite]))])
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold_youden": threshold,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "PPV": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "NPV": float(tn / (tn + fn)) if (tn + fn) else 0.0,
    }


def evaluate_model(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    outcome: str,
    features: list[str],
    model_name: str,
    split: str,
    c_value: float,
    include_threshold_metrics: bool,
) -> tuple[dict[str, float | int | str], np.ndarray, FittedModel]:
    fitted = fit_model(train_df, outcome, features, c_value)
    prob = predict_proba(fitted, eval_df)
    y_true = eval_df[outcome].to_numpy()
    row: dict[str, float | int | str] = {
        "outcome": OUTCOME_LABELS[outcome],
        "model": model_name,
        "split": split,
        "n": int(len(eval_df)),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "n_features": len(features),
        "C": c_value,
        **classification_metrics(y_true, prob),
    }
    if include_threshold_metrics:
        row.update(threshold_metrics(y_true, prob))
    return row, prob, fitted


def split_summary(splits: dict[str, pd.DataFrame], outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for split, df in splits.items():
        for outcome in outcomes:
            y = df[outcome].to_numpy()
            rows.append(
                {
                    "split": split,
                    "outcome": OUTCOME_LABELS[outcome],
                    "n": int(len(df)),
                    "events": int(y.sum()),
                    "event_rate": float(y.mean()),
                }
            )
    return pd.DataFrame(rows)


def feature_block(feature: str, outcome: str = "metastatic", gwas_panels: dict[str, list[str]] | None = None) -> str:
    if feature in selected_clinical_features(outcome):
        return "clinical_baseline"
    if feature in AST_RESISTANCE:
        return "ast_resistance"
    if feature in FIVE_MARKERS or feature in FIVE_MARKER_BINARY:
        return "five_marker"
    if feature in resolve_gwas_panels(gwas_panels).get(outcome, []):
        return "gwas_features"
    return "other"


def feature_panel_table(
    tiers_by_outcome: dict[str, dict[str, list[str]]],
    gwas_panels: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    rows = []
    for outcome, tiers in tiers_by_outcome.items():
        for tier, features in tiers.items():
            for order, feature in enumerate(features, start=1):
                rows.append(
                    {
                        "outcome": OUTCOME_LABELS[outcome],
                        "model": tier,
                        "feature_order": order,
                        "feature": feature,
                        "feature_block": feature_block(feature, outcome, gwas_panels),
                    }
                )
    return pd.DataFrame(rows)


def roc_curve_table(y_true: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def pr_curve_table(y_true: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.append(thresholds, np.nan)})


def calibration_curve_table(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "prob": prob})
    df["bin"] = pd.qcut(df["prob"], q=min(n_bins, len(df)), duplicates="drop")
    grouped = df.groupby("bin", observed=False)
    return grouped.agg(
        n=("y", "size"),
        events=("y", "sum"),
        mean_predicted_risk=("prob", "mean"),
        observed_event_rate=("y", "mean"),
    ).reset_index(drop=True)


def decision_curve_table(
    y_true: np.ndarray,
    prob_by_model: dict[str, np.ndarray],
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.05, 0.51, 0.01), 2)
    prevalence = float(np.mean(y_true))
    rows = []
    n = len(y_true)
    for threshold in thresholds:
        rows.append(
            {
                "model": "treat_all",
                "threshold": float(threshold),
                "net_benefit": float(prevalence - (1 - prevalence) * threshold / (1 - threshold)),
            }
        )
        rows.append({"model": "treat_none", "threshold": float(threshold), "net_benefit": 0.0})
        for model, prob in prob_by_model.items():
            pred = prob >= threshold
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            net_benefit = tp / n - fp / n * threshold / (1 - threshold)
            rows.append({"model": model, "threshold": float(threshold), "net_benefit": float(net_benefit)})
    return pd.DataFrame(rows)


def paired_bootstrap_metric_delta(
    predictions: pd.DataFrame,
    *,
    split_col: str,
    outcome_col: str,
    reference_col: str,
    comparator_col: str,
    metric_label: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for split in TEST_SPLITS:
        df = predictions[predictions[split_col] == split].reset_index(drop=True)
        y = df[outcome_col].to_numpy()
        ref = df[reference_col].to_numpy()
        comp = df[comparator_col].to_numpy()
        point_deltas = {
            "AUROC": roc_auc_score(y, comp) - roc_auc_score(y, ref),
            "AUPRC": average_precision_score(y, comp) - average_precision_score(y, ref),
            "Brier": brier_score_loss(y, comp) - brier_score_loss(y, ref),
            "LogLoss": log_loss(y, comp) - log_loss(y, ref),
        }
        deltas = {"AUROC": [], "AUPRC": [], "Brier": [], "LogLoss": []}
        for _ in range(n_bootstrap):
            idx = rng.choice(len(df), size=len(df), replace=True)
            y_b = y[idx]
            if y_b.sum() == 0 or y_b.sum() == len(y_b):
                continue
            ref_b = ref[idx]
            comp_b = comp[idx]
            deltas["AUROC"].append(roc_auc_score(y_b, comp_b) - roc_auc_score(y_b, ref_b))
            deltas["AUPRC"].append(average_precision_score(y_b, comp_b) - average_precision_score(y_b, ref_b))
            deltas["Brier"].append(brier_score_loss(y_b, comp_b) - brier_score_loss(y_b, ref_b))
            deltas["LogLoss"].append(log_loss(y_b, comp_b) - log_loss(y_b, ref_b))
        for metric, values in deltas.items():
            arr = np.asarray(values)
            rows.append(
                {
                    "split": split,
                    "comparison": metric_label,
                    "metric": metric,
                    "delta_comparator_minus_reference": float(point_deltas[metric]),
                    "ci95_low": float(np.percentile(arr, 2.5)),
                    "ci95_high": float(np.percentile(arr, 97.5)),
                    "n_bootstrap_valid": int(len(arr)),
                }
            )
    return pd.DataFrame(rows)
