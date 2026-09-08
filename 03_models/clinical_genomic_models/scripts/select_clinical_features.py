"""Training-only, within-domain and across-domain clinical stability selection."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from modeling_contract import C_CANDIDATES, OUTCOME_LABELS, stratified_sample_indices, require_columns

MODULES = {
    "demographics": ["年龄", "男1女2"],
    "comorbidities": [
        "是否有糖尿病（1是0否）", "是否有肝炎（1是0否）", "是否有癌症（1是0否）",
        "高血压（1是0否）", "冠心病（1是0否）", "脑梗死脑出血（1是0否）",
        "肾功能不全（1是0否）", "外伤（1是0否）", "其他疾病",
    ],
    "immune_status": [
        "是否放化疗（1是0否）", "是否激素（1是0否）",
        "是否免疫抑制剂（1是0否）", "器官移植状态（1是0否）",
    ],
    "healthcare_history": ["2-3月前是否入院（1是0否）", "手术史（1是0否）", "血培养前手术（1是0否）"],
    "severity": ["APACHEⅡ评分", "pitt_cont", "SOFA评分"],
    "blood_count": ["WBC-1", "N-1", "HB-1", "PLT-1"],
    "inflammation": ["CRP-1", "PCT-1"],
    "biochemistry": ["ALB-1", "ALT-1", "AST-1", "CHE-1", "Tbil-1", "Cr-1", "INR-1"],
}


CANDIDATES = [f for features in MODULES.values() for f in features]
SEED = 42
CORRELATION_THRESHOLD = 0.95
STABILITY_REPEATS = 80
SUBSAMPLE_FRACTION = 0.8
SELECTION_THRESHOLD = 0.6
L1_RATIOS = [0.5, 1.0]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(out: Path, name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(out / name, index=False, encoding="utf-8-sig")


def validate_training_frame(train: pd.DataFrame, train_end: int = 2021) -> None:
    if train_end != 2021:
        raise ValueError("Clinical selection is restricted to 2013-2021")
    if train.empty or not train["年份"].between(2013, train_end).all():
        raise ValueError(f"Selection and base-model fitting require 2013-{train_end} training data only")


def remove_correlated(train: pd.DataFrame, features: list[str], train_end: int = 2021) -> tuple[list[str], list[dict]]:
    validate_training_frame(train, train_end)
    order = {f: i for i, f in enumerate(features)}
    sorted_features = sorted(features, key=lambda f: (train[f].isna().mean(), order[f]))
    correlations = train[features].corr(method="spearman").abs()
    kept, rows = [], []
    for feature in sorted_features:
        representative = next((other for other in kept if correlations.loc[feature, other] >= CORRELATION_THRESHOLD), None)
        rows.append({"feature": feature, "representative": representative or feature,
                     "excluded": representative is not None,
                     "absolute_spearman": float(correlations.loc[feature, representative]) if representative else None})
        if representative is None:
            kept.append(feature)
    return [f for f in features if f in kept], rows


def elastic_model(c_value: float, l1_ratio: float):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
        LogisticRegression(penalty="elasticnet", solver="saga", C=c_value, l1_ratio=l1_ratio,
                           max_iter=30000, tol=1e-4, random_state=SEED),
    )


def sparse_selection(train: pd.DataFrame, outcome: str, features: list[str], stage: str,
                     repeats: int = STABILITY_REPEATS, train_end: int = 2021) -> tuple[list[str], list[dict], list[dict]]:
    validate_training_frame(train, train_end)
    if not features:
        return [], [], []
    x, y = train[features], train[outcome].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y))
    cv_rows = []
    for c_value in C_CANDIDATES:
        for ratio in L1_RATIOS:
            losses = []
            for fit_idx, score_idx in folds:
                model = elastic_model(c_value, ratio).fit(x.iloc[fit_idx], y[fit_idx])
                losses.append(log_loss(y[score_idx], model.predict_proba(x.iloc[score_idx])[:, 1]))
            cv_rows.append({"outcome": outcome, "stage": stage, "C": c_value, "l1_ratio": ratio,
                            "mean_cv_log_loss": float(np.mean(losses))})
    best = min(cv_rows, key=lambda row: (row["mean_cv_log_loss"], row["C"], -row["l1_ratio"]))
    rng, counts = np.random.default_rng(SEED), np.zeros(len(features), dtype=int)
    for _ in range(repeats):
        indices = stratified_sample_indices(y, SUBSAMPLE_FRACTION, rng)
        model = elastic_model(best["C"], best["l1_ratio"]).fit(x.iloc[indices], y[indices])
        counts += np.abs(model[-1].coef_[0]) > 1e-7
    frequencies = counts / repeats
    selected = [f for f, frequency in zip(features, frequencies) if frequency >= SELECTION_THRESHOLD]
    for row in cv_rows:
        row["selected_setting"] = row["C"] == best["C"] and row["l1_ratio"] == best["l1_ratio"]
    rows = [{"outcome": outcome, "stage": stage, "feature": f, "frequency": float(frequency),
             "selected": f in selected, "C": best["C"], "l1_ratio": best["l1_ratio"]}
            for f, frequency in zip(features, frequencies)]
    print(f"{outcome} / {stage}: {len(features)} -> {len(selected)} features; C={best['C']}, l1={best['l1_ratio']}", flush=True)
    return selected, rows, cv_rows


def select_clinical(train: pd.DataFrame, outcome: str, out: Path, train_end: int = 2021) -> list[str]:
    validate_training_frame(train, train_end)
    retained, stability, tuning, redundancy, eligibility = [], [], [], [], []
    for module, features in MODULES.items():
        eligible = []
        for feature in features:
            usable = train[feature].nunique(dropna=True) > 1
            eligibility.append({"outcome": outcome, "module": module, "feature": feature,
                                "training_missing_fraction": float(train[feature].isna().mean()),
                                "training_variable": usable})
            if usable:
                eligible.append(feature)
        filtered, rows = remove_correlated(train, eligible, train_end)
        redundancy.extend({"outcome": outcome, "stage": module, **r} for r in rows)
        selected, rows, cv = sparse_selection(train, outcome, filtered, module, train_end=train_end)
        retained.extend(selected)
        stability.extend(rows)
        tuning.extend(cv)
    filtered, rows = remove_correlated(train, retained, train_end)
    redundancy.extend({"outcome": outcome, "stage": "across_modules", **r} for r in rows)
    selected, rows, cv = sparse_selection(train, outcome, filtered, "across_modules", train_end=train_end)
    stability.extend(rows)
    tuning.extend(cv)
    if not selected:
        raise ValueError(f"No stable clinical feature for {outcome}; do not silently substitute a panel")
    write_csv(out, f"{outcome}_stability.csv", stability)
    write_csv(out, f"{outcome}_selector_cv.csv", tuning)
    write_csv(out, f"{outcome}_correlation_filter.csv", redundancy)
    for row in eligibility:
        row["selected_clinical_baseline"] = row["feature"] in selected
    write_csv(out, f"{outcome}_feature_decisions.csv", eligibility)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Function\nSelect clinical predictors separately by outcome using training-only two-stage stability selection.\n\nOutput\nAn output directory containing clinical_selection.json and selection audits.", formatter_class=argparse.RawTextHelpFormatter)
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("--input", type=Path, required=True, help="Private analysis-ready numeric CSV; one episode per strain")
    required.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument_group("Optional Arguments").add_argument("--run", action="store_true", help="Execute the selection")
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    raw = pd.read_csv(args.input)
    require_columns(raw, ["strain", "年份", *OUTCOME_LABELS, *CANDIDATES], "clinical selection input")
    if raw.strain.isna().any() or raw.strain.duplicated().any():
        raise ValueError("Expected one nonmissing strain ID per first episode")
    train = raw.loc[raw["年份"].between(2013, 2021)].copy()
    validate_training_frame(train)
    for outcome in OUTCOME_LABELS:
        if not train[outcome].isin([0, 1]).all() or train[outcome].nunique() != 2:
            raise ValueError(f"Training outcome must be observed binary: {outcome}")
    train[CANDIDATES] = train[CANDIDATES].apply(pd.to_numeric, errors="raise")
    if np.isinf(train[CANDIDATES].to_numpy(dtype=float)).any():
        raise ValueError("Infinite clinical predictor")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    panels = {o: select_clinical(train, o, args.output_dir) for o in OUTCOME_LABELS}
    write_json(args.output_dir / "clinical_selection.json", {
        "selected_panels": panels, "training_years": list(range(2013, 2022)),
        "input_sha256": sha256(args.input), "modules": MODULES,
        "resamples": STABILITY_REPEATS, "subsample_fraction": SUBSAMPLE_FRACTION,
        "selection_frequency_threshold": SELECTION_THRESHOLD,
        "status": "complete", "selection_source": "training_only_two_stage_elastic_net",
    })


if __name__ == "__main__":
    main()
