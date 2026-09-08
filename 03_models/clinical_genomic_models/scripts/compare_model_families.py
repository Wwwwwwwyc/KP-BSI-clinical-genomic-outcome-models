"""Compare model families for the full clinical-genomic model."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import pandas as pd
from modeling_contract import (
    AST_RESISTANCE,
    outcome_tiers,
    load_clinical_selection,
    validate_recorded_clinical_panel,
    FIVE_MARKERS,
    OUTCOME_LABELS,
    add_derived_predictors,
    classification_metrics,
    prepare_features,
    project_root,
    require_columns,
    threshold_metrics,
)
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - availability is recorded in the manifest.
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - availability is recorded in the manifest.
    CatBoostClassifier = None

warnings.filterwarnings("ignore", category=FutureWarning)


HELP = """Function
  Compare logistic, elastic-net, XGBoost, CatBoost, and simple ensemble model families for the full clinical-genomic model. The script uses clinical predictors, antimicrobial susceptibility, hypervirulence-associated markers, and validation-selected GWAS features to select a model for liver abscess or metastatic infection on validation years and evaluate it on the temporal test set.

Required Arguments
  --run
      Explicitly confirms execution. Running with no arguments prints this help and exits without doing work.

Optional Arguments
  --input PATH
      Model-ready feature table. Default: 02_features/analysis_datasets/merged_full.csv.
  --information-model-dir PATH
      Upstream nested-model GWAS output directory. Default: 03_models/clinical_genomic_models/model_outputs/nested_information_models.
  --output-dir PATH
      Primary output directory. Default: 03_models/clinical_genomic_models/model_outputs/model_family_comparison.
  --validation-years CSV
      Calendar years used for model-family selection. Discovery uses all earlier years from 2013 onward. Default: 2022,2023.
  --test-years CSV
      Calendar years held out for final testing. Default: 2024,2025.
  --seed N
      Random seed. Default: 42.
  --overwrite
      Allow replacing files inside the output directory.
  -h, --help
      Show this help text.

Output
  Primary output path: --output-dir. The script writes validation_full_model_model_family_grid.csv, selected_full_model_model_family.json, selected_full_model_test_metrics.csv, selected_full_model_test_predictions.csv, selected_model_ladder_test_metrics.csv, selected_model_ladder_test_predictions.csv, feature_panel.csv, and run_manifest.json.
"""


OUTCOME = "metastatic"
PANEL_ID = "full_clinical_genomic_model"


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
class ModelSpec:
    model_id: str
    family: str
    params: dict[str, object]
    factory: Callable[[int], object]


def default_paths() -> Paths:
    root = project_root()
    model_dir = root / "03_models" / "clinical_genomic_models"
    return Paths(
        input_path=root / "02_features" / "analysis_datasets" / "merged_full.csv",
        information_model_dir=model_dir / "model_outputs" / "nested_information_models",
        output_dir=model_dir / "model_outputs" / "model_family_comparison",
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
    min_validation_year = min(validation_years)
    discovery_years = sorted(int(year) for year in df.loc[df["年份"].between(2013, min_validation_year - 1), "年份"].unique())
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


def tier_features(gwas_features: list[str]) -> dict[str, list[str]]:
    return outcome_tiers(OUTCOME, {OUTCOME: gwas_features})


def model_specs(seed: int) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for c_value, class_weight in itertools.product([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0], [None, "balanced"]):
        suffix = "balanced" if class_weight else "unweighted"
        specs.append(
            ModelSpec(
                model_id=f"logistic_l2_C{c_value:g}_{suffix}",
                family="logistic_l2",
                params={"C": c_value, "class_weight": class_weight},
                factory=lambda rs, c_value=c_value, class_weight=class_weight: LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=10000,
                    class_weight=class_weight,
                ),
            )
        )

    for c_value, l1_ratio, class_weight in itertools.product([0.01, 0.02, 0.05, 0.1, 0.2, 0.5], [0.3, 0.5, 0.7, 0.9], [None, "balanced"]):
        suffix = "balanced" if class_weight else "unweighted"
        specs.append(
            ModelSpec(
                model_id=f"elasticnet_C{c_value:g}_l1{l1_ratio:g}_{suffix}",
                family="elasticnet",
                params={"C": c_value, "l1_ratio": l1_ratio, "class_weight": class_weight},
                factory=lambda rs, c_value=c_value, l1_ratio=l1_ratio, class_weight=class_weight: LogisticRegression(
                    C=c_value,
                    penalty="elasticnet",
                    l1_ratio=l1_ratio,
                    solver="saga",
                    max_iter=10000,
                    random_state=rs,
                    class_weight=class_weight,
                ),
            )
        )

    if XGBClassifier is not None:
        for learning_rate, max_depth, reg_lambda, scale_weight in itertools.product([0.03, 0.05], [2, 3], [1.0, 5.0], [1.0, 3.5]):
            specs.append(
                ModelSpec(
                    model_id=f"xgboost_lr{learning_rate:g}_depth{max_depth}_lambda{reg_lambda:g}_sw{scale_weight:g}",
                    family="xgboost",
                    params={"learning_rate": learning_rate, "max_depth": max_depth, "reg_lambda": reg_lambda, "scale_pos_weight": scale_weight},
                    factory=lambda rs, learning_rate=learning_rate, max_depth=max_depth, reg_lambda=reg_lambda, scale_weight=scale_weight: XGBClassifier(
                        n_estimators=300,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        reg_lambda=reg_lambda,
                        scale_pos_weight=scale_weight,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=2,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=rs,
                        n_jobs=-1,
                    ),
                )
            )

    if CatBoostClassifier is not None:
        for learning_rate, depth, l2_leaf_reg, auto_class_weights in itertools.product([0.03, 0.05], [2, 3], [3.0, 10.0], [None, "Balanced"]):
            suffix = "balanced" if auto_class_weights else "unweighted"
            specs.append(
                ModelSpec(
                    model_id=f"catboost_lr{learning_rate:g}_depth{depth}_l2{l2_leaf_reg:g}_{suffix}",
                    family="catboost",
                    params={"learning_rate": learning_rate, "depth": depth, "l2_leaf_reg": l2_leaf_reg, "auto_class_weights": auto_class_weights},
                    factory=lambda rs, learning_rate=learning_rate, depth=depth, l2_leaf_reg=l2_leaf_reg, auto_class_weights=auto_class_weights: CatBoostClassifier(
                        iterations=300,
                        learning_rate=learning_rate,
                        depth=depth,
                        l2_leaf_reg=l2_leaf_reg,
                        auto_class_weights=auto_class_weights,
                        loss_function="Logloss",
                        random_seed=rs,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                )
            )
    return specs


def fit_predict(spec: ModelSpec, train_df: pd.DataFrame, eval_df: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    x_train, preprocessor = prepare_features(train_df, features, fit=True)
    x_eval, _ = prepare_features(eval_df, features, preprocessor=preprocessor)
    model = spec.factory(seed)
    model.fit(x_train, train_df[OUTCOME].to_numpy())
    return np.asarray(model.predict_proba(x_eval)[:, 1], dtype=float)


def metric_row(
    tier: str,
    model_id: str,
    family: str,
    split: str,
    y_true: np.ndarray,
    prob: np.ndarray,
    n_features: int,
    params: dict[str, object],
) -> dict[str, object]:
    return {
        "panel_id": PANEL_ID,
        "outcome": OUTCOME_LABELS[OUTCOME],
        "tier": tier,
        "model_id": model_id,
        "family": family,
        "split": split,
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "n_features": n_features,
        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        **classification_metrics(y_true, prob),
    }


def component_ids(row: pd.Series) -> list[str]:
    if row["family"] != "soft_voting":
        return [str(row["model_id"])]
    params = json.loads(str(row["params_json"]))
    return [str(model_id) for model_id in params["component_model_ids"]]


def add_ensembles(grid: pd.DataFrame, predictions: dict[str, np.ndarray], y_true: np.ndarray, n_features: int, split_label: str) -> pd.DataFrame:
    rows = []
    family_best = (
        grid.sort_values(["AUROC", "Brier"], ascending=[False, True], kind="mergesort")
        .drop_duplicates("family", keep="first")
    )
    best_by_family = {row.family: row.model_id for row in family_best.itertuples()}
    ensemble_specs = {
        "softvote_logistic_elasticnet": ["logistic_l2", "elasticnet"],
        "softvote_logistic_xgboost": ["logistic_l2", "xgboost"],
        "softvote_elasticnet_xgboost": ["elasticnet", "xgboost"],
        "softvote_xgboost_catboost": ["xgboost", "catboost"],
        "softvote_all_available": ["logistic_l2", "elasticnet", "xgboost", "catboost"],
    }
    for model_id, families in ensemble_specs.items():
        component_model_ids = [best_by_family[family] for family in families if family in best_by_family]
        if len(component_model_ids) < 2:
            continue
        prob = np.mean([predictions[component_id] for component_id in component_model_ids], axis=0)
        rows.append(
            metric_row(
                "clinical_ast_markers_gwas",
                model_id,
                "soft_voting",
                split_label,
                y_true,
                prob,
                n_features,
                {"component_model_ids": component_model_ids},
            )
        )
    return pd.concat([grid, pd.DataFrame(rows)], ignore_index=True)


def screen_validation_full_model(specs: list[ModelSpec], discovery: pd.DataFrame, validation: pd.DataFrame, features: list[str], seed: int, split_label: str) -> pd.DataFrame:
    y_validation = validation[OUTCOME].to_numpy()
    rows = []
    predictions = {}
    for index, spec in enumerate(specs):
        prob = fit_predict(spec, discovery, validation, features, seed + index)
        predictions[spec.model_id] = prob
        rows.append(metric_row("clinical_ast_markers_gwas", spec.model_id, spec.family, split_label, y_validation, prob, len(features), spec.params))
    return add_ensembles(pd.DataFrame(rows), predictions, y_validation, len(features), split_label)


def spec_lookup(specs: list[ModelSpec]) -> dict[str, ModelSpec]:
    return {spec.model_id: spec for spec in specs}


def predict_model(row: pd.Series, specs: dict[str, ModelSpec], train_df: pd.DataFrame, eval_df: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    probs = [fit_predict(specs[model_id], train_df, eval_df, features, seed) for model_id in component_ids(row)]
    return np.mean(probs, axis=0)


def select_full_model(validation_grid: pd.DataFrame) -> tuple[pd.Series, dict[str, float | int | str]]:
    event_rate = float(validation_grid["event_rate"].iloc[0])
    no_information_brier = event_rate * (1 - event_rate)
    eligible = validation_grid[validation_grid["Brier"] <= no_information_brier].copy()
    if eligible.empty:
        eligible = validation_grid.copy()
    best_eligible_auc = float(eligible["AUROC"].max())
    near_auc = eligible[eligible["AUROC"] >= best_eligible_auc - 0.01].copy()
    family_priority = {"logistic_l2": 0, "elasticnet": 1, "catboost": 2, "xgboost": 3, "soft_voting": 4}
    near_auc["family_priority"] = near_auc["family"].map(family_priority).fillna(99)
    selected = near_auc.sort_values(
        ["family_priority", "Brier", "AUROC", "model_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).iloc[0]
    return selected, {
        "validation_event_rate": event_rate,
        "validation_no_information_brier": no_information_brier,
        "eligible_model_count": int(len(eligible)),
        "near_auc_model_count": int(len(near_auc)),
        "total_model_count": int(len(validation_grid)),
        "near_auc_tolerance": 0.01,
        "family_priority": "logistic_l2, elasticnet, catboost, xgboost, soft_voting",
    }


def selected_ladder_metrics(
    selected: pd.Series,
    specs: dict[str, ModelSpec],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tiers: dict[str, list[str]],
    seed: int,
    test_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = test_df[["strain", "年份"]].copy() if "strain" in test_df.columns else test_df[["年份"]].copy()
    y_test = test_df[OUTCOME].to_numpy()
    predictions["outcome"] = OUTCOME_LABELS[OUTCOME]
    predictions["split"] = test_label
    predictions["observed"] = y_test
    for tier, features in tiers.items():
        prob = predict_model(selected, specs, train_df, test_df, features, seed)
        row = metric_row(tier, str(selected["model_id"]), str(selected["family"]), test_label, y_test, prob, len(features), json.loads(str(selected["params_json"])))
        validation_prob = predict_model(selected, specs, train_df, validation_df, features, seed)
        threshold = threshold_metrics(validation_df[OUTCOME].to_numpy(), validation_prob)["threshold_youden"]
        row.update(threshold_metrics(y_test, prob, threshold=threshold))
        row["threshold_source"] = "validation"
        rows.append(row)
        predictions[f"{tier}_prob"] = prob
    out = pd.DataFrame(rows)
    marker_model_auc = float(out.loc[out["tier"] == "clinical_ast_markers", "AUROC"].iloc[0])
    full_model_auc = float(out.loc[out["tier"] == "clinical_ast_markers_gwas", "AUROC"].iloc[0])
    out["delta_AUROC_vs_marker_model"] = np.where(out["tier"] == "clinical_ast_markers_gwas", full_model_auc - marker_model_auc, np.nan)
    return out, predictions


def feature_panel_table(tiers: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "panel_id": PANEL_ID,
            "outcome": OUTCOME_LABELS[OUTCOME],
            "tier": tier,
            "feature_order": order,
            "feature": feature,
        }
        for tier, features in tiers.items()
        for order, feature in enumerate(features, start=1)
    )


def run(paths: Paths, args: argparse.Namespace) -> None:
    raw = pd.read_csv(paths.input_path, low_memory=False)
    require_columns(raw, ["年份", OUTCOME], "merged feature table")
    merged = add_derived_predictors(raw)
    gwas_markers = read_locked_gwas_panel(paths.information_model_dir)
    tiers = tier_features(gwas_markers)
    for tier, features in tiers.items():
        require_columns(merged, features, f"{tier} feature panel")

    splits, split_config = temporal_splits(merged, args.validation_years, args.test_years)
    discovery = splits[split_config.discovery_label]
    validation = splits[split_config.validation_label]
    test = splits[split_config.test_label]
    specs = model_specs(args.seed)
    validation_grid = screen_validation_full_model(specs, discovery, validation, tiers["clinical_ast_markers_gwas"], args.seed, split_config.validation_label)
    selected, selection_audit = select_full_model(validation_grid)
    validation_grid["selected_for_test"] = validation_grid["model_id"].eq(selected["model_id"])
    validation_grid["passes_brier_gate"] = validation_grid["Brier"] <= float(selection_audit["validation_no_information_brier"])

    lookup = spec_lookup(specs)
    full_model_prob = predict_model(selected, lookup, discovery, test, tiers["clinical_ast_markers_gwas"], args.seed)
    y_test = test[OUTCOME].to_numpy()
    selected_test = metric_row(
        "clinical_ast_markers_gwas",
        str(selected["model_id"]),
        str(selected["family"]),
        split_config.test_label,
        y_test,
        full_model_prob,
        len(tiers["clinical_ast_markers_gwas"]),
        json.loads(str(selected["params_json"])),
    )
    validation_prob = predict_model(selected, lookup, discovery, validation, tiers["clinical_ast_markers_gwas"], args.seed)
    threshold = threshold_metrics(validation[OUTCOME].to_numpy(), validation_prob)["threshold_youden"]
    selected_test.update(threshold_metrics(y_test, full_model_prob, threshold=threshold))
    selected_test["threshold_source"] = split_config.validation_label
    predictions = test[["strain", "年份"]].copy() if "strain" in test.columns else test[["年份"]].copy()
    predictions["outcome"] = OUTCOME_LABELS[OUTCOME]
    predictions["split"] = split_config.test_label
    predictions["observed"] = y_test
    predictions["full_model_prob"] = full_model_prob

    ladder_metrics, ladder_predictions = selected_ladder_metrics(selected, lookup, discovery, validation, test, tiers, args.seed, split_config.test_label)

    validation_grid.to_csv(paths.output_dir / "validation_full_model_model_family_grid.csv", index=False)
    pd.DataFrame([selected.to_dict()]).to_csv(paths.output_dir / "selected_full_model_model_family.csv", index=False)
    pd.DataFrame([selected_test]).to_csv(paths.output_dir / "selected_full_model_test_metrics.csv", index=False)
    predictions.to_csv(paths.output_dir / "selected_full_model_test_predictions.csv", index=False)
    ladder_metrics.to_csv(paths.output_dir / "selected_model_ladder_test_metrics.csv", index=False)
    ladder_predictions.to_csv(paths.output_dir / "selected_model_ladder_test_predictions.csv", index=False)
    feature_panel_table(tiers).to_csv(paths.output_dir / "feature_panel.csv", index=False)
    with (paths.output_dir / "selected_full_model_model_family.json").open("w", encoding="utf-8") as handle:
        json.dump(selected.to_dict(), handle, ensure_ascii=False, indent=2)

    manifest = {
        "input_path": str(paths.input_path),
        "information_model_dir": str(paths.information_model_dir),
        "output_dir": str(paths.output_dir),
        "panel_id": PANEL_ID,
        "gwas_markers": gwas_markers,
        "outcome": OUTCOME_LABELS[OUTCOME],
        "selection_metric": "Brier gate against validation no-information risk; then models within 0.01 validation AUROC of the best eligible model; then simpler family priority",
        "selection_audit": selection_audit,
        "discovery_split": split_config.discovery_label,
        "validation_split": split_config.validation_label,
        "test_split": split_config.test_label,
        "test_model_training_split": split_config.discovery_label,
        "test_model_training_years": split_config.discovery_years,
        "discovery_years": split_config.discovery_years,
        "validation_years": split_config.validation_years,
        "test_years": split_config.test_years,
        "xgboost_available": XGBClassifier is not None,
        "catboost_available": CatBoostClassifier is not None,
        "candidate_model_count_before_ensembles": len(specs),
        "selected_model_id": str(selected["model_id"]),
        "selected_family": str(selected["family"]),
        "generated_files": sorted({path.name for path in paths.output_dir.iterdir() if path.is_file()} | {"run_manifest.json"}),
    }
    with (paths.output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


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
    print(f"Wrote model-family comparison to: {paths.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
