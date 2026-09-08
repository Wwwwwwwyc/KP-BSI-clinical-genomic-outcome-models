"""Construct nested clinical-genomic models and select GWAS features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from modeling_contract import (
    AST_RESISTANCE,
    C_CANDIDATES,
    outcome_tiers,
    load_clinical_selection,
    selected_clinical_features,
    FIVE_MARKERS,
    OUTCOME_LABELS,
    add_derived_predictors,
    evaluate_model,
    fit_model,
    gene_base,
    project_root,
    require_columns,
    predict_proba,
    threshold_metrics,
    tune_c,
)


HELP = """Function
  Construct nested KP-BSI clinical-genomic models for 30-day mortality and liver abscess or metastatic infection. Candidate pangenome features are screened by population-structure-adjusted pyseer association results, deduplicated by correlation structure, ranked by SHAP/pruning, and selected on the 2022-2023 validation set before temporal testing in 2024-2025.

Required Arguments
  --run
      Explicitly confirms execution. Running with no arguments prints this help and exits without doing work.

Optional Arguments
  --input PATH
      Model-ready feature table. Default: 02_features/analysis_datasets/merged_full.csv.
  --clinical-selection PATH
      Training-only clinical_selection.json; omission uses the frozen Table S2 lists.
  --metastatic-pyseer PATH
      Training-cohort pyseer output for liver abscess or metastatic infection.
  --death-pyseer PATH
      Training-cohort pyseer output for 30-day mortality. Default: 03_models/clinical_genomic_models/gwas_inputs/mortality/pyseer.tsv.
  --max-gwas-panel-size N
      Largest validation-compared GWAS top-k panel for each outcome. Default: 6.
  --output-dir PATH
      Primary output directory. Default: 03_models/clinical_genomic_models/model_outputs/nested_information_models.
  --validation-years CSV
      Calendar years used for C tuning and validation reporting. Discovery uses all earlier years from 2013 onward. Default: 2022,2023.
  --test-years CSV
      Calendar years held out for final testing. Default: 2024,2025.
  --overwrite
      Allow replacing files inside the output directory.
  -h, --help
      Show this help text.

Output
  Primary output path: --output-dir. The script writes gwas_candidate_pool.csv, gwas_module_contract.csv, gwas_pruning_trace.csv, gwas_shap_pruning_rank.csv, gwas_topk_panel_comparison.csv, gwas_feature_panel.csv, ladder_metrics.csv, ladder_deltas.csv, ladder_predictions.csv, split_summary.csv, run_manifest.json, and README.md.
"""


OUTCOMES = ["death_30d", "metastatic"]
STRICT_GWAS_LRT_PVALUE = 0.005
MODULE_CORR_ABS_THRESHOLD = 0.95
MODEL_ORDER = [
    "clinical",
    "clinical_ast",
    "clinical_ast_markers",
    "clinical_ast_markers_gwas",
]


@dataclass(frozen=True)
class Paths:
    input_path: Path
    metastatic_pyseer: Path
    death_pyseer: Path
    output_dir: Path


@dataclass(frozen=True)
class SplitConfig:
    discovery_label: str
    validation_label: str
    test_label: str
    discovery_years: list[int]
    validation_years: list[int]
    test_years: list[int]


def default_paths() -> Paths:
    root = project_root()
    model_dir = root / "03_models" / "clinical_genomic_models"
    metastatic_input_dir = model_dir / "gwas_inputs" / "invasive_phenotype"
    gwas_dir = model_dir / "gwas_inputs" / "mortality"
    return Paths(
        input_path=root / "02_features" / "analysis_datasets" / "merged_full.csv",
        metastatic_pyseer=metastatic_input_dir / "pyseer.tsv",
        death_pyseer=gwas_dir / "pyseer.tsv",
        output_dir=model_dir / "model_outputs" / "nested_information_models",
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
    parser.add_argument("--clinical-selection", type=Path)
    parser.add_argument("--input", type=Path, default=defaults.input_path)
    parser.add_argument("--metastatic-pyseer", type=Path, default=defaults.metastatic_pyseer)
    parser.add_argument("--death-pyseer", type=Path, default=defaults.death_pyseer)
    parser.add_argument("--max-gwas-panel-size", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--validation-years", type=lambda x: parse_year_csv(x, "validation-years"), default=[2022, 2023])
    parser.add_argument("--test-years", type=lambda x: parse_year_csv(x, "test-years"), default=[2024, 2025])
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
    if args.validation_years != [2022, 2023] or args.test_years != [2024, 2025]:
        raise SystemExit("The manuscript workflow locks training 2013-2021, validation 2022-2023 and test 2024-2025")
    if set(args.validation_years) & set(args.test_years):
        raise SystemExit("--validation-years and --test-years must not overlap")
    if not 1 <= args.max_gwas_panel_size <= 6:
        raise SystemExit("--max-gwas-panel-size must be between 1 and 6")
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else project_root() / path


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing_files = [item for item in path.iterdir() if item.is_file()]
    if existing_files and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite or choose a new --output-dir.")
    if overwrite:
        for item in existing_files:
            item.unlink()


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


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    pvalues = pd.to_numeric(values, errors="coerce")
    qvalues = pd.Series(np.nan, index=values.index, dtype=float)
    finite = pvalues.gt(0) & np.isfinite(pvalues)
    if not finite.any():
        return qvalues
    ranked = pvalues[finite].sort_values(kind="mergesort")
    n = len(ranked)
    adjusted = ranked.to_numpy(dtype=float) * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    qvalues.loc[ranked.index] = np.clip(adjusted, 0, 1)
    return qvalues


def alias_members(variant: str) -> list[str]:
    return [part.strip() for part in str(variant).split("~~~") if part.strip()]


def map_variant_to_features(variant: str, available_features: set[str]) -> list[str]:
    mapped: list[str] = []
    if variant in available_features:
        mapped.append(variant)
    for member in alias_members(variant):
        if member in available_features and member not in mapped:
            mapped.append(member)
    return mapped


def read_pyseer_candidate_pool(path: Path, outcome: str, available_features: set[str]) -> pd.DataFrame:
    required = ["variant", "af", "filter-pvalue", "lrt-pvalue", "beta", "beta-std-err", "variant_h2"]
    table = pd.read_csv(path, sep="\t")
    require_columns(table, required, f"pyseer output for {OUTCOME_LABELS[outcome]}")

    source = table.copy()
    for column in ["af", "filter-pvalue", "lrt-pvalue", "beta", "beta-std-err", "variant_h2"]:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source["source_variant"] = source["variant"].astype(str)
    source["source_lrt_qvalue"] = benjamini_hochberg(source["lrt-pvalue"])

    rows: list[dict[str, object]] = []
    ordered_source = source.sort_values(["lrt-pvalue", "filter-pvalue", "source_variant"], kind="mergesort")
    for source_rank, (_, row) in enumerate(ordered_source.iterrows(), start=1):
        variant = str(row["source_variant"])
        allele_frequency = float(row["af"])
        filter_pvalue = float(row["filter-pvalue"])
        lrt_pvalue = float(row["lrt-pvalue"])
        beta = float(row["beta"])
        if not (0.05 <= allele_frequency <= 0.95):
            continue
        if not (filter_pvalue <= 0.05):
            continue
        if not (lrt_pvalue > 0 and np.isfinite(lrt_pvalue) and np.isfinite(beta)):
            continue
        for feature in map_variant_to_features(variant, available_features):
            if feature in marker_model_features(outcome):
                continue
            rows.append(
                {
                    "outcome": OUTCOME_LABELS[outcome],
                    "source_rank_by_lrt": source_rank,
                    "source_variant": variant,
                    "feature": feature,
                    "gene_base": gene_base(feature),
                    "mapped_from_alias": feature != variant,
                    "source_is_panaroo_alias": "~~~" in variant,
                    "source_members": "|".join(alias_members(variant)),
                    "is_unnamed_group": feature.startswith("group_"),
                    "allele_frequency": allele_frequency,
                    "filter_pvalue": filter_pvalue,
                    "lrt_pvalue": lrt_pvalue,
                    "lrt_qvalue": float(row["source_lrt_qvalue"]) if np.isfinite(row["source_lrt_qvalue"]) else np.nan,
                    "beta": beta,
                    "beta_std_error": float(row["beta-std-err"]),
                    "variant_h2": float(row["variant_h2"]),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "outcome",
                "candidate_rank",
                "source_rank_by_lrt",
                "source_variant",
                "feature",
                "gene_base",
                "mapped_from_alias",
                "source_is_panaroo_alias",
                "source_members",
                "is_unnamed_group",
                "allele_frequency",
                "filter_pvalue",
                "lrt_pvalue",
                "lrt_qvalue",
                "beta",
                "beta_std_error",
                "variant_h2",
            ]
        )
    candidates = pd.DataFrame(rows)
    candidates["source_priority"] = np.select(
        [
            (~candidates["mapped_from_alias"]) & candidates["lrt_pvalue"].le(0.05),
            candidates["mapped_from_alias"] & candidates["lrt_pvalue"].le(0.05),
            ~candidates["mapped_from_alias"],
        ],
        [0, 1, 2],
        default=3,
    )
    candidates = candidates.sort_values(
        ["feature", "source_priority", "lrt_pvalue", "filter_pvalue", "source_variant"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).drop_duplicates("feature", keep="first").drop(columns=["source_priority"])
    candidates = candidates.sort_values(["lrt_pvalue", "filter_pvalue", "feature"], kind="mergesort").reset_index(drop=True)
    candidates.insert(1, "candidate_rank", np.arange(1, len(candidates) + 1))
    return candidates




def build_outcome_module_contract(
    candidate_pool: pd.DataFrame,
    discovery: pd.DataFrame,
    outcome: str,
    *,
    selected_panel: list[str] | None = None,
    corr_threshold: float = MODULE_CORR_ABS_THRESHOLD,
) -> pd.DataFrame:
    if discovery.empty or not discovery["年份"].between(2013, 2021).all():
        raise ValueError("GWAS modules require 2013-2021 training data only")
    label = OUTCOME_LABELS[outcome]
    evidence = candidate_pool[candidate_pool["outcome"].eq(label)].drop_duplicates("feature").copy()
    if evidence.empty:
        raise ValueError(f"No strict GWAS candidates available for {label}")
    features = evidence["feature"].tolist()
    require_columns(discovery, features, f"{label} GWAS module de-redundancy")

    matrix = discovery[features].apply(pd.to_numeric, errors="raise")
    if not matrix.isin([0, 1]).all().all():
        raise ValueError("GWAS module input must be complete binary presence/absence")
    values = matrix.to_numpy(dtype=float)
    if values.shape[1] == 1:
        corr = np.eye(1)
    else:
        corr = np.corrcoef(values, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)

    parent = list(range(len(features)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if abs(float(corr[i, j])) >= corr_threshold:
                union(i, j)

    clusters: dict[int, list[str]] = {}
    for idx, feature in enumerate(features):
        clusters.setdefault(find(idx), []).append(feature)

    selected_panel = selected_panel or []
    rows: list[dict[str, object]] = []
    evidence_by_feature = evidence.set_index("feature", drop=False)
    for genes in clusters.values():
        ranked = (
            evidence[evidence["feature"].isin(genes)]
            .sort_values(
                ["lrt_pvalue", "filter_pvalue", "is_unnamed_group", "mapped_from_alias", "feature"],
                ascending=[True, True, True, True, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        representative = str(ranked.loc[0, "feature"])
        rep = evidence_by_feature.loc[representative]
        ordered_genes = ranked["feature"].astype(str).tolist()
        rows.append(
            {
                "outcome": label,
                "n_genes": len(ordered_genes),
                "representative": representative,
                "rep_prevalence": float(pd.to_numeric(discovery[representative], errors="coerce").fillna(0.0).mean()),
                "rep_pval": float(rep["lrt_pvalue"]),
                "all_genes": ", ".join(ordered_genes),
                "all_genes_pipe": "|".join(ordered_genes),
                "candidate_set_role": "module_representative",
                "contains_final_panel_marker": any(feature in selected_panel for feature in ordered_genes),
                "source_variant": rep["source_variant"],
                "lrt_pvalue": float(rep["lrt_pvalue"]),
                "lrt_qvalue": float(rep["lrt_qvalue"]) if pd.notna(rep["lrt_qvalue"]) else np.nan,
                "filter_pvalue": float(rep["filter_pvalue"]),
                "beta": float(rep["beta"]),
                "mapped_from_alias": bool(rep["mapped_from_alias"]),
                "source_is_panaroo_alias": bool(rep["source_is_panaroo_alias"]),
                "selected_final_panel": representative in selected_panel,
            }
        )

    out = pd.DataFrame(rows).sort_values(["lrt_pvalue", "filter_pvalue", "representative"], kind="mergesort").reset_index(drop=True)
    out.insert(1, "module_id", np.arange(1, len(out) + 1))
    if selected_panel:
        out["candidate_set_role"] = np.where(out["representative"].isin(selected_panel), "final_panel_representative", out["candidate_set_role"])
    return out


def within_module_ablation(module_contract: pd.DataFrame, discovery: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Training-CV comparisons audit the LRT-chosen representative without test selection."""
    if discovery.empty or not discovery["年份"].between(2013, 2021).all():
        raise ValueError("Within-module comparisons require training records only")
    representatives = module_contract.representative.tolist()
    baseline = marker_model_features(outcome)
    c_value = tune_c(discovery, outcome, baseline + representatives)
    y = discovery[outcome].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    def score(features):
        predictions = np.zeros(len(y))
        for train_index, held_index in folds:
            fitted = fit_model(discovery.iloc[train_index], outcome, features, c_value)
            predictions[held_index] = predict_proba(fitted, discovery.iloc[held_index])
        return {"AUROC": roc_auc_score(y, predictions), "Brier": brier_score_loss(y, predictions), "LogLoss": log_loss(y, predictions)}

    full_metrics = score(baseline + representatives)
    rows = []
    for module in module_contract.itertuples():
        others = [f for f in representatives if f != module.representative]
        for member in [None, *module.all_genes_pipe.split("|")]:
            metrics = full_metrics if member == module.representative else score(baseline + others + ([member] if member else []))
            rows.append({"outcome": OUTCOME_LABELS[outcome], "representative": module.representative,
                         "candidate": member, "comparison": "drop_module" if member is None else "replace_representative",
                         "selected_by_lrt": member == module.representative, "C": c_value,
                         "evaluation": "training_5fold_CV", **metrics,
                         **{f"delta_{m}_vs_representatives": metrics[m] - full_metrics[m] for m in metrics}})
    return pd.DataFrame(rows)


def linear_shap_pruning_rank(
    module_contract: pd.DataFrame,
    discovery: pd.DataFrame,
    outcome: str,
    max_panel_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label = OUTCOME_LABELS[outcome]
    representatives = module_contract["representative"].drop_duplicates().astype(str).tolist()
    require_columns(discovery, [*marker_model_features(outcome), *representatives], f"{label} SHAP/pruning")
    c_value = tune_c(discovery, outcome, [*marker_model_features(outcome), *representatives])
    fitted = fit_model(discovery, outcome, [*marker_model_features(outcome), *representatives], c_value)
    transformed = fitted.preprocessor.transform(discovery)
    coef = fitted.model.coef_[0]
    owners = np.asarray(fitted.preprocessor.transformed_to_feature)

    shap_rows = []
    for feature in representatives:
        cols = np.flatnonzero(owners == feature)
        if len(cols) == 0:
            mean_abs_shap = 0.0
        else:
            contribution = transformed[:, cols] @ coef[cols]
            mean_abs_shap = float(np.mean(np.abs(contribution)))
        shap_rows.append({"feature": feature, "mean_abs_shap": mean_abs_shap})
    shap = pd.DataFrame(shap_rows)

    out = module_contract.merge(shap, left_on="representative", right_on="feature", how="left").drop(columns=["feature"])
    out = out.sort_values(["mean_abs_shap", "lrt_pvalue", "representative"], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)
    out.insert(1, "shap_pruning_rank", np.arange(1, len(out) + 1))
    out["included_in_topk_scope"] = out["shap_pruning_rank"].le(max_panel_size)
    out["ranking_source"] = "linear_logit_shap"

    removal_order = out.sort_values(["mean_abs_shap", "lrt_pvalue", "representative"], ascending=[True, False, True], kind="mergesort")[
        "representative"
    ].tolist()
    retained = representatives.copy()
    prune_rows: list[dict[str, object]] = []

    def append_prune_row(step: int, removed: str) -> None:
        metrics, _, _ = evaluate_model(
            discovery,
            discovery,
            outcome,
            [*marker_model_features(outcome), *retained],
            "clinical_ast_markers_gwas",
            "training_pruning_trace",
            c_value,
            include_threshold_metrics=False,
        )
        prune_rows.append(
            {
                "outcome": label,
                "step": step,
                "removed": removed,
                "n_markers": len(retained),
                "brier": metrics["Brier"],
                "bss": np.nan,
                "auroc": metrics["AUROC"],
                "retained_features_after_step": "|".join(retained),
                "selection_stage": "training_set_shap_pruning",
            }
        )

    append_prune_row(0, "none")
    stop_size = min(max_panel_size, len(retained))
    step = 1
    for feature in removal_order:
        if len(retained) <= stop_size:
            break
        if feature in retained:
            retained.remove(feature)
            append_prune_row(step, feature)
            step += 1

    pruning_trace = pd.DataFrame(prune_rows)
    removed_step = {row["removed"]: int(row["step"]) for row in prune_rows if row["removed"] != "none"}
    out["pruning_removed_step"] = out["representative"].map(removed_step)
    out["n_markers_after_removal"] = out["pruning_removed_step"].map(
        lambda value: int(len(representatives) - value) if pd.notna(value) else np.nan
    )
    out["pruning_brier_after_removal"] = out["pruning_removed_step"].map(
        pruning_trace.set_index("step")["brier"].to_dict()
    )
    out["pruning_bss_after_removal"] = np.nan
    out["pruning_auroc_after_removal"] = out["pruning_removed_step"].map(
        pruning_trace.set_index("step")["auroc"].to_dict()
    )
    out["survived_full_pruning"] = out["pruning_removed_step"].isna()
    out = out.rename(columns={"representative": "feature"})
    return out, pruning_trace








def select_validation_panel(topk_comparison: pd.DataFrame) -> tuple[list[str], pd.Series]:
    best_auroc = float(topk_comparison["AUROC"].max())
    eligible = topk_comparison[np.isclose(topk_comparison["AUROC"], best_auroc, rtol=0, atol=1e-12)].copy()
    selected = eligible.sort_values(
        ["panel_size", "Brier", "AUPRC", "C"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).iloc[0]
    features = str(selected["gwas_features"]).split("|")
    return features, selected


def topk_panel_comparison(discovery: pd.DataFrame, validation: pd.DataFrame, ranked_markers: list[str], outcome: str) -> pd.DataFrame:
    rows = []
    for panel_size in range(1, len(ranked_markers) + 1):
        gwas_features = ranked_markers[:panel_size]
        features = [*marker_model_features(outcome), *gwas_features]
        for c_value in C_CANDIDATES:
            row, _, _ = evaluate_model(
                discovery,
                validation,
                outcome,
                features,
                "clinical_ast_markers_gwas",
                "validation_topk_panel_comparison",
                c_value,
                include_threshold_metrics=False,
            )
            rows.append(
                {
                    "outcome": OUTCOME_LABELS[outcome],
                    "panel_size": panel_size,
                    "gwas_features": "|".join(gwas_features),
                    "C": c_value,
                    "AUROC": row["AUROC"],
                    "AUPRC": row["AUPRC"],
                    "Brier": row["Brier"],
                    "LogLoss": row["LogLoss"],
                }
            )
    out = pd.DataFrame(rows)
    out["best_validation_AUROC"] = out["AUROC"].eq(out["AUROC"].max())
    selected_panel, selected_row = select_validation_panel(out)
    out["selected_by_validation"] = out["gwas_features"].eq("|".join(selected_panel)) & out["C"].eq(selected_row["C"])
    return out.sort_values(["AUROC", "Brier", "panel_size", "C"], ascending=[False, True, True, True], kind="mergesort")


def tune_c_on_validation(discovery: pd.DataFrame, validation: pd.DataFrame, outcome: str, features: list[str], model_name: str) -> float:
    rows = []
    for c_value in C_CANDIDATES:
        metrics, _, _ = evaluate_model(
            discovery,
            validation,
            outcome,
            features,
            model_name,
            "validation_tuning",
            c_value,
            include_threshold_metrics=False,
        )
        rows.append((c_value, metrics))
    best_c, _ = sorted(rows, key=lambda item: (-float(item[1]["AUROC"]), float(item[1]["Brier"]), item[0]))[0]
    return float(best_c)


def marker_model_features(outcome: str) -> list[str]:
    return [*selected_clinical_features(outcome), *AST_RESISTANCE, *FIVE_MARKERS]


def tiers_for_panel(gwas_features: list[str], outcome: str) -> dict[str, list[str]]:
    return outcome_tiers(outcome, {outcome: gwas_features})


def evaluate_ladder(
    splits: dict[str, pd.DataFrame],
    config: SplitConfig,
    outcome: str,
    gwas_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    discovery = splits[config.discovery_label]
    validation = splits[config.validation_label]
    test = splits[config.test_label]
    tiers = tiers_for_panel(gwas_features, outcome)

    tuned_c = {
        model_name: tune_c_on_validation(discovery, validation, outcome, features, model_name)
        for model_name, features in tiers.items()
    }
    rows = []
    predictions = []
    validation_thresholds = {}
    for split_name, eval_df, train_df, include_threshold in [
        (config.validation_label, validation, discovery, False),
        (config.test_label, test, discovery, True),
    ]:
        pred = eval_df[[col for col in ["strain", "年份"] if col in eval_df.columns]].copy()
        pred["outcome"] = OUTCOME_LABELS[outcome]
        pred["split"] = split_name
        pred["observed"] = eval_df[outcome].to_numpy()
        for model_name, features in tiers.items():
            row, prob, _ = evaluate_model(
                train_df,
                eval_df,
                outcome,
                features,
                model_name,
                split_name,
                tuned_c[model_name],
                include_threshold_metrics=False,
            )
            if split_name == config.validation_label:
                validation_thresholds[model_name] = threshold_metrics(validation[outcome].to_numpy(), prob)["threshold_youden"]
            if include_threshold:
                row.update(threshold_metrics(eval_df[outcome].to_numpy(), prob, threshold=validation_thresholds[model_name]))
                row["threshold_source"] = config.validation_label
            row["c_strategy"] = "validation_auroc_tuned"
            rows.append(row)
            pred[f"{model_name}_prob"] = prob
        predictions.append(pred)

    metrics = pd.DataFrame(rows)
    delta_rows = []
    for keys, table in metrics.groupby(["outcome", "split"], sort=False):
        indexed = table.set_index("model")
        for left, right in zip(MODEL_ORDER[:-1], MODEL_ORDER[1:]):
            delta_rows.append(
                {
                    "outcome": keys[0],
                    "split": keys[1],
                    "comparison": f"{left}_to_{right}",
                    "delta_AUROC": float(indexed.loc[right, "AUROC"] - indexed.loc[left, "AUROC"]),
                    "delta_AUPRC": float(indexed.loc[right, "AUPRC"] - indexed.loc[left, "AUPRC"]),
                    "delta_Brier": float(indexed.loc[right, "Brier"] - indexed.loc[left, "Brier"]),
                    "delta_LogLoss": float(indexed.loc[right, "LogLoss"] - indexed.loc[left, "LogLoss"]),
                }
            )

    panel_rows = []
    for model_name, features in tiers.items():
        for order, feature in enumerate(features, start=1):
            panel_rows.append(
                {
                    "outcome": OUTCOME_LABELS[outcome],
                    "model": model_name,
                    "feature_order": order,
                    "feature": feature,
                    "feature_block": "gwas_features" if feature in gwas_features else "pre_gwas_feature",
                }
            )
    return metrics, pd.DataFrame(delta_rows), pd.concat(predictions, ignore_index=True), pd.DataFrame(panel_rows)


def split_summary(splits: dict[str, pd.DataFrame], config: SplitConfig) -> pd.DataFrame:
    rows = []
    for split in [config.discovery_label, config.validation_label, config.test_label]:
        table = splits[split]
        for outcome in OUTCOMES:
            y = table[outcome].to_numpy()
            rows.append(
                {
                    "split": split,
                    "outcome": OUTCOME_LABELS[outcome],
                    "n": int(len(table)),
                    "events": int(y.sum()),
                    "event_rate": float(y.mean()),
                }
            )
    return pd.DataFrame(rows)


def write_readme(output_dir: Path, selected_panels: dict[str, list[str]], metrics: pd.DataFrame, deltas: pd.DataFrame) -> None:
    selected_lines = "\n".join(f"- {OUTCOME_LABELS[outcome]}: {', '.join(features)}" for outcome, features in selected_panels.items())
    test_metrics = metrics[metrics["split"].str.startswith("test_") & metrics["model"].eq("clinical_ast_markers_gwas")]
    metric_lines = "\n".join(
        f"- {row.outcome}: AUROC {row.AUROC:.3f}, AUPRC {row.AUPRC:.3f}, Brier {row.Brier:.3f}"
        for row in test_metrics.itertuples()
    )
    full_model_deltas = deltas[deltas["comparison"].str.endswith("_to_clinical_ast_markers_gwas") & deltas["split"].str.startswith("test_")]
    delta_lines = "\n".join(
        f"- {row.outcome}: delta AUROC {row.delta_AUROC:.3f}, delta AUPRC {row.delta_AUPRC:.3f}, delta Brier {row.delta_Brier:.3f}"
        for row in full_model_deltas.itertuples()
    )
    text = f"""# GWAS Feature Selection and Nested-Model Evaluation

This directory contains the nested clinical-genomic model evaluation and GWAS feature-selection results.

## Selected GWAS Feature Panels

{selected_lines}

## Temporal-Test Performance of the Full Model

{metric_lines}

## Increment from GWAS Features

{delta_lines}

GWAS markers are compact prediction-oriented pangenome markers. They are not claimed as causal loci.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run(paths: Paths, args: argparse.Namespace) -> None:
    if args.clinical_selection is not None:
        load_clinical_selection(args.clinical_selection)
    clinical_record = {
        "selected_panels": {o: selected_clinical_features(o) for o in OUTCOMES},
        "training_years": list(range(2013, 2022)),
        "selection_source": str(args.clinical_selection) if args.clinical_selection else "frozen_Table_S2",
    }
    (paths.output_dir / "clinical_selection.json").write_text(json.dumps(clinical_record, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = pd.read_csv(paths.input_path, low_memory=False)
    require_columns(raw, ["年份", *OUTCOMES], "merged feature table")
    merged = add_derived_predictors(raw)
    for outcome in OUTCOMES:
        require_columns(merged, marker_model_features(outcome), f"{outcome} feature table after derived predictors")
    available_features = set(merged.columns)
    splits, config = temporal_splits(merged, args.validation_years, args.test_years)

    pyseer_paths = {"death_30d": paths.death_pyseer, "metastatic": paths.metastatic_pyseer}
    candidate_pool = pd.concat(
        [read_pyseer_candidate_pool(path, outcome, available_features) for outcome, path in pyseer_paths.items()],
        ignore_index=True,
    )

    selected_panels, ranking_scopes, best_rows = {}, {}, {}
    modules, ranks, traces, comparisons, ablations = [], [], [], [], []
    for outcome in OUTCOMES:
        module = build_outcome_module_contract(candidate_pool, splits[config.discovery_label], outcome)
        # Group first; apply the stricter LRT gate to the retained representatives.
        module["eligible_for_ranking"] = module["lrt_pvalue"].le(STRICT_GWAS_LRT_PVALUE)
        eligible = module.loc[module["eligible_for_ranking"]].copy()
        if eligible.empty:
            raise ValueError(f"No module representative passes LRT <=0.005 for {outcome}")
        ablations.append(within_module_ablation(eligible, splits[config.discovery_label], outcome))
        rank, trace = linear_shap_pruning_rank(eligible, splits[config.discovery_label], outcome, args.max_gwas_panel_size)
        scope = rank.loc[rank["included_in_topk_scope"], "feature"].tolist()
        comparison = topk_panel_comparison(splits[config.discovery_label], splits[config.validation_label], scope, outcome)
        selected, best = select_validation_panel(comparison)
        selected_panels[outcome], ranking_scopes[outcome], best_rows[outcome] = selected, scope, best
        module["selected_final_panel"] = module["representative"].isin(selected)
        rank["selected_final_panel"] = rank["feature"].isin(selected)
        trace["matches_locked_final_panel"] = trace["retained_features_after_step"].eq("|".join(selected))
        modules.append(module)
        ranks.append(rank)
        traces.append(trace)
        comparisons.append(comparison)
    module_contract = pd.concat(modules, ignore_index=True)
    gwas_rank = pd.concat(ranks, ignore_index=True)
    pruning_trace = pd.concat(traces, ignore_index=True)
    topk_comparison = pd.concat(comparisons, ignore_index=True)
    death_ranked_markers, ranked_markers = ranking_scopes["death_30d"], ranking_scopes["metastatic"]
    selected_death_topk_row, selected_topk_row = best_rows["death_30d"], best_rows["metastatic"]

    metric_tables = []
    delta_tables = []
    prediction_tables = []
    feature_panel_tables = []
    for outcome in OUTCOMES:
        metrics, deltas, predictions, feature_panel = evaluate_ladder(splits, config, outcome, selected_panels[outcome])
        metric_tables.append(metrics)
        delta_tables.append(deltas)
        prediction_tables.append(predictions)
        feature_panel_tables.append(feature_panel)

    ladder_metrics = pd.concat(metric_tables, ignore_index=True)
    ladder_deltas = pd.concat(delta_tables, ignore_index=True)
    ladder_predictions = pd.concat(prediction_tables, ignore_index=True)
    feature_panel = pd.concat(feature_panel_tables, ignore_index=True)

    candidate_pool.to_csv(paths.output_dir / "gwas_candidate_pool.csv", index=False)
    pd.concat(ablations, ignore_index=True).to_csv(paths.output_dir / "gwas_within_module_comparisons.csv", index=False)
    module_contract.to_csv(paths.output_dir / "gwas_module_contract.csv", index=False)
    pruning_trace.to_csv(paths.output_dir / "gwas_pruning_trace.csv", index=False)
    gwas_rank.to_csv(paths.output_dir / "gwas_shap_pruning_rank.csv", index=False)
    topk_comparison.to_csv(paths.output_dir / "gwas_topk_panel_comparison.csv", index=False)
    feature_panel.to_csv(paths.output_dir / "gwas_feature_panel.csv", index=False)
    ladder_metrics.to_csv(paths.output_dir / "ladder_metrics.csv", index=False)
    ladder_deltas.to_csv(paths.output_dir / "ladder_deltas.csv", index=False)
    ladder_predictions.to_csv(paths.output_dir / "ladder_predictions.csv", index=False)
    split_summary(splits, config).to_csv(paths.output_dir / "split_summary.csv", index=False)

    manifest = {
        "input_path": str(paths.input_path),
        "output_dir": str(paths.output_dir),
        "pyseer_paths": {outcome: str(path) for outcome, path in pyseer_paths.items()},
        "evidence_generation": "Both outcomes: training-only correlation modules, linear SHAP and sequential pruning",
        "candidate_rule": {
            "selection_scope": "2013-2021 training cohort only; validation and temporal-test splits were excluded from GWAS candidate screening",
            "metastatic_gwas_source_years": config.discovery_years,
            "adjusted_pyseer_metric": "lrt-pvalue",
            "unadjusted_prefilter": "filter-pvalue <= 0.05",
            "strict_module_representative_gate": f"lrt-pvalue <= {STRICT_GWAS_LRT_PVALUE}",
            "allele_frequency_range": [0.05, 0.95],
            "finite_lrt_pvalue_and_beta": True,
            "panaroo_alias_handling": "retain alias sources and map them to available member features",
            "unnamed_group_features": "retained when present and passing the same rule",
            "exclude_pre_gwas_features": "clinical, AST, and hypervirulence-marker predictors excluded from the GWAS candidate pool",
        },
        "panel_selection": {
            "metastatic": "training-set module representatives ranked by computed linear SHAP/pruning; final top-k selected on validation only",
            "mortality_30d": "strict training-set GWAS candidates with module de-redundancy and SHAP/pruning ranking; final top-k selected on validation only",
            "selected_panels": selected_panels,
            "mortality_ranked_topk_candidates": death_ranked_markers,
            "mortality_validation_topk_peak": {
                "panel_size": int(selected_death_topk_row["panel_size"]),
                "gwas_features": str(selected_death_topk_row["gwas_features"]),
                "C": float(selected_death_topk_row["C"]),
                "validation_AUROC": float(selected_death_topk_row["AUROC"]),
                "validation_AUPRC": float(selected_death_topk_row["AUPRC"]),
                "validation_Brier": float(selected_death_topk_row["Brier"]),
                "selection_rule": "choose the smallest top-k panel among validation-AUROC maxima; break remaining ties by lower Brier, higher AUPRC, then lower C",
            },
            "metastatic_ranked_topk_candidates": ranked_markers,
            "metastatic_selected_topk": {
                "panel_size": int(selected_topk_row["panel_size"]),
                "gwas_features": str(selected_topk_row["gwas_features"]),
                "C": float(selected_topk_row["C"]),
                "validation_AUROC": float(selected_topk_row["AUROC"]),
                "validation_AUPRC": float(selected_topk_row["AUPRC"]),
                "validation_Brier": float(selected_topk_row["Brier"]),
                "selection_rule": "choose the smallest top-k panel among validation-AUROC maxima; break remaining ties by lower Brier, higher AUPRC, then lower C",
            },
            "c_strategy": "validation_auroc_tuned",
            "test_set_role": "untouched final temporal evaluation only",
        },
        "test_model_training_split": config.discovery_label,
        "test_model_training_years": config.discovery_years,
        "split_config": {
            "discovery_split": config.discovery_label,
            "validation_split": config.validation_label,
            "test_split": config.test_label,
            "discovery_years": config.discovery_years,
            "validation_years": config.validation_years,
            "test_years": config.test_years,
        },
        "generated_files": sorted({path.name for path in paths.output_dir.iterdir() if path.is_file()} | {"run_manifest.json"}),
    }
    with (paths.output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    write_readme(paths.output_dir, selected_panels, ladder_metrics, ladder_deltas)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args is None:
        return 0
    paths = Paths(
        input_path=resolve_path(args.input).resolve(),
        metastatic_pyseer=resolve_path(args.metastatic_pyseer).resolve(),
        death_pyseer=resolve_path(args.death_pyseer).resolve(),
        output_dir=resolve_path(args.output_dir).resolve(),
    )
    for path in [
        paths.input_path,
        paths.metastatic_pyseer,
        paths.death_pyseer,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    # The public GWAS entry point records the training scope and output hashes.
    manifest_path = paths.death_pyseer.parent.parent / "run_manifest.json"
    gwas_source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if gwas_source.get("status") != "complete" or gwas_source.get("training_years") != list(range(2013, 2022)):
        raise ValueError("Nested models require completed training-only GWAS evidence")
    def digest(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    if gwas_source.get("cohort_sha256") != digest(paths.input_path):
        raise ValueError("Model input differs from the GWAS cohort")
    for outcome, path in {"death_30d": paths.death_pyseer, "metastatic": paths.metastatic_pyseer}.items():
        if gwas_source["pyseer_output_sha256"].get(outcome) != digest(path):
            raise ValueError(f"GWAS output differs from its manifest: {outcome}")
    if args.clinical_selection:
        source = json.loads(args.clinical_selection.read_text(encoding="utf-8"))
        if source.get("status") != "complete" or source.get("input_sha256") != digest(paths.input_path):
            raise ValueError("Clinical selection is incomplete or used a different model input")
    prepare_output_dir(paths.output_dir, args.overwrite)
    run(paths, args)
    print(f"Wrote nested clinical-genomic model outputs to: {paths.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
