"""Complete-case adjusted and individual-marker logistic associations; no plotting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "03_models/clinical_genomic_models/scripts"))
from modeling_contract import FIVE_MARKERS, OUTCOME_LABELS, require_columns

HOST_COLUMNS = {
    "diabetes": "是否有糖尿病（1是0否）", "hepatitis": "是否有肝炎（1是0否）",
    "malignancy": "是否有癌症（1是0否）", "renal_insufficiency": "肾功能不全（1是0否）",
    "organ_transplant": "器官移植状态（1是0否）",
}


def association_frame(raw: pd.DataFrame) -> pd.DataFrame:
    require_columns(raw, ["年龄", "男1女2", "pitt_cont", "resistance", *HOST_COLUMNS.values(), *FIVE_MARKERS, *OUTCOME_LABELS], "association input")
    out = pd.DataFrame(index=raw.index)
    out["age10"] = pd.to_numeric(raw["年龄"], errors="raise") / 10
    sex = pd.to_numeric(raw["男1女2"], errors="raise")
    if not sex.dropna().isin([1, 2]).all():
        raise ValueError("Sex must be 1=male, 2=female, or missing")
    out["male"] = sex.eq(1).astype(float).where(sex.notna())
    out["pitt"] = pd.to_numeric(raw["pitt_cont"], errors="raise")
    for name, column in {**HOST_COLUMNS, **{o: o for o in OUTCOME_LABELS}}.items():
        out[name] = pd.to_numeric(raw[column], errors="raise")
        if not out[name].dropna().isin([0, 1]).all():
            raise ValueError(f"Nonbinary clinical field: {column}")
    markers = raw[FIVE_MARKERS].apply(pd.to_numeric, errors="raise")
    if not markers.stack().isin([0, 1]).all():
        raise ValueError("Marker calls must be binary or absent")
    # The manuscript assigns absent hypervirulence-marker calls zero.
    markers = markers.fillna(0)
    out[FIVE_MARKERS] = markers
    out["profile_all"] = markers.sum(axis=1).eq(5).astype(float)
    out["profile_incomplete"] = (markers.iucA.eq(1) & out.profile_all.eq(0)).astype(float)
    resistance = raw.resistance
    if not resistance.dropna().isin(["SKPN", "ESBL", "CRKP"]).all():
        raise ValueError("Unsupported resistance phenotype")
    for category in ["ESBL", "CRKP"]:
        out[f"res_{category}"] = resistance.eq(category).astype(float).where(resistance.notna())
    if np.isinf(out.to_numpy(dtype=float)).any():
        raise ValueError("Infinite association input")
    return out


def fit_association(frame: pd.DataFrame, outcome: str, features: list[str], analysis: str) -> pd.DataFrame:
    data = frame[[outcome, *features]].dropna()
    if data.empty or data[outcome].nunique() != 2:
        raise ValueError(f"No two-class complete-case sample for {outcome}/{analysis}")
    design = sm.add_constant(data[features], has_constant="add")
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        raise ValueError(f"Rank-deficient association design: {outcome}/{analysis}")
    fitted = sm.GLM(data[outcome], design, family=sm.families.Binomial()).fit()
    if not fitted.converged:
        raise ValueError(f"Association did not converge: {outcome}/{analysis}")
    ci = fitted.conf_int()
    return pd.DataFrame({
        "outcome": OUTCOME_LABELS[outcome], "analysis": analysis, "term": fitted.params.index,
        "coefficient": fitted.params.to_numpy(), "standard_error": fitted.bse.to_numpy(),
        "odds_ratio": np.exp(fitted.params.to_numpy()), "ci95_low": np.exp(ci[0].to_numpy()),
        "ci95_high": np.exp(ci[1].to_numpy()), "p_value": fitted.pvalues.to_numpy(),
        "n": len(data), "events": int(data[outcome].sum()), "excluded_missing": len(frame) - len(data),
    }).query("term != 'const'")


def run(input_path: Path, output_dir: Path) -> None:
    frame = association_frame(pd.read_csv(input_path))
    features = ["age10", "male", *HOST_COLUMNS, "pitt", "res_ESBL", "res_CRKP", "profile_incomplete", "profile_all"]
    adjusted = pd.concat([fit_association(frame, o, features, "adjusted") for o in OUTCOME_LABELS], ignore_index=True)
    markers = pd.concat([fit_association(frame, o, [m], "unadjusted_marker") for o in OUTCOME_LABELS for m in FIVE_MARKERS], ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    adjusted.to_csv(output_dir / "adjusted_associations.csv", index=False)
    markers.to_csv(output_dir / "marker_associations.csv", index=False)
    manifest = {"status": "complete", "analysis": "Unpenalized binomial GLM; complete cases per model; Wald 95% CI",
                "reference_categories": {"resistance": "SKPN", "virulence_profile": "iucA-negative"},
                "age_unit": "10 years", "pitt": "continuous score", "clinical_missingness": "excluded per model, never coded negative",
                "absent_marker_calls": "zero, as specified in manuscript", "causal_interpretation": False,
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Function\nRun complete-case clinical-genomic logistic associations.\n\nOutput\nAdjusted and individual-marker association CSVs and manifest; no figures.", formatter_class=argparse.RawTextHelpFormatter)
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("--input", type=Path, required=True, help="Private analysis-ready clinical-genomic CSV")
    required.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument_group("Optional Arguments").add_argument("--run", action="store_true")
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
