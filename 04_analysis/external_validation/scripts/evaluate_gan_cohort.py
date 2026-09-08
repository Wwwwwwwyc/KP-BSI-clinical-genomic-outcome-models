"""Reconstruct genomic predictors and evaluate the Gan et al. cohort."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from evaluate_external import DEFAULT_MODEL, predict, evaluate
from reference_calls import call_genomes

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--blast-bin", type=Path)
    parser.add_argument("--threads", type=int, default=6)
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args()

    clinical = pd.read_excel(args.table, sheet_name="Table S1")
    clinical = clinical.loc[clinical["Patient"].isin(["Adult, Pyogenic liver abscess patient", "Adult, Pneumonia patient"])]
    clinical = clinical.assign(outcome_pla=clinical["Patient"].eq("Adult, Pyogenic liver abscess patient").astype(int))
    assert len(clinical) == 194 and clinical.outcome_pla.sum() == 124
    amr = pd.read_excel(args.table, sheet_name="Table S2", header=1)
    amr["sample_key"] = amr["Sample name"].astype(str).str.lower()
    clinical["sample_key"] = clinical["Sample_name"].astype(str).str.lower()
    kpc_cols = [column for column in amr.columns if str(column).lower().startswith("blakpc")]
    if not kpc_cols or not amr["sample_key"].is_unique or not clinical["sample_key"].is_unique:
        raise ValueError("KPC annotations and unique sample identifiers are required")
    kpc = amr.set_index("sample_key")[kpc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1).gt(0).astype(int)
    clinical["bla_2"] = clinical["sample_key"].map(kpc)
    if clinical["bla_2"].isna().any():
        raise ValueError("At least one cohort strain lacks a matched isolate-level KPC call")

    calls = call_genomes(clinical, args.genomes, args.references, args.out_dir, args.blast_bin, args.threads)
    calls = calls.rename(columns={"bla_2": "bla_2_sequence_qc"})
    evaluation = clinical.merge(calls, on="Sample_name", validate="one_to_one")
    model = json.loads(args.model.read_text(encoding="utf-8"))
    predictors = evaluation.rename(columns={"fimA_2": "fimA", "bla_2": "blaKPC", "yhdJ_1": "yhdJ"})
    raw, probability = predict(predictors, model)
    metrics = evaluate(evaluation.outcome_pla, probability, model["threshold"])
    metrics["blaKPC_sequence_concordant_n"] = int((evaluation["bla_2"] == evaluation["bla_2_sequence_qc"]).sum())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = predictors[["Sample_name", "Accession number", "iroB", "fimA", "blaKPC", "yhdJ"]].copy()
    result["outcome"] = evaluation.outcome_pla
    result["raw_probability"] = raw
    result["probability"] = probability
    result.to_csv(args.out_dir / "predictions.csv", index=False)
    (args.out_dir / "performance.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
