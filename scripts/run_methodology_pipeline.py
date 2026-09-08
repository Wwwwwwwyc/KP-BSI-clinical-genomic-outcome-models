"""Run the manuscript methods against explicitly supplied private, analysis-ready inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "03_models/clinical_genomic_models/scripts"


def commands(args) -> list[list[str]]:
    output = args.output_dir.resolve()
    selection = output / "clinical_selection/clinical_selection.json"
    gwas, nested = output / "gwas_inputs", output / "nested_information_models"
    def python(script, *options):
        return [sys.executable, str(script), "--run", *map(str, options)]
    tree_option = ("--tree", args.tree.resolve()) if args.tree else ("--core-alignment", args.core_alignment.resolve())
    return [
        python(MODEL / "select_clinical_features.py", "--input", args.input.resolve(), "--output-dir", output / "clinical_selection"),
        python(MODEL / "run_training_gwas.py", "--input", args.input.resolve(), "--pangenome", args.pangenome.resolve(), *tree_option,
               "--output-dir", gwas, "--pyseer-executable", args.pyseer_executable, "--fasttree-executable", args.fasttree_executable, "--cpu", args.cpu),
        python(MODEL / "build_nested_information_models.py", "--input", args.input.resolve(), "--clinical-selection", selection,
               "--death-pyseer", gwas / "mortality/pyseer.tsv", "--metastatic-pyseer", gwas / "invasive_phenotype/pyseer.tsv", "--output-dir", nested),
        python(MODEL / "compare_model_families.py", "--input", args.input.resolve(), "--information-model-dir", nested, "--output-dir", output / "model_family_comparison"),
        python(MODEL / "build_compact_model.py", "--input", args.input.resolve(), "--information-model-dir", nested, "--output-dir", output / "compact_model"),
        python(ROOT / "04_analysis/association/scripts/run_association_analysis.py", "--input", args.input.resolve(), "--output-dir", output / "associations"),
        python(ROOT / "04_analysis/supplementary_statistics/scripts/build_supplementary_model_statistics.py", "--features", args.input.resolve(),
               "--clinical-selection", selection, "--ladder-predictions", nested / "ladder_predictions.csv", "--output-dir", output / "supplementary_statistics"),
        python(ROOT / "04_analysis/supplementary_statistics/scripts/evaluate_locked_predictions.py",
               "--ladder-predictions", nested / "ladder_predictions.csv", "--compact-model-dir", output / "compact_model",
               "--output-dir", output / "locked_evaluation"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Function\nRun clinical selection, tree/kinship, pyseer, nested models, model-family/compact comparisons and associations.\n\nOutput\nA new private analysis output directory; no upload or plotting commands.", formatter_class=argparse.RawTextHelpFormatter)
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("--input", type=Path, required=True, help="Analysis-ready numeric CSV including all 34 clinical candidates, outcomes and PAV columns")
    required.add_argument("--pangenome", type=Path, required=True, help="Aligned Panaroo/Roary .Rtab with the same gene names and isolate IDs")
    required.add_argument("--output-dir", type=Path, required=True)
    tree = required.add_mutually_exclusive_group(required=True)
    tree.add_argument("--tree", type=Path)
    tree.add_argument("--core-alignment", type=Path)
    optional = parser.add_argument_group("Optional Arguments")
    optional.add_argument("--pyseer-executable", default="pyseer")
    optional.add_argument("--fasttree-executable", default="FastTree")
    optional.add_argument("--cpu", type=int, default=1)
    optional.add_argument("--run", action="store_true")
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    if not args.run or args.cpu < 1:
        parser.error("--run and positive --cpu are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "completed_stages": [], "commands": commands(args)}
    try:
        for number, command in enumerate(manifest["commands"], 1):
            with (args.output_dir / f"stage_{number}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            manifest["completed_stages"].append(number)
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        (args.output_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
