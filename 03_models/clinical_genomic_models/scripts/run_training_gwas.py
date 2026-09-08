"""Build training-isolate phylogenetic kinship and run population-adjusted pyseer."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from modeling_contract import OUTCOME_LABELS, require_columns


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def training_phenotypes(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, ["strain", "年份", *OUTCOME_LABELS], "phenotypes")
    if frame.strain.isna().any() or frame.strain.astype(str).duplicated().any():
        raise ValueError("Expected one nonmissing strain ID per first episode")
    years = pd.to_numeric(frame["年份"], errors="raise")
    if not years.between(2013, 2025).all():
        raise ValueError("Missing or unsupported cohort year")
    train = frame.loc[years.between(2013, 2021), ["strain", *OUTCOME_LABELS]].copy()
    if train.empty:
        raise ValueError("No 2013-2021 training isolates")
    train["strain"] = train.strain.astype(str)
    for outcome in OUTCOME_LABELS:
        if not train[outcome].isin([0, 1]).all() or train[outcome].nunique() != 2:
            raise ValueError(f"GWAS requires observed binary training outcomes: {outcome}")
    return train.set_index("strain")


def training_kinship(tree, sample_ids: list[str], midpoint: bool = True):
    """Shared root-to-MRCA branch length, matching pyseer phylogeny_distance --lmm.

    Only the training-isolate subtree is used. Rooting is recorded explicitly;
    an already scientifically rooted tree can be preserved with --keep-root.
    """
    tree = copy.deepcopy(tree)
    names = [tip.name for tip in tree.get_terminals()]
    if len(names) != len(set(names)) or None in names:
        raise ValueError("Tree tips must have unique nonmissing isolate names")
    if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) - set(names):
        raise ValueError("Tree and training sample IDs do not match")
    for branch in tree.find_clades():
        if branch is tree.root and branch.branch_length is None:
            continue
        if branch.branch_length is None or not np.isfinite(branch.branch_length) or branch.branch_length < 0:
            raise ValueError("Kinship requires finite nonnegative branch lengths")
    selected = set(sample_ids)
    for tip in list(tree.get_terminals()):
        if tip.name not in selected:
            tree.prune(tip)
    if midpoint:
        tree.root_at_midpoint()
    depths = tree.depths()
    matrix = np.empty((len(sample_ids), len(sample_ids)))
    for i, left in enumerate(sample_ids):
        for j in range(i + 1):
            value = depths[tree.common_ancestor(left, sample_ids[j])]
            matrix[i, j] = matrix[j, i] = value
    if not np.isfinite(matrix).all() or np.allclose(matrix, 0):
        raise ValueError("Degenerate phylogenetic kinship")
    return tree, pd.DataFrame(matrix, index=sample_ids, columns=sample_ids)


def subset_rtab(source: Path, destination: Path, samples: list[str]) -> int:
    """Stream gene-by-isolate .Rtab into an exactly aligned training-only file."""
    with source.open(encoding="utf-8-sig", newline="") as incoming, destination.open("w", encoding="utf-8", newline="") as outgoing:
        reader, writer = csv.reader(incoming, delimiter="\t"), csv.writer(outgoing, delimiter="\t", lineterminator="\n")
        header = next(reader)
        if len(set(header[1:])) != len(header) - 1 or set(samples) - set(header[1:]):
            raise ValueError("Pangenome columns must uniquely cover every training isolate")
        indices = [header.index(sample) for sample in samples]
        writer.writerow([header[0], *samples])
        count, seen = 0, set()
        for row in reader:
            if len(row) != len(header) or not row[0] or row[0] in seen:
                raise ValueError("Malformed or duplicate gene row in pangenome .Rtab")
            values = [row[index] for index in indices]
            if set(values) - {"0", "1"}:
                raise ValueError(f"Training PAV must be complete binary data: {row[0]}")
            seen.add(row[0])
            writer.writerow([row[0], *values])
            count += 1
    if not count:
        raise ValueError("No pangenome gene rows")
    return count


def pyseer_command(executable: str, phenotypes: Path, pav: Path, kinship: Path, outcome: str, cpu: int) -> list[str]:
    return [executable, "--lmm", "--phenotypes", str(phenotypes), "--phenotype-column", outcome,
            "--pres", str(pav), "--similarity", str(kinship), "--min-af", "0.05", "--max-af", "0.95",
            "--filter-pvalue", "0.05", "--lrt-pvalue", "1", "--cpu", str(cpu)]


def run(args) -> None:
    from Bio import Phylo

    train = training_phenotypes(pd.read_csv(args.input, dtype={"strain": str}))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "training_years": list(range(2013, 2022)), "n_training": len(train),
                "kinship": "shared root-to-MRCA branch length on training-isolate subtree",
                "rooting": "preserve supplied root" if args.keep_root else "training-subtree midpoint",
                "population_adjustment": "phylogenetic kinship random effect; no clinical fixed covariates",
                "inputs_sha256": {str(p): sha256(p) for p in [args.input, args.pangenome, args.tree or args.core_alignment]},
                "cohort_sha256": sha256(args.input), "pyseer_output_sha256": {}, "commands": []}
    manifest_path = args.output_dir / "run_manifest.json"
    try:
        tree_path = args.tree
        if tree_path is None:
            tree_path = args.output_dir / "core_genome.tree"
            command = [args.fasttree_executable, "-nt", str(args.core_alignment)]
            manifest["commands"].append(command)
            with tree_path.open("w") as out, (args.output_dir / "fasttree.log").open("w") as err:
                subprocess.run(command, stdout=out, stderr=err, check=True)
        tree, kinship = training_kinship(Phylo.read(tree_path, "newick"), train.index.tolist(), not args.keep_root)
        Phylo.write(tree, args.output_dir / "training.tree", "newick")
        kinship_path = args.output_dir / "kinship.tsv"
        kinship.to_csv(kinship_path, sep="\t")
        phenotypes = args.output_dir / "phenotypes.tsv"
        train.to_csv(phenotypes, sep="\t")
        pav = args.output_dir / "training_gene_presence_absence.Rtab"
        manifest["n_gene_families"] = subset_rtab(args.pangenome, pav, train.index.tolist())
        for outcome, folder in [("death_30d", "mortality"), ("metastatic", "invasive_phenotype")]:
            destination = args.output_dir / folder
            destination.mkdir()
            command = pyseer_command(args.pyseer_executable, phenotypes, pav, kinship_path, outcome, args.cpu)
            manifest["commands"].append(command)
            with (destination / "pyseer.tsv").open("w") as out, (destination / "pyseer.log").open("w") as err:
                subprocess.run(command, stdout=out, stderr=err, check=True)
            table = pd.read_csv(destination / "pyseer.tsv", sep="\t")
            require_columns(table, ["variant", "af", "filter-pvalue", "lrt-pvalue", "beta"], "pyseer output")
            manifest["pyseer_output_sha256"][outcome] = sha256(destination / "pyseer.tsv")
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Function\nBuild training-only phylogenetic kinship and both pyseer LMM GWAS scans.\n\nOutput\nPrivate GWAS output directory with training.tree, kinship.tsv, two pyseer.tsv files and run_manifest.json.", formatter_class=argparse.RawTextHelpFormatter)
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("--input", type=Path, required=True, help="Private CSV with strain, year and observed outcomes")
    required.add_argument("--pangenome", type=Path, required=True, help="Private gene-by-isolate Panaroo/Roary .Rtab")
    required.add_argument("--output-dir", type=Path, required=True)
    tree = required.add_mutually_exclusive_group(required=True)
    tree.add_argument("--tree", type=Path, help="Core-SNP Newick phylogeny with branch lengths")
    tree.add_argument("--core-alignment", type=Path, help="Core-SNP FASTA alignment for FastTree -nt")
    optional = parser.add_argument_group("Optional Arguments")
    optional.add_argument("--pyseer-executable", default="pyseer")
    optional.add_argument("--fasttree-executable", default="FastTree")
    optional.add_argument("--cpu", type=int, default=1)
    optional.add_argument("--keep-root", action="store_true", help="Preserve the supplied scientific rooting")
    optional.add_argument("--run", action="store_true")
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    if not args.run or args.cpu < 1:
        parser.error("--run and positive --cpu are required")
    run(args)


if __name__ == "__main__":
    main()
