"""Call external genomic predictors using reference-sequence homology."""
from concurrent.futures import ThreadPoolExecutor
import gzip
from pathlib import Path
import shutil
import subprocess
import tempfile

import pandas as pd

FIELDS = ['query', 'subject', 'identity', 'alignment_length', 'query_length',
          'query_start', 'query_end', 'subject_start', 'subject_end', 'evalue', 'bitscore']


def classify_hits(hits, sample_names, genes):
    """A qualifying single alignment must cover 80% of its full reference."""
    hits = hits.copy()
    hits['sample_index'] = hits.subject.str.extract(r'^s(\d+)_', expand=False).astype(int)
    hits['gene'] = hits['query'].map(lambda q: 'iroB' if str(q).startswith('iroB|') else q)
    hits['query_coverage'] = (hits.query_end - hits.query_start).abs().add(1) / hits.query_length * 100
    hits['passes'] = (hits.identity >= 90) & (hits.query_coverage >= 80) & (hits.evalue <= 1e-5)
    hits['Sample_name'] = hits.sample_index.map(dict(enumerate(sample_names)))
    if hits.Sample_name.isna().any():
        raise ValueError('Alignment sample identifiers do not match the cohort.')
    calls = pd.DataFrame({'Sample_name': sample_names}).set_index('Sample_name')
    for gene in genes:
        positive = set(hits.loc[(hits.gene == gene) & hits.passes, 'Sample_name'])
        calls[gene] = calls.index.isin(positive).astype(int)
    return calls, hits


def call_genomes(cohort, genomes, references, output, blast_bin=None, threads=6):
    """Match all iroB alleles and the fimA/yhdJ/blaKPC nucleotide references."""
    if threads < 1 or not cohort.Sample_name.is_unique:
        raise ValueError('Positive thread count and unique sample identifiers are required.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    executables = {}
    for name in ['makeblastdb', 'blastn', 'tblastn']:
        executable = shutil.which(str(Path(blast_bin) / name)) if blast_bin else shutil.which(name)
        if executable is None:
            raise FileNotFoundError(f'BLAST+ executable not found: {name}')
        executables[name] = executable
    def execute(command, name):
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(x) for x in command], stdout=log, stderr=subprocess.STDOUT, check=True)
    with tempfile.TemporaryDirectory(prefix='kp_reference_') as directory:
        temporary = Path(directory)
        fasta = temporary / 'cohort.fna'
        contig_count = 0
        with fasta.open('w', encoding='ascii') as target:
            for i, row in cohort.reset_index(drop=True).iterrows():
                path = Path(genomes) / f"{row.Sample_name}__{row['Accession number']}.fna.gz"
                contig = 0
                with gzip.open(path, 'rt', encoding='ascii') as source:
                    for line in source:
                        if line.startswith('>'):
                            target.write(f'>s{i}_{contig}\n')
                            contig += 1
                        else:
                            target.write(line)
                if not contig:
                    raise ValueError(f'Empty assembly: {path.name}')
                contig_count += contig
        protein, dna = temporary / 'iroB.faa', temporary / 'gwas.fna'
        protein.write_bytes((Path(references) / 'iroB_reference.faa').read_bytes())
        dna.write_bytes((Path(references) / 'locked_gwas_reference.fna').read_bytes())
        protein_ids = [line[1:].split()[0] for line in protein.read_text().splitlines() if line.startswith('>')]
        dna_ids = [line[1:].split()[0] for line in dna.read_text().splitlines() if line.startswith('>')]
        if not protein_ids or not all(x.startswith('iroB|') for x in protein_ids) or not {'fimA_2','yhdJ_1','bla_2'}.issubset(dna_ids):
            raise ValueError('Required reference identifiers are absent.')
        database = temporary / 'cohort'
        execute([executables['makeblastdb'], '-in', fasta, '-dbtype', 'nucl', '-out', database, '-parse_seqids'], 'database')
        commands = []
        for program, query, name in [('tblastn', protein, 'protein'), ('blastn', dna, 'nucleotide')]:
            command = [executables[program], '-query', query, '-db', database,
                       '-out', temporary / f'{name}.tsv', '-outfmt',
                       '6 qseqid sseqid pident length qlen qstart qend sstart send evalue bitscore',
                       '-evalue', '1e-5', '-max_target_seqs', str(contig_count), '-num_threads', str(threads)]
            command += ['-seg', 'no'] if program == 'tblastn' else ['-task', 'blastn', '-dust', 'no']
            commands.append((command, name))
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(execute, c, n) for c, n in commands]
            for job in jobs:
                job.result()
        tables = []
        for name, genes in [('protein', ['iroB']), ('nucleotide', ['fimA_2', 'yhdJ_1', 'bla_2'])]:
            path = temporary / f'{name}.tsv'
            hits = pd.read_csv(path, sep='\t', names=FIELDS) if path.stat().st_size else pd.DataFrame(columns=FIELDS)
            calls, evidence = classify_hits(hits, cohort.Sample_name.tolist(), genes)
            evidence.to_csv(output / f'{name}_alignments.csv', index=False)
            tables.append(calls)
        return tables[0].join(tables[1]).reset_index()
