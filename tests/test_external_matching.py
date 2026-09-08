import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / '04_analysis/external_validation/scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identical_inputs_remain_tied_in_large_batches():
    module = load('evaluate_external')
    model = json.loads((ROOT / 'parameters/compact_model.json').read_text())
    table = pd.DataFrame({'iroB': [1]*194, 'fimA': [1]*194,
                          'blaKPC': [0]*194, 'yhdJ': [0]*194})
    raw, probability = module.predict(table, model)
    assert len(np.unique(raw)) == len(np.unique(probability)) == 1
    result = module.evaluate(np.arange(194) % 2, probability, model['threshold'], 10)
    assert result['AUROC'] == .5


def test_identity_and_single_alignment_coverage_thresholds():
    module = load('reference_calls')
    hits = pd.DataFrame({
        'query': ['fimA_2']*5, 'subject': ['s0_a','s1_a','s2_a','s2_b','s3_a'],
        'identity': [90,89.99,99,99,99], 'query_length': [100]*5,
        'query_start': [1,1,1,51,1], 'query_end': [80,100,50,100,100],
        'evalue': [1e-5,0,0,0,1e-4]})
    calls, _ = module.classify_hits(hits, ['a','b','c','d'], ['fimA_2'])
    assert calls.fimA_2.tolist() == [1,0,0,0]
