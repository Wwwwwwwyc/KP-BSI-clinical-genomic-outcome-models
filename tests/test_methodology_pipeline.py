"""Regression tests use generated data and never load private cohort files."""
import importlib.util
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from Bio import Phylo
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / '03_models/clinical_genomic_models/scripts'
sys.path.insert(0, str(SCRIPTS))
import modeling_contract as contract
import build_nested_information_models as nested
import select_clinical_features as clinical
import run_training_gwas as gwas


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


statistics = load_module('public_statistics', ROOT / '04_analysis/supplementary_statistics/scripts/build_supplementary_model_statistics.py')
association = load_module('public_association', ROOT / '04_analysis/association/scripts/run_association_analysis.py')


class MethodologyTests(unittest.TestCase):
    def setUp(self):
        self.original = {o: list(p) for o, p in contract.ACTIVE_CLINICAL_PANELS.items()}
        self.addCleanup(lambda: contract.ACTIVE_CLINICAL_PANELS.update(self.original))

    def test_bootstrap_point_estimate_does_not_depend_on_resampling(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        ref = np.array([.1, .3, .5, .7, .2, .4, .8, .6])
        comp = np.array([.2, .8, .4, .6, .1, .3, .7, .5])
        data = pd.DataFrame({'split': 'test_2024_2025', 'y': y, 'ref': ref, 'comp': comp})
        expected = {name: metric(y, comp) - metric(y, ref) for name, metric in [('AUROC', roc_auc_score), ('AUPRC', average_precision_score), ('Brier', brier_score_loss), ('LogLoss', log_loss)]}
        for seed in [1, 91]:
            result = contract.paired_bootstrap_metric_delta(data, split_col='split', outcome_col='y', reference_col='ref', comparator_col='comp', metric_label='paired', n_bootstrap=30, seed=seed)
            self.assertEqual(result.set_index('metric').delta_comparator_minus_reference.to_dict(), expected)

    def test_locked_threshold_is_not_reselected_on_test(self):
        with patch.object(contract, 'roc_curve', side_effect=AssertionError('Threshold reselected')):
            metrics = contract.threshold_metrics(np.array([0,1,0,1]), np.array([.1,.2,.7,.9]), threshold=.8)
        self.assertEqual(metrics['threshold_youden'], .8)
        self.assertEqual(metrics['sensitivity'], .5)

    def test_marker_absence_and_unknown_ast_contract(self):
        frame = pd.DataFrame({'resistance': ['SKPN', 'CRKP'], 'pitt_cont': [0, 1]})
        for marker in contract.FIVE_MARKERS:
            frame[marker] = [np.nan, 1]
        derived = contract.add_derived_predictors(frame)
        self.assertEqual(derived.loc[0, contract.FIVE_MARKERS].sum(), 0)
        self.assertFalse(derived[contract.FIVE_MARKERS].isna().any().any())
        with self.assertRaisesRegex(ValueError, 'observed'):
            contract.add_derived_predictors(frame.assign(resistance=np.nan))

    def test_continuous_pitt_is_required_by_modeling_and_association(self):
        frame = pd.DataFrame({
            '年龄': [40, 60, 70], '男1女2': [1, 2, 1],
            'resistance': ['SKPN', 'CRKP', 'ESBL'],
            'pitt（<4 0,>=4 1）': [0, 1, 1],
        })
        for column in [*association.HOST_COLUMNS.values(), *contract.FIVE_MARKERS, *contract.OUTCOME_LABELS]:
            frame[column] = [0, 1, 0]
        for transform in [contract.add_derived_predictors, association.association_frame]:
            with self.subTest(transform=transform.__name__):
                for invalid in [frame, frame.drop(columns=['pitt（<4 0,>=4 1）'])]:
                    with self.assertRaisesRegex(ValueError, 'pitt_cont'):
                        transform(invalid)
        frame['pitt_cont'] = [2, 7, 13]
        np.testing.assert_array_equal(contract.add_derived_predictors(frame)['pitt_cont'], [2, 7, 13])
        np.testing.assert_array_equal(association.association_frame(frame)['pitt'], [2, 7, 13])

    def test_five_marker_models_fit_training_only(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame({'strain': range(80), '年份': [2020]*40 + [2023]*20 + [2024]*20,
                              'host': rng.normal(size=80), 'pitt_cont': 1, 'resistance': 'SKPN',
                              'death_30d': [0, 1]*40, 'metastatic': [1, 0]*40})
        for marker in contract.FIVE_MARKERS:
            frame[marker] = rng.integers(0, 2, len(frame))
        contract.ACTIVE_CLINICAL_PANELS.update({o: ['host'] for o in contract.OUTCOME_LABELS})
        fit_years = []
        original = contract.fit_model
        def fit(train, *args, **kwargs):
            fit_years.append(set(train['年份']))
            return original(train, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'synthetic.csv'
            frame.to_csv(path, index=False)
            with patch.object(contract, 'C_CANDIDATES', [.1]), patch.object(contract, 'fit_model', side_effect=fit):
                metrics, predictions = statistics.build_five_marker_predictions(path)
        self.assertTrue(fit_years)
        self.assertTrue(all(years == {2020} for years in fit_years))
        self.assertEqual(set(predictions.split), {'validation_2022_2023', 'test_2024_2025'})
        self.assertEqual(len(metrics), 12)

    def test_clinical_selection_is_fold_local_and_outcome_specific(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(240, 2))
        train = pd.DataFrame({'年份': 2020, 'a': x[:, 0], 'b': x[:, 1], 'death_30d': (x[:, 0] > 0).astype(int), 'metastatic': (x[:, 1] > 0).astype(int)})
        with patch.object(clinical, 'C_CANDIDATES', [.1]), patch.object(clinical, 'L1_RATIOS', [1.]):
            first, _, _ = clinical.sparse_selection(train, 'death_30d', ['a', 'b'], 'test', repeats=8)
            second, _, _ = clinical.sparse_selection(train, 'metastatic', ['a', 'b'], 'test', repeats=8)
        self.assertEqual(first, ['a'])
        self.assertEqual(second, ['b'])
        with self.assertRaises(ValueError):
            clinical.remove_correlated(train.assign(年份=2023), ['a', 'b'])
        fitted = clinical.elastic_model(.1, .5).fit(pd.DataFrame({'a': [0, 2, np.nan, 4]}), [0, 0, 1, 1])
        fitted.predict_proba(pd.DataFrame({'a': [9999, np.nan]}))
        self.assertEqual(fitted[0].statistics_[0], 2)

    def test_kinship_geometry_and_training_alignment(self):
        tree = Phylo.read(StringIO('(a:1,b:1,c:2);'), 'newick')
        _, matrix = gwas.training_kinship(tree, ['b', 'a', 'c'], midpoint=False)
        np.testing.assert_allclose(matrix.to_numpy(), np.diag([1, 1, 2]))
        _, subset = gwas.training_kinship(tree, ['b', 'a'])
        self.assertEqual(subset.index.tolist(), ['b', 'a'])
        with self.assertRaises(ValueError):
            gwas.training_kinship(tree, ['missing', 'a'])
        frame = pd.DataFrame({'strain': ['b','a','c'], '年份': [2020,2021,2024], 'death_30d': [0,1,1], 'metastatic': [1,0,1]})
        self.assertEqual(gwas.training_phenotypes(frame).index.tolist(), ['b', 'a'])
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / 'pav.Rtab', Path(temporary) / 'train.Rtab'
            source.write_text('Gene\ta\tc\tb\ngene_x\t1\t1\t0\n')
            self.assertEqual(gwas.subset_rtab(source, target, ['b','a']), 1)
            self.assertEqual(target.read_text(), 'Gene\tb\ta\ngene_x\t0\t1\n')
        command = gwas.pyseer_command('pyseer', Path('p'), Path('v'), Path('k'), 'death_30d', 2)
        self.assertIn('--lmm', command)
        self.assertIn('--similarity', command)
        self.assertNotIn('--continuous', command)

    def test_nested_workflow_generates_both_outcome_evidence(self):
        rng = np.random.default_rng(12)
        n = 160
        frame = pd.DataFrame({'strain': range(n), '年份': [2020]*80 + [2023]*40 + [2024]*40,
                              'host': rng.normal(size=n), 'pitt_cont': 1, 'resistance': 'SKPN',
                              'death_30d': [0,1]*(n//2), 'metastatic': [1,0]*(n//2)})
        for feature in [*contract.FIVE_MARKERS, 'gene_a', 'gene_b']:
            frame[feature] = rng.integers(0,2,n)
        frame['gene_alias'] = 1-frame['gene_a']
        evidence = pd.DataFrame({'variant': ['gene_a','gene_alias','gene_b'], 'af': [.5]*3,
                                 'filter-pvalue': [.01]*3, 'lrt-pvalue': [.001,.002,.003],
                                 'beta': [.2]*3, 'beta-std-err': [.1]*3, 'variant_h2': [.1]*3})
        contract.ACTIVE_CLINICAL_PANELS.update({o:['host'] for o in contract.OUTCOME_LABELS})
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            input_path, pyseer_path = base/'input.csv', base/'pyseer.tsv'
            frame.to_csv(input_path, index=False)
            evidence.to_csv(pyseer_path, sep='\t', index=False)
            output = base/'nested'
            output.mkdir()
            paths = nested.Paths(input_path, pyseer_path, pyseer_path, output)
            args = SimpleNamespace(clinical_selection=None, validation_years=[2022,2023], test_years=[2024,2025], max_gwas_panel_size=2)
            with patch.object(nested, 'C_CANDIDATES', [.1]), patch.object(contract, 'C_CANDIDATES', [.1]):
                nested.run(paths,args)
            for name in ['gwas_module_contract.csv','gwas_shap_pruning_rank.csv','gwas_pruning_trace.csv','gwas_within_module_comparisons.csv']:
                self.assertEqual(set(pd.read_csv(output/name).outcome), set(contract.OUTCOME_LABELS.values()))
            manifest = json.loads((output/'run_manifest.json').read_text())
            self.assertEqual(manifest['test_model_training_years'], [2020])
            modules = pd.read_csv(output/'gwas_module_contract.csv')
            self.assertTrue((modules.loc[modules.all_genes_pipe.str.contains('gene_alias'), 'representative']=='gene_a').all())

    def test_association_missing_values_are_excluded_not_negative(self):
        rng = np.random.default_rng(5)
        frame = pd.DataFrame({'y': rng.integers(0,2,100), 'x': rng.normal(size=100)})
        frame.loc[0,'x'] = np.nan
        frame.loc[1,'y'] = np.nan
        result = association.fit_association(frame.rename(columns={'y':'death_30d'}), 'death_30d', ['x'], 'test')
        self.assertEqual(result.iloc[0]['n'],98)
        self.assertEqual(result.iloc[0]['excluded_missing'],2)

    def test_new_entrypoints_print_help_without_private_inputs(self):
        for script in [SCRIPTS/'select_clinical_features.py', SCRIPTS/'run_training_gwas.py',
                       ROOT/'04_analysis/association/scripts/run_association_analysis.py', ROOT/'scripts/run_methodology_pipeline.py']:
            result = subprocess.run([sys.executable, '-X', 'utf8', str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            for section in ['Function','Required Arguments','Optional Arguments','Output']:
                self.assertIn(section,result.stdout)


if __name__ == '__main__':
    unittest.main()
