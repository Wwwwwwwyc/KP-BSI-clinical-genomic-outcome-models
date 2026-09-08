"""Synthetic checks for locked-probability evaluation; no clinical inputs."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('locked_evaluation', ROOT/'04_analysis/supplementary_statistics/scripts/evaluate_locked_predictions.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def predictions():
    rng = np.random.default_rng(5)
    frames = []
    for split, prefix in [('validation_2022_2023', 'v'), ('test_2024_2025', 't')]:
        frame = pd.DataFrame({'strain':[f'{prefix}{i}' for i in range(40)],
                              'observed':[0,1]*20, 'split':split, 'outcome':'metastatic_infection'})
        for tier in evaluation.TIERS:
            frame[f'{tier}_prob'] = rng.uniform(.1,.9,40)
        frames.append(frame)
    compact = frames[1][['strain','observed','split','outcome']].copy()
    compact[f'{evaluation.COMPACT}_prob'] = rng.uniform(.1,.9,40)
    compact[f'{evaluation.FULL}_prob'] = rng.uniform(.1,.9,40)
    return pd.concat(frames,ignore_index=True), compact


def test_paired_point_is_original_difference_and_identical_predictions_have_zero_interval():
    _, frame = predictions()
    models = [evaluation.COMPACT,evaluation.FULL]
    _, point, draws = evaluation.bootstrap_group(frame,models,25,9)
    delta = evaluation.paired_intervals(point,draws,models[1],models[0])
    expected = roc_auc_score(frame.observed,frame[f'{models[0]}_prob'])-roc_auc_score(frame.observed,frame[f'{models[1]}_prob'])
    assert delta['AUROC']['estimate'] == pytest.approx(expected)
    frame[f'{models[1]}_prob'] = frame[f'{models[0]}_prob']
    _, point, draws = evaluation.bootstrap_group(frame,models,25,9)
    delta = evaluation.paired_intervals(point,draws,*models)
    for row in delta.values():
        assert row['estimate'] == row['ci_low'] == row['ci_high'] == 0


def test_dca_alignment_and_validation_only_calibration():
    ladder, compact = predictions()
    original = evaluation.fit_platt_recalibration
    with patch.object(evaluation,'fit_platt_recalibration',wraps=original) as fit:
        curves, calibrators = evaluation.decision_curves(ladder,compact,'validation_2022_2023','test_2024_2025')
    assert fit.call_count == 3
    for call in fit.call_args_list:
        np.testing.assert_array_equal(call.args[0],ladder.loc[ladder.split.eq('validation_2022_2023'),'observed'])
    assert len(curves) == 7*46
    assert set(curves.model) == {'Clinical','Clinical + AST','Clinical + AST + markers','Compact','Full','treat_all','treat_none'}
    shuffled, _ = evaluation.decision_curves(ladder.sample(frac=1,random_state=1),compact,'validation_2022_2023','test_2024_2025')
    np.testing.assert_allclose(curves.net_benefit,shuffled.net_benefit)
    # Altering held-out outcomes must not alter any reference calibration fit.
    changed = ladder.copy();changed.loc[changed.split.eq('test_2024_2025'),'observed'] = 1-changed.loc[changed.split.eq('test_2024_2025'),'observed']
    compact.observed = 1-compact.observed
    _, changed_calibrators = evaluation.decision_curves(changed,compact,'validation_2022_2023','test_2024_2025')
    assert changed_calibrators == calibrators


def test_wilson_uses_locked_cutoff_and_handles_undefined_ppv():
    result = evaluation.operating_intervals(np.array([1,1,0,0]),np.array([.8,.3,.7,.2]),.5)
    assert (result['TP'],result['FN'],result['TN'],result['FP']) == (1,1,1,1)
    assert result['Sensitivity']['estimate'] == .5
    assert result['Sensitivity']['ci_low'] == pytest.approx(.09453120573423074)
    empty = evaluation.operating_intervals(np.array([1,0]),np.array([.1,.2]),.5)
    assert empty['PPV']['estimate'] is None


def test_misaligned_or_invalid_predictions_are_rejected():
    ladder, compact = predictions()
    compact.loc[0,'strain']='unmatched'
    with pytest.raises(ValueError,match='unmatched'):
        evaluation.decision_curves(ladder,compact,'validation_2022_2023','test_2024_2025')
    compact.loc[0,f'{evaluation.COMPACT}_prob']=np.nan
    with pytest.raises(ValueError,match='finite'):
        evaluation.bootstrap_group(compact,[evaluation.COMPACT],20,42)
