"""Apply the compact model and estimate external discrimination."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / 'parameters' / 'compact_model.json'
GENOMIC = {'iroB', 'fimA', 'blaKPC', 'yhdJ'}


def predict(table, model):
    """Apply development imputation, scaling, coefficients and calibration."""
    if not GENOMIC.issubset(table.columns):
        raise ValueError('Four genomic predictors are required: iroB, fimA, blaKPC, yhdJ.')
    features = model['features']
    x = table.reindex(columns=features).apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if np.isinf(x).any():
        raise ValueError('Predictors must be finite or missing.')
    for i, feature in enumerate(features):
        present = x[:, i][~np.isnan(x[:, i])]
        if feature in GENOMIC and np.isnan(x[:, i]).any():
            raise ValueError(f'Missing genomic calls: {feature}')
        if feature == 'pitt_cont':
            if np.any((present < 0) | (present > 13) | (present != np.floor(present))):
                raise ValueError('pitt_cont must be the continuous integer Pitt score, 0–13.')
        elif not np.isin(present, [0, 1]).all():
            raise ValueError(f'{feature} requires binary 0/1 values.')
    arrays = [np.asarray(model[k]) for k in ['imputation_medians', 'scaler_mean', 'scaler_scale', 'coefficients']]
    if any(a.shape != (len(features),) or not np.isfinite(a).all() for a in arrays) or np.any(arrays[2] <= 0):
        raise ValueError('Invalid model parameters.')
    medians, mean, scale, coef = arrays
    # Score each distinct input once so BLAS batch rounding cannot split tied risks.
    unique_x, inverse = np.unique(np.where(np.isnan(x), medians, x), axis=0, return_inverse=True)
    linear = ((unique_x - mean) / scale) @ coef + model['intercept']
    raw = np.exp(-np.logaddexp(0, -linear))
    clipped = np.clip(raw, model['probability_clip'], 1-model['probability_clip'])
    calibrated = model['platt']['intercept'] + model['platt']['slope'] * np.log(clipped / (1-clipped))
    return raw[inverse], np.exp(-np.logaddexp(0, -calibrated))[inverse]


def evaluate(y, probability, threshold, resamples=2000, seed=20260909):
    y, probability = np.asarray(y, dtype=float), np.asarray(probability, dtype=float)
    if y.ndim != 1 or y.shape != probability.shape or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Recorded binary outcomes with both classes present are required.')
    if resamples < 1:
        raise ValueError('resamples must be positive.')
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        index = np.concatenate([rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)])
        values.append(roc_auc_score(y[index], probability[index]))
    return dict(n=len(y), positive_n=len(positive), negative_n=len(negative),
                AUROC=float(roc_auc_score(y, probability)), AUROC_95_CI=np.quantile(values, [.025, .975]).tolist(),
                bootstrap_resamples=resamples, bootstrap_seed=seed, threshold=threshold,
                sensitivity=float(np.mean(probability[positive] >= threshold)),
                specificity=float(np.mean(probability[negative] < threshold)))


def main():
    parser = argparse.ArgumentParser(description='Function: ' + __doc__)
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('--input', type=Path, required=True, help='CSV containing predictors and outcome.')
    required.add_argument('--output-dir', type=Path, required=True, help='Output: predictions.csv and performance.json.')
    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    optional.add_argument('--resamples', type=int, default=2000)
    optional.add_argument('--seed', type=int, default=20260909)
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    model = json.loads(args.model.read_text(encoding='utf-8'))
    raw, probability = predict(table, model)
    result = evaluate(table['outcome'], probability, model['threshold'], args.resamples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(outcome=table['outcome'], raw_probability=raw, probability=probability)).to_csv(args.output_dir / 'predictions.csv', index=False)
    (args.output_dir / 'performance.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
