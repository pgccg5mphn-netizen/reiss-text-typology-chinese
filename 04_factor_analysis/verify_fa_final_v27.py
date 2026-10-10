#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full verification of the final factor-analysis results (manuscript Table 4).

Same configuration as save_fa_v27_json.py: factor_analyzer + Varimax +
Horn parallel analysis, 200 random datasets / 95th percentile.
Purpose: print KMO, Bartlett, number of factors, cumulative variance and the
loading matrix per type, for manual cross-checking against Table 4.
Reproduce: python verify_fa_final_v27.py

NOTE: the per-type keys (信息型/表情型/感染型) are the Chinese corpus labels
used by the reference JSON; the feature names are the shared Chinese keys.
"""
import json
import numpy as np
import os, sys
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG, '01_feature_engineering'))
from feature_extractor_v9_4 import TextTypeFeatureExtractor
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
import warnings
warnings.filterwarnings('ignore')

# Compatibility: factor_analyzer 0.9.x calls scipy.sum; scipy>=1.13 removed the alias
import scipy, numpy as _np
if not hasattr(scipy, 'sum'):
    scipy.sum = _np.sum


with open(os.path.join(PKG, 'corpus', 'corpus_relabeled_v27.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

texts, labels = [], []
for item in data:
    t = item.get('data', {}).get('text', '').strip()
    a = item.get('annotations', [])
    if not t or not a: continue
    r = a[0].get('result', [])
    if not r: continue
    c = r[0].get('value', {}).get('choices', [])
    if not c: continue
    texts.append(t); labels.append(c[0])

y = np.array([{'informative':0,'expressive':1,'infective':2}[l] for l in labels])
ext = TextTypeFeatureExtractor()
X = np.array([ext.extract(t) for t in texts])
names = ext.FEATURE_NAMES

keep = [i for i in range(13) if i != 6]  # remove F7
X_12 = X[:, keep]
names_12 = [names[i] for i in keep]

def horn_pa(data, n_random=200, seed=42):
    """Horn parallel analysis; return the number of factors."""
    rng = np.random.RandomState(seed)
    n, p = data.shape
    real_eigenvalues = np.sort(np.linalg.eigvalsh(np.corrcoef(data.T)))[::-1]
    random_eigenvalues = np.zeros((n_random, p))
    for i in range(n_random):
        random_data = rng.standard_normal((n, p))
        random_eigenvalues[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(random_data.T)))[::-1]
    # Paper's final criterion: 95th percentile of the parallel analysis (not the
    # random mean), consistent with retrain_v9_3_cascade.py
    pa_95 = np.percentile(random_eigenvalues, 95, axis=0)
    n_factors = sum(1 for j in range(p) if real_eigenvalues[j] > pa_95[j])
    return n_factors, real_eigenvalues, pa_95

print('='*70)
print('Final factor-analysis verification (12 dims, F7 removed)')
print('='*70)

for type_name, type_idx in [('信息型', 0), ('表情型', 1), ('感染型', 2)]:
    mask = y == type_idx
    X_type = X_12[mask]

    kmo_per, kmo_total = calculate_kmo(X_type)
    chi2, p = calculate_bartlett_sphericity(X_type)
    n_factors, real_ev, pa_ev = horn_pa(X_type)

    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(X_type)
    loadings = fa.loadings_
    var = fa.get_factor_variance()
    cum_var = var[2][-1] * 100

    print(f'\n#### {type_name} (n={mask.sum()}) ####')
    print(f'KMO={kmo_total:.4f}, Bartlett chi2={chi2:.1f}')
    print(f'PA factors={n_factors}, cumulative variance={cum_var:.1f}%')
    print(f'\n{"Feature":>20}', end='')
    for j in range(n_factors):
        print(f'  {"Factor "+str(j+1):>9}', end='')
    print()
    for i, name in enumerate(names_12):
        print(f'{name:>20}', end='')
        for j in range(n_factors):
            v = loadings[i, j]
            flag = '*' if abs(v) >= 0.40 else ' '
            print(f'  {v:>+8.4f}{flag}', end='')
        print()
    print()
    print(f'Eigenvalues:  ', end='')
    for j in range(n_factors):
        print(f'  {var[0][j]:>8.4f}', end='')
    print()
    print(f'Variance %:   ', end='')
    for j in range(n_factors):
        print(f'  {var[1][j]*100:>8.2f}', end='')
    print()
