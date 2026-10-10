#!/usr/bin/env python3
"""Final factor analysis -> saves factor_analysis_12dim_v27.json (data source
of manuscript Table 4 / Section 3 factor results).

Final configuration (do not change):
  - Extraction: factor_analyzer.FactorAnalyzer (minres)
  - Rotation: orthogonal Varimax
  - Number of factors: Horn parallel analysis, 200 standard-normal random
    datasets, 95th percentile as reference
  - Criterion: count of real eigenvalues > reference values (random seed 42)
  - Dimensions: 13 with F7 removed (F7+F8+F9=1 linear dependency)

Reproduce: python save_fa_v27_json.py

NOTE on Chinese strings kept below: the per-type result keys (信息型/表情型/
感染型), the feature names, dim_groups labels and the 'removed' description are
written verbatim into the reference JSON that the plotting scripts and the
paper consume; renaming them would break cross-file consistency.
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

# Chinese group labels are saved verbatim into the reference JSON.
DIM_GROUPS_12 = [
    ('人称指向', [0,1,2]),
    ('情态类型', [3,4,5]),
    ('句式策略', [6,7]),
    ('信息组织', [8,9]),
    ('形式变异', [10,11]),
]

results = {}
# Per-type keys (信息型/表情型/感染型) are the Chinese corpus labels; they are
# the keys used by the reference JSON and the figure scripts.
for type_name, type_idx in [('信息型', 0), ('表情型', 1), ('感染型', 2)]:
    mask = y == type_idx
    X_type = X_12[mask]
    kmo_per, kmo_total = calculate_kmo(X_type)
    chi2, p_val = calculate_bartlett_sphericity(X_type)
    n_factors, real_ev, pa_ev = horn_pa(X_type)
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(X_type)
    loadings = fa.loadings_
    var = fa.get_factor_variance()
    results[type_name] = {
        'n': int(mask.sum()),
        'kmo': round(float(kmo_total), 4),
        'bartlett_chi2': round(float(chi2), 2),
        'bartlett_p': float(p_val),
        'n_factors': int(n_factors),
        'eigenvalues': [round(float(x), 4) for x in real_ev],
        'parallel_95': [round(float(x), 4) for x in pa_ev],
        'loadings': [[round(float(x), 4) for x in row] for row in loadings.tolist()],
        'variance_pct': [round(float(x*100), 2) for x in var[1]],
        'variance_cum': [round(float(x*100), 2) for x in var[2]],
    }
    print(f'{type_name}: n={results[type_name]["n"]}, KMO={results[type_name]["kmo"]}, {n_factors} factors, cumulative {results[type_name]["variance_cum"][-1]}%')

out = os.path.join(PKG, '04_factor_analysis', 'factor_analysis_12dim_v27.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump({
        'features': names_12,
        'removed': 'F7_陈述句比例 (F7+F8+F9=1线性依赖)',
        'dim_groups': DIM_GROUPS_12,
        'results': results
    }, f, ensure_ascii=False, indent=2)
print(f'\nSaved: {out}')
