#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap stability test (Section 3.2: sampling stability of the covariation
structure).

Identical configuration to the main analysis (save_fa_v27_json.py), so that the
reproduction rates reflect sampling variability only, with no method differences
mixed in:
  - Extraction: factor_analyzer.FactorAnalyzer (minres, correlation matrix used
    internally)
  - Rotation: orthogonal Varimax
  - Dimensions: 13 with F7 removed (F7+F8+F9=1 linear dependency), i.e. 12
  - Number of factors: fixed to the full-sample parallel-analysis result
    (informative 4 / expressive 4 / operative 3)
  - Input: no pre-standardisation (factor_analyzer converts to the correlation
    matrix internally, same as the main analysis)

Factor alignment: the full-sample solution is the reference; columns are matched
by maximum absolute inner product with sign alignment.
Covariation criterion: the bootstrap proportion of draws in which two features
are simultaneously high-loading (|loading| >= 0.40) on the same factor slot.
"""
import os
import sys
import json
import numpy as np
import scipy
from factor_analyzer import FactorAnalyzer

# Compatibility: factor_analyzer 0.9.x calls scipy.sum; scipy>=1.13 removed the alias
if not hasattr(scipy, 'sum'):
    scipy.sum = np.sum

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG, '01_feature_engineering'))
from feature_extractor_v9_4 import TextTypeFeatureExtractor

CORPUS = os.path.join(PKG, 'corpus', 'corpus_relabeled_v27.json')
N_BOOT = 200
FIXED_FACTORS = {'informative': 4, 'expressive': 4, 'infective': 3}
THRESH = 0.4
SEED = 42

with open(CORPUS, encoding='utf-8') as f:
    data = json.load(f)
texts, labels = [], []
for item in data:
    t = item.get('data', {}).get('text', '').strip()
    a = item.get('annotations', [])
    if not t or not a:
        continue
    r = a[0].get('result', [])
    if not r:
        continue
    c = r[0].get('value', {}).get('choices', [])
    if not c:
        continue
    texts.append(t)
    labels.append(c[0])

ext = TextTypeFeatureExtractor()
X_all = np.array([ext.extract(t) for t in texts])
names = ext.FEATURE_NAMES
keep = [i for i in range(13) if i != 6]   # remove F7
X_12 = X_all[:, keep]
names_12 = [names[i] for i in keep]
p = len(names_12)


def align_loadings(L, ref):
    """L, ref: (p, k). Align the columns of L to the columns of ref by maximum
    absolute inner product, with sign alignment.

    factor_analyzer's Varimax is order-stable under repeated fits on the same
    data (measured deviation: 0), so only the occasional permutation induced by
    resampled subsamples needs handling.

    Implementation note: a **globally optimal matching** is used (enumerate all
    k! permutations of the k factors and take the one maximising the total inner
    product) rather than per-column greedy matching. Greedy can pick wrongly when
    two factors have similar inner products; in one of 200 operative-type draws
    it mismatched F6 and F8 into different slots, producing a co-occurrence
    frequency above 1. With k of only 3-4 the exhaustive cost is negligible.
    """
    from itertools import permutations
    k = L.shape[1]
    # Precompute the |inner product| matrix (with sign alignment)
    score = np.zeros((k, k))
    for j in range(k):
        for i in range(k):
            rj, li = ref[:, j], L[:, i]
            s = np.sign(np.dot(rj, li)) or 1.0
            score[j, i] = abs(np.dot(rj, s * li))
    # Global optimum: maximise sum_j score[j, perm[j]]
    best_perm, best_total = None, -np.inf
    for perm in permutations(range(k)):
        total = sum(score[j, perm[j]] for j in range(k))
        if total > best_total:
            best_total, best_perm = total, perm
    Laligned = np.zeros_like(L)
    for newj, i in enumerate(best_perm):
        s = np.sign(np.dot(ref[:, newj], L[:, i])) or 1.0
        Laligned[:, newj] = s * L[:, i]
    return Laligned


summary = {}
# label_zh values are the Chinese corpus labels; they become the summary keys of
# the saved reference JSON.
for label_en, label_zh in [('informative', '信息型'), ('expressive', '表情型'), ('infective', '感染型')]:
    mask = np.array([l == label_en for l in labels])
    X = X_12[mask]
    n = int(mask.sum())
    k = FIXED_FACTORS[label_en]

    # Full-sample solution as the alignment reference (same as the main
    # analysis: no pre-standardisation)
    fa0 = FactorAnalyzer(n_factors=k, rotation='varimax')
    fa0.fit(X)
    ref = np.array(fa0.loadings_)

    factor_feat_count = np.zeros((k, p))
    cooccur = np.zeros((p, p))
    rng = np.random.default_rng(SEED)
    total = N_BOOT
    for rep in range(N_BOOT):
        idx = rng.integers(0, n, n)
        Xb = X[idx]
        try:
            fab = FactorAnalyzer(n_factors=k, rotation='varimax')
            fab.fit(Xb)
        except Exception:
            total -= 1
            continue
        L = align_loadings(np.array(fab.loadings_), ref)
        high = np.abs(L) >= THRESH
        for j in range(k):
            fh = np.where(high[:, j])[0]
            factor_feat_count[j, fh] += 1
        # Co-occurrence counting: deduplicate by feature pair; within one
        # resample a given pair is counted once.
        # If accumulated slot-by-slot, a feature falling into several slots
        # would be counted repeatedly, pushing the frequency above 1
        # (operative-type F6-F8 once hit 201/200 = 1.005).
        for a1 in range(p):
            for b1 in range(a1 + 1, p):
                if high[a1, :].any() and high[b1, :].any() \
                        and np.any(high[a1, :] & high[b1, :]):
                    cooccur[a1, b1] += 1
                    cooccur[b1, a1] += 1

    print(f"\n=== {label_zh} (n={n}, fixed factors={k}, valid bootstraps={total}) ===")
    print(f"Feature order: {names_12}")
    factor_report = []
    for j in range(k):
        oi = np.argsort(-factor_feat_count[j])
        line = [(names_12[i], round(float(factor_feat_count[j, i] / total), 3))
                for i in oi if factor_feat_count[j, i] > 0]
        factor_report.append(line)
        txt = ' | '.join(f"{nm}({fr:.2f})" for nm, fr in line)
        print(f"  Factor {j+1} high-loading features [frequency]: {txt}")
    pairs = []
    for i in range(p):
        for jj in range(i + 1, p):
            if cooccur[i, jj] > 0:
                pairs.append((round(float(cooccur[i, jj] / total), 3), names_12[i], names_12[jj]))
    pairs.sort(reverse=True)
    print("  Key covariation pairs (same-factor co-high-loading frequency, TOP10):")
    for fr, a, b in pairs[:10]:
        print(f"    {a} <-> {b}: {fr:.2f}")

    summary[label_zh] = {
        'n': n, 'fixed_factors': k, 'effective_bootstrap': int(total),
        'thresh': THRESH,
        'extraction': 'factor_analyzer.FactorAnalyzer(minres)',
        'rotation': 'varimax',
        'factor_highload_freq': [[[nm, fr] for nm, fr in line] for line in factor_report],
        'top_cooccur': [[fr, a, b] for fr, a, b in pairs[:15]],
    }

out = os.path.join(PKG, '04_factor_analysis', 'bootstrap_fa_stability_v27.json')
with open(out, 'w', encoding='utf-8') as f:
    # The 'method' description is saved verbatim into the reference JSON.
    json.dump({'features': names_12, 'removed': 'F7',
               'method': 'factor_analyzer.FactorAnalyzer(minres) + varimax，与主分析同口径',
               'summary': summary}, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out}")
