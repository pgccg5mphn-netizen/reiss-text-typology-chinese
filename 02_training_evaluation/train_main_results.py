#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_main_results.py
Full cascaded recomputation after the v9_4 finalisation (sentence-final
punctuation retained + F8 switched to jieba token matching).

Purpose: after the paper adopted v9_4, recompute all affected metrics.
Affected: F7/F8/F9/F11/F13 (other dimensions unchanged one by one) -> overall
accuracy, per-class metrics, single-feature/layer ablation, factor analysis
(12 dimensions with F7 removed).

Configuration (identical to the paper):
- Corpus corpus/corpus_relabeled_v27.json (n=1165; a demo corpus
  demo_corpus.json is also included in the package)
- Features: 13 dimensions; LogisticRegression C=1.0, class_weight='balanced',
  max_iter=1000, solver='lbfgs'
- CV: StratifiedKFold(5, shuffle=True, random_state=42), StandardScaler
  fitted within each fold
- Factor analysis: 12 dimensions (F7 removed), number of factors determined by
  parallel analysis (95th percentile), Varimax rotation, |lambda| >= 0.4
"""
import os
import sys
import json
import numpy as np
import scipy
scipy.sum = np.sum  # compatibility patch: factor_analyzer 0.3.1 + scipy>=1.12

HERE = os.path.dirname(os.path.abspath(__file__))
FE = os.path.abspath(os.path.join(HERE, '..', '01_feature_engineering'))
sys.path.insert(0, FE)
from feature_extractor_v9_4 import TextTypeFeatureExtractor

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

CORPUS = os.path.abspath(os.path.join(HERE, '..', 'corpus', 'corpus_relabeled_v27.json'))
OUT = os.path.join(HERE, 'main_results.json')

ext = TextTypeFeatureExtractor()
with open(CORPUS, encoding='utf-8') as f:
    data = json.load(f)
T, L = [], []
for it in data:
    dd = it.get('data', {})
    a = it.get('annotations') or []
    t = (dd.get('text') or '').strip()
    if not t or not a or not a[0].get('result'):
        continue
    ch = a[0]['result'][0].get('value', {}).get('choices') or []
    if not ch:
        continue
    T.append(t); L.append(ch[0])
y = np.array(L)
NAMES = list(ext.FEATURE_NAMES)
X = np.array(ext.extract_batch(T))
print('n=%d  classes=%s' % (len(T), {c: int((y == c).sum()) for c in ['informative', 'expressive', 'infective']}))


def cv(Xm, drop_idx=None):
    Xw = Xm if drop_idx is None else np.delete(Xm, drop_idx, axis=1)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    pred = np.empty(len(y), dtype=object)
    for tr, te in skf.split(Xw, y):
        sc = StandardScaler().fit(Xw[tr])
        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs')
        clf.fit(sc.transform(Xw[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(Xw[te]))
    return accuracy_score(y, pred) * 100, f1_score(y, pred, average='macro'), pred


base, macro, pred = cv(X)
p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=['informative', 'expressive', 'infective'])
out = {
    'overall_acc': round(base, 4),
    'macro_f1': round(macro, 4),
    'per_class': {c: {'P': round(p[i]*100, 2), 'R': round(r[i]*100, 2), 'F1': round(f1[i]*100, 2)}
                  for i, c in enumerate(['informative', 'expressive', 'infective'])},
    'per_feature_ablation': {},
    'per_group_ablation': {},
    'factor_analysis': {},
}
print('Overall Acc=%.2f%%  macroF1=%.4f' % (base, macro))
for i, fn in enumerate(NAMES):
    d = base - cv(X, i)[0]
    out['per_feature_ablation'][fn] = round(d, 2)
print('Single-feature ablation:', out['per_feature_ablation'])
# Layer keys are Chinese because the layer names double as the ablation keys in
# the saved JSON (person reference / modality type / sentence strategy /
# information organisation / formal variation).
GROUPS = {'人称指向': [0, 1, 2], '情态类型': [3, 4, 5], '句式策略': [6, 7, 8], '信息组织': [9, 10], '形式变异': [11, 12]}
for k, idx in GROUPS.items():
    out['per_group_ablation'][k] = round(base - cv(np.delete(X, idx, axis=1))[0], 2)
print('Layer-level ablation:', out['per_group_ablation'])

# ---- Factor analysis (12 dimensions, F7 removed) ----
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

    def horn_p95(Xk, nrep=200, seed=42):
        rng = np.random.RandomState(seed); N, P = Xk.shape; ev = []
        for _ in range(nrep):
            R = rng.normal(size=(N, P))
            ev.append(np.sort(np.linalg.eigvalsh(np.corrcoef(R, rowvar=False)))[::-1])
        return np.percentile(np.array(ev), 95, axis=0)

    keep = [i for i in range(13) if i != 6]  # remove F7
    Fn = [NAMES[i] for i in keep]
    for c in ['informative', 'expressive', 'infective']:
        Xk = X[y == c][:, keep]
        kmo = calculate_kmo(Xk)[1]
        chi, pv = calculate_bartlett_sphericity(Xk)
        p95 = horn_p95(Xk)
        real = np.sort(np.linalg.eigvalsh(np.corrcoef(Xk, rowvar=False)))[::-1]
        nf = int(sum(real[j] > p95[j] for j in range(len(real))))
        fa = FactorAnalyzer(n_factors=nf, rotation='varimax').fit(Xk)
        Lm = fa.loadings_
        vc = np.cumsum(fa.get_factor_variance()[1]) * 100
        facs = []
        for fi in range(nf):
            facs.append({Fn[j]: round(float(Lm[j, fi]), 3) for j in range(len(Fn)) if abs(Lm[j, fi]) >= 0.4})
        out['factor_analysis'][c] = {'n': int(Xk.shape[0]), 'kmo': round(float(kmo), 4),
                                     'bartlett_p': float(pv), 'n_factors': nf,
                                     'cum_variance': [round(float(v), 2) for v in vc], 'loadings': facs}
        print('  FA %-12s n=%d KMO=%.3f nfac=%d cum=%.1f%%' % (c, Xk.shape[0], kmo, nf, vc[-1]))
except Exception as e:
    print('Factor analysis failed:', e)

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('Written:', OUT)
