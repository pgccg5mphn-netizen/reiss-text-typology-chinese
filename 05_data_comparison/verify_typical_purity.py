#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_typical_purity.py — numerical verification of the paper Section 4.1
"typical passages" values (threshold = 0.53).

Reproduction target (English manuscript P60 / Section 4.1):
  - informative 48.1% / expressive 65.1% / operative 55.5% typical share
  - overall 650 passages (55.8%) typical
  - typical-subset purity mean 0.79 / median 0.80

Configuration: purity = top-class probability minus runner-up probability
(margin); typical = margin >= 0.53. Probabilities are stratified 5-fold
(seed 42) out-of-fold predictions, within-fold standardisation,
LR(C=1.0, balanced).

Usage: python3 verify_typical_purity.py
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, '01_feature_engineering'))

CORPUS = os.path.join(PKG, 'corpus', 'corpus_relabeled_v27.json')
if not os.path.exists(CORPUS):
    CORPUS = os.path.join(PKG, 'corpus', 'demo_corpus.json')
    print('[note] real corpus not found; running on demo corpus (18 passages)')
TH = 0.53

from feature_extractor_v9_4 import TextTypeFeatureExtractor

data = json.load(open(CORPUS, encoding='utf-8'))
texts, labels = [], []
for it in data:
    t = (it.get('data', {}).get('text') or '').strip()
    a = it.get('annotations') or []
    if not t or not a:
        continue
    texts.append(t)
    labels.append(a[0]['result'][0]['value']['choices'][0])
y = np.array(labels)
print(f'n={len(y)}  threshold={TH}')

X = np.array(TextTypeFeatureExtractor().extract_batch(texts))
P = np.zeros((len(y), 3))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                             solver='lbfgs').fit(sc.transform(X[tr]), y[tr])
    P[te] = clf.predict_proba(sc.transform(X[te]))

srt = np.sort(P, axis=1)
margin = srt[:, -1] - srt[:, -2]
typical = margin >= TH

order = ['informative', 'expressive', 'infective']
per_class = {}
for lab in order:
    m = y == lab
    per_class[lab] = {'n': int(m.sum()),
                      'typical_pct': round(100 * float(typical[m].mean()), 1)}
n_typ = int(typical.sum())
sub = margin[typical]
print(f'overall typical: n={n_typ} ({100*n_typ/len(y):.1f}%)  '
      f'mean purity={sub.mean():.2f}  median={np.median(sub):.2f}')
for lab in order:
    print(f'  {lab:12s} n={per_class[lab]["n"]:4d}  typical%={per_class[lab]["typical_pct"]}')

out = {'threshold': TH, 'n': int(len(y)),
       'overall_typical_n': n_typ,
       'overall_typical_pct': round(100 * n_typ / len(y), 1),
       'typical_subset_mean_purity': round(float(sub.mean()), 2),
       'typical_subset_median_purity': round(float(np.median(sub)), 2),
       'per_class': per_class}
with open(os.path.join(HERE, 'verify_typical_purity_out.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print('saved: verify_typical_purity_out.json')
