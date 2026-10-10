#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_multiseed5.py — verification of the paper Section 4.1 claim
"5 seeds including seed 42".

Reproduction target (English manuscript Section 4.1):
  5-fold mean accuracy across 5 random seeds (42/7/123/2024/999)
  falls within 76.1-77.2 per cent.

Configuration: identical to the main experiment (13 theory features,
within-fold standardisation, LR(C=1.0, balanced, max_iter=1000, lbfgs));
only the split seed varies.

Usage: python3 verify_multiseed5.py
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, '01_feature_engineering'))

CORPUS = os.path.join(PKG, 'corpus', 'corpus_relabeled_v27.json')
if not os.path.exists(CORPUS):
    CORPUS = os.path.join(PKG, 'corpus', 'demo_corpus.json')
    print('[note] real corpus not found; running on demo corpus (18 passages)')

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
X = np.array(TextTypeFeatureExtractor().extract_batch(texts))

SEEDS = [42, 7, 123, 2024, 999]
means = {}
for seed in SEEDS:
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                                 solver='lbfgs').fit(sc.transform(X[tr]), y[tr])
        accs.append(100 * accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
    means[seed] = round(float(np.mean(accs)), 2)
    print(f'seed {seed}: mean {means[seed]:.2f}')

lo, hi = min(means.values()), max(means.values())
print(f'range across seeds: {lo:.1f} - {hi:.1f}  (paper: 76.1 - 77.2)')

with open(os.path.join(HERE, 'verify_multiseed5_out.json'), 'w', encoding='utf-8') as f:
    json.dump({'seeds': means, 'range': [lo, hi]}, f, indent=1)
print('saved: verify_multiseed5_out.json')
