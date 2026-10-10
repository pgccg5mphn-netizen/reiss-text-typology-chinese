#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_spearman_39pairs.py — verification of the paper Section 4.4 claim
"32 of the 39 pairs have p < .001".

Reproduction target (English manuscript Section 4.4):
  13 theory features x 3 class probabilities = 39 feature-probability
  combinations; after Spearman rank correlation, exactly 32 pairs have
  p < .001.

Configuration: probabilities are stratified 5-fold (seed 42) out-of-fold
predictions of the character-level TF-IDF baseline (char_wb 2-4 grams,
max_features=5000, LR(C=1.0, balanced)).

Usage: python3 verify_spearman_39pairs.py
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr

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

X13 = np.array(TextTypeFeatureExtractor().extract_batch(texts))
Xc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                     max_features=5000).fit_transform(texts)
P = np.zeros((len(y), 3))
classes_sorted = np.array(sorted(set(y)))          # sklearn alphabetical class order
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Xc, y):
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                             solver='lbfgs').fit(Xc[tr], y[tr])
    P[te] = clf.predict_proba(Xc[te])

names = [f'F{i+1}' for i in range(13)]
rows, sig = [], 0
for fi, fname in enumerate(names):
    for ci, cname in enumerate(classes_sorted):
        r, p = spearmanr(X13[:, fi], P[:, ci])
        sig += int(p < 0.001)
        rows.append({'feature': fname, 'class': str(cname),
                     'rho': round(float(r), 3), 'p': float(p)})
print(f'total pairs tested: {len(rows)}')
print(f'pairs with p < .001: {sig}')

rows_sorted = sorted(rows, key=lambda x: -abs(x['rho']))
print('|rho| top 6 (cf. Fig. 1):')
for r in rows_sorted[:6]:
    print(f'  {r["feature"]:>4s} ↔ {r["class"]:12s}  rho={r["rho"]:+.3f}  p={r["p"]:.2e}')

with open(os.path.join(HERE, 'verify_spearman_39pairs_out.json'), 'w', encoding='utf-8') as f:
    json.dump({'n_pairs': len(rows), 'n_sig_p001': sig, 'pairs': rows},
              f, ensure_ascii=False, indent=1)
print('saved: verify_spearman_39pairs_out.json')
