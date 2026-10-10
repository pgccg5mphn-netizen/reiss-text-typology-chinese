#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_random13_control.py — verification of the paper Section 4.1 control
"13 random features".

Reproduction target (English manuscript Section 4.1):
  a classifier on 13 random features scores ~33% (chance level), below the
  majority-class baseline 40.69% and the 13-dim theory model 77.17%.

Configuration: features are standard-normal random values (independent of the
corpus); stratified 5-fold (seed 42), within-fold standardisation,
LR(C=1.0, balanced); averaged over 10 different random seeds.
Original run record (the original construction script altframe.py is lost):
10-run mean 33.02% (range 28.41-37.00). This rebuilt script reproduces the
chance-level conclusion within random variation.

Usage: python3 verify_random13_control.py
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

CORPUS = os.path.join(PKG, 'corpus', 'corpus_relabeled_v27.json')
if not os.path.exists(CORPUS):
    CORPUS = os.path.join(PKG, 'corpus', 'demo_corpus.json')
    print('[note] real corpus not found; running on demo corpus (18 passages)')
data = json.load(open(CORPUS, encoding='utf-8'))
y = np.array([it['annotations'][0]['result'][0]['value']['choices'][0]
              for it in data
              if it.get('annotations') and it.get('data', {}).get('text', '').strip()])
print(f'n={len(y)}  majority baseline = {100*max(np.bincount([list(np.unique(y)).index(v) for v in y]))/len(y):.2f}%')

accs = []
for run in range(10):
    rng = np.random.RandomState(run)
    Xr = rng.randn(len(y), 13)
    pred = np.zeros(len(y), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Xr, y):
        sc = StandardScaler().fit(Xr[tr])
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                                 solver='lbfgs').fit(sc.transform(Xr[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(Xr[te]))
    accs.append(100 * accuracy_score(y, pred))

print('10 runs:', [round(a, 2) for a in accs])
print(f'mean {np.mean(accs):.2f}%  range {min(accs):.2f}-{max(accs):.2f}%')
print('(paper: 33.02%, range 28.41-37.00; chance level = 1/3)')

with open(os.path.join(HERE, 'verify_random13_control_out.json'), 'w', encoding='utf-8') as f:
    json.dump({'accs': [round(a, 2) for a in accs],
               'mean': round(float(np.mean(accs)), 2),
               'min': round(float(min(accs)), 2),
               'max': round(float(max(accs)), 2)}, f, indent=1)
print('saved: verify_random13_control_out.json')
