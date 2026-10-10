#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full verification of the ablation results (v27 + Scaler; features zeroed
and retrained; 5-fold CV)."""
import os
import json
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '01_feature_engineering')))
from feature_extractor_v9_4 import TextTypeFeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

with open(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'corpus', 'corpus_relabeled_v27.json')), 'r', encoding='utf-8') as f:
    data = json.load(f)

texts, labels = [], []
for item in data:
    text = item.get('data', {}).get('text', '').strip()
    annos = item.get('annotations', [])
    if not text or not annos:
        continue
    results = annos[0].get('result', [])
    if not results:
        continue
    choices = results[0].get('value', {}).get('choices', [])
    if not choices:
        continue
    texts.append(text)
    labels.append(choices[0])

label_map = {'informative': 0, 'expressive': 1, 'infective': 2}
inv_map = {0: 'informative', 1: 'expressive', 2: 'infective'}
y = np.array([label_map[l] for l in labels])

extractor = TextTypeFeatureExtractor()
X = np.array([extractor.extract(t) for t in texts])
names = extractor.FEATURE_NAMES
print(f'Corpus: {len(texts)} passages, X: {X.shape}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_5fold(X_, y_):
    """5-fold CV; return (mean_acc, std_acc, per-fold acc list)."""
    accs = []
    for tr, te in skf.split(X_, y_):
        s = StandardScaler()
        Xt = s.fit_transform(X_[tr])
        Xv = s.transform(X_[te])
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=42)
        clf.fit(Xt, y_[tr])
        accs.append(accuracy_score(y_[te], clf.predict(Xv)))
    return np.mean(accs), np.std(accs), accs

# Baseline
base_mean, base_std, base_accs = cv_5fold(X, y)
print(f'\nBaseline (13-dim): {base_mean:.4f} ± {base_std:.4f}')
print(f'  Per fold: {[f"{a:.4f}" for a in base_accs]}')

# Layer-level ablation (zero out the layer's features and retrain)
print(f'\n==== Layer-level ablation (zeroed and retrained) ====')
layers = {
    'Person reference (F1-F3)': [0, 1, 2],
    'Modality type (F4-F6)': [3, 4, 5],
    'Sentence strategy (F7-F9)': [6, 7, 8],
    'Information organisation (F10-F11)': [9, 10],
    'Formal variation (F12-F13)': [11, 12],
}
layer_results = {}
for lname, idxs in layers.items():
    Xz = X.copy()
    Xz[:, idxs] = 0.0
    m, s, _ = cv_5fold(Xz, y)
    delta = (base_mean - m) * 100
    layer_results[lname] = (m, delta)
    print(f'{lname:>36}: {m:.4f} ± {s:.4f}  delta={delta:+.2f}pp')

# Single-feature ablation
print(f'\n==== Single-feature ablation (zeroed and retrained) ====')
feat_results = []
for i in range(13):
    Xz = X.copy()
    Xz[:, i] = 0.0
    m, s, _ = cv_5fold(Xz, y)
    delta = (base_mean - m) * 100
    feat_results.append((i, names[i], m, s, delta))

feat_results.sort(key=lambda x: x[4], reverse=True)
for i, name, m, s, delta in feat_results:
    print(f'{name:>20}: {m:.4f} ± {s:.4f}  delta={delta:+.2f}pp')

# Coefficient matrix (5-fold average)
print(f'\n==== Coefficient matrix (5-fold average) ====')
coefs = []
for tr, te in skf.split(X, y):
    s = StandardScaler()
    Xt = s.fit_transform(X[tr])
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=42)
    clf.fit(Xt, y[tr])
    coefs.append(clf.coef_)
coef_mean = np.mean(coefs, axis=0)

# Column heads are Chinese because the feature names (rows) are the shared
# Chinese keys; classes follow the same label order as the corpus.
print(f'{"Feature":>20}  {"Informative":>12}  {"Expressive":>12}  {"Operative":>12}')
for i, name in enumerate(names):
    print(f'{name:>20}  {coef_mean[0][i]:>+12.4f}  {coef_mean[1][i]:>+12.4f}  {coef_mean[2][i]:>+12.4f}')

# Key-coefficient checks
print(f'\n==== Key coefficient checks ====')
print(f'F3 (third person) expressive coefficient: {coef_mean[1][2]:+.4f}  (paper: +0.98)')
print(f'F3 (third person) operative coefficient: {coef_mean[2][2]:+.4f}  (paper: -1.00)')
print(f'F6 (deontic modality) operative coefficient: {coef_mean[2][5]:+.4f}')
print(f'F1 (first person) expressive coefficient: {coef_mean[1][0]:+.4f}')
