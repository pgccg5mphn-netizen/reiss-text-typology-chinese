#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genre_internal_tests_v27.py
Reproduction script for the three "genre-dimension internal test" numbers of
Section 4 (v9_4 final configuration).

The three numbers (paper ref [52]):
  1. Recall for non-initiative operative passages after retraining without the
     government/institution initiative texts            71.4%
  2. Balanced accuracy, informative vs operative, within the
     government/institution initiative source            63.7%
  3. Balanced accuracy, informative vs expressive, within
     the brand-statement source                          90.9%

Protocol (identical to the main experiment; not in-sample, not a single
hold-out):
  - 13 theory-driven features (feature_extractor_v9_4), stratified random
    5-fold, StratifiedKFold(5, shuffle=True, random_state=42)
  - StandardScaler fitted on the training folds only, then applied to the test
    fold (within-fold standardisation; essential)
  - LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs')

Caution: without within-fold standardisation of the 13 features the three
numbers collapse to 62.9 / 52.6 / 36.9, far off the paper values. The
within-fold StandardScaler is therefore a necessary condition for this test.

NOTE: '政府/机构倡议书' and '品牌宣言' below are literal source_file values of
the Chinese corpus and must stay as-is.
"""
import os
import sys
import json
import warnings

import numpy as np

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(BASE, 'corpus', 'corpus_relabeled_v27.json')
sys.path.insert(0, os.path.join(BASE, '01_feature_engineering'))
from feature_extractor_v9_4 import TextTypeFeatureExtractor  # noqa: E402

SEED = 42


def load():
    with open(CORPUS, encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for it in data:
        d = it.get('data', {})
        t = (d.get('text') or '').strip()
        if not t:
            continue
        a = it.get('annotations') or []
        if not a:
            continue
        rows.append((d, a[0]['result'][0]['value']['choices'][0]))
    return rows


def cv_predict(X, y, seed=SEED):
    """Stratified random 5-fold + within-fold StandardScaler + LR(balanced);
    return out-of-fold predictions."""
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000,
                                 solver='lbfgs', random_state=SEED)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return pred


def main():
    rows = load()
    texts = [r[0]['text'] for r in rows]
    labels = np.array([r[1] for r in rows])
    srcs = np.array([r[0].get('source_file') for r in rows])

    ex = TextTypeFeatureExtractor(resource_dir=os.path.join(BASE, 'wordlists'))
    X = ex.extract_batch(texts)
    print('Corpus n=%d  feature matrix %s' % (len(labels), X.shape))

    # Chinese source names are literal corpus values
    gov = srcs == '政府/机构倡议书'
    brand = srcs == '品牌宣言'
    out = {}

    # 1 Retrain without initiative texts -> recall for non-initiative operative
    Xr, yr = X[~gov], labels[~gov]
    p = cv_predict(Xr, yr)
    r_inf = recall_score(yr, p, labels=['infective'], average='macro', zero_division=0)
    print('[1] Non-initiative operative recall = %.2f%%  (n=%d, operative %d)'
          % (r_inf * 100, len(yr), int((yr == 'infective').sum())))
    out['non_gov_infective_recall'] = round(r_inf, 4)

    # 2 Within initiative texts: informative vs operative
    k = np.isin(labels[gov], ['informative', 'infective'])
    b = balanced_accuracy_score(labels[gov][k], cv_predict(X[gov][k], labels[gov][k]))
    print('[2] Within-initiative info-vs-inf balanced accuracy = %.2f%%  (n=%d)'
          % (b * 100, int(k.sum())))
    out['gov_info_vs_inf_balacc'] = round(b, 4)

    # 3 Within brand statements: informative vs expressive
    k2 = np.isin(labels[brand], ['informative', 'expressive'])
    b2 = balanced_accuracy_score(labels[brand][k2], cv_predict(X[brand][k2], labels[brand][k2]))
    print('[3] Within-brand info-vs-exp balanced accuracy = %.2f%%  (n=%d)'
          % (b2 * 100, int(k2.sum())))
    out['brand_info_vs_exp_balacc'] = round(b2, 4)

    print('\nPaper values 71.4 / 63.7 / 90.9 ; this run %.1f / %.1f / %.1f'
          % (r_inf * 100, b * 100, b2 * 100))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'genre_internal_tests_v27.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
