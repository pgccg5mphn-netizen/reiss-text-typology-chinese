#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_group_split_v27.py
Section 4 supplementary experiments: internal decomposition of the theory
features + baseline hyper-parameter robustness.

Two questions:
1. How much of the 13 features' discriminative power comes from word-list-type
   features (lexical counts) vs structural features?
   - 8 word-list-type dimensions (F1/F2/F3/F4/F5/F6/F8/F10) vs 5 structural
     dimensions (F7/F9/F11/F12/F13)
2. Do the character-level / word-level baseline conclusions depend on C?
   - C sweep over {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}

Configuration identical to data_route_baselines_v27.py (v27 corpus, stratified
random 5-fold, seed=42).
Output: feature_group_split_v27.json

NOTE: the combo names (13维全 / 词表型8维 / ...) and LEX_NAMES / STR_NAMES are
Chinese labels written verbatim into the reference JSON; do not rename.
"""
import os
import sys
import json
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import jieba
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, 'corpus', 'corpus_relabeled_v27.json')
RES_DIR = os.path.join(BASE_DIR, 'wordlists')
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_group_split_v27.json')
sys.path.insert(0, os.path.join(BASE_DIR, '01_feature_engineering'))
from feature_extractor_v9_4 import TextTypeFeatureExtractor  # noqa: E402

LABEL_ORDER = ['informative', 'expressive', 'infective']
SEED = 42
# 13-dim indices: 0 F1 1 F2 2 F3 3 F4 4 F5 5 F6 6 F7 7 F8 8 F9 9 F10 10 F11 11 F12 12 F13
LEX_IDX = [0, 1, 2, 3, 4, 5, 7, 9]   # word-list type: F1/F2/F3/F4/F5/F6/F8/F10
STR_IDX = [6, 8, 10, 11, 12]         # structural type: F7/F9/F11/F12/F13
LEX_NAMES = ['F1第一人称', 'F2第二人称', 'F3第三人称', 'F4事实认知', 'F5情感态度',
             'F6义务情态', 'F8祈使句', 'F10连接词']
STR_NAMES = ['F7陈述句比例', 'F9感叹句比例', 'F11平均句长', 'F12词汇丰富度', 'F13句长变异']


def load_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts, labels = [], []
    for item in data:
        text = (item.get('data', {}).get('text') or '').strip()
        if not text:
            continue
        annos = item.get('annotations') or []
        if not annos or not annos[0].get('result'):
            continue
        choices = annos[0]['result'][0].get('value', {}).get('choices') or []
        if not choices:
            continue
        texts.append(text)
        labels.append(choices[0])
    return texts, labels


def acc_cv(X, y, C, scale, seed=SEED):
    pred = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        if scale:
            sc = StandardScaler().fit(X[tr])
            a, b = sc.transform(X[tr]), sc.transform(X[te])
            clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                                     solver='lbfgs', random_state=SEED)
        else:
            a, b = X[tr], X[te]
            clf = LogisticRegression(C=C, max_iter=1000, class_weight='balanced',
                                     solver='lbfgs', random_state=SEED)
        pred[te] = clf.fit(a, y[tr]).predict(b)
    return accuracy_score(y, pred) * 100


def main():
    texts, labels = load_corpus(CORPUS_PATH)
    y = np.array([LABEL_ORDER.index(l) for l in labels])
    print('n=%d' % len(y))

    ext = TextTypeFeatureExtractor(resource_dir=RES_DIR)
    X13 = ext.extract_batch(texts)
    jieba.initialize()
    jtexts = [' '.join(jieba.cut(t)) for t in texts]

    result = {}

    # ---- Experiment 1: feature group split ----
    print('\n===== Experiment 1: feature group split (word-list type vs structural) =====')
    Xlex, Xstr = X13[:, LEX_IDX], X13[:, STR_IDX]
    combos = {
        '13维全': (X13, 1.0, True),
        '词表型8维': (Xlex, 1.0, True),
        '结构型5维': (Xstr, 1.0, True),
        '词表型+结构型': (np.hstack([Xlex, Xstr]), 1.0, True),
    }
    # Pure word-list counts (no extractor call; word frequencies/densities
    # counted directly from the word lists)
    lens = np.array([max(len(t), 1) for t in texts], dtype=float)
    groups_vocab = {'F1第一人称': ext.pronouns['first'], 'F2第二人称': ext.pronouns['second'],
                    'F3第三人称': ext.pronouns['third'], 'F4事实认知': ext.factual_cognitive_words,
                    'F5情感': ext.emotion_words, 'F6义务情态': ext.deontic_words,
                    'F8祈使标记': ext.imperative_markers, 'F10连接词': ext.connectives}
    cnt = np.array([[sum(t.count(w) for w in ws) for ws in groups_vocab.values()] for t in texts], dtype=float)
    combos['纯词表计数8维'] = (cnt, 1.0, True)
    combos['纯词表密度8维'] = (cnt / lens[:, None] * 1000, 1.0, True)

    result['feature_group_split'] = {}
    for nm, (X, C, sc) in combos.items():
        a = acc_cv(X, y, C, sc)
        result['feature_group_split'][nm] = round(a, 2)
        print('  %-16s dim=%2d  Acc=%.2f%%' % (nm, X.shape[1], a))
    result['feature_group_counts'] = {'lexical': LEX_NAMES, 'structural': STR_NAMES}

    # ---- Experiment 2: C sweep ----
    print('\n===== Experiment 2: baseline C robustness (stratified random 5-fold) =====')
    Xchar = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=5000).fit_transform(texts)
    Xword = TfidfVectorizer(max_features=5000).fit_transform(jtexts)
    result['c_sweep'] = {'char_tfidf': {}, 'word_tfidf_jieba': {}}
    for C in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        ac = acc_cv(Xchar, y, C, False)
        aw = acc_cv(Xword, y, C, False)
        result['c_sweep']['char_tfidf'][str(C)] = round(ac, 2)
        result['c_sweep']['word_tfidf_jieba'][str(C)] = round(aw, 2)
        print('  C=%-4s char=%.2f%%  word=%.2f%%' % (C, ac, aw))

    # ---- Experiment 3: multi-seed robustness (stratified random) ----
    print('\n===== Experiment 3: multi-seed robustness (10 seeds) =====')
    X13r = X13
    result['multiseed'] = {}
    for nm, (X, C, sc) in [('theory_13dim', (X13r, 1.0, True)),
                           ('char_tfidf', (Xchar, 1.0, False)),
                           ('word_tfidf_jieba', (Xword, 1.0, False))]:
        accs = [acc_cv(X, y, C, sc, seed=k) for k in range(10)]
        result['multiseed'][nm] = {'mean': round(float(np.mean(accs)), 2),
                                   'std': round(float(np.std(accs, ddof=1)), 2),
                                   'min': round(float(np.min(accs)), 2),
                                   'max': round(float(np.max(accs)), 2)}
        print('  %-18s %.2f±%.2f  [%.2f, %.2f]' % (nm, np.mean(accs), np.std(accs, ddof=1),
                                                   np.min(accs), np.max(accs)))

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('\nResults written to: %s' % OUT_PATH)


if __name__ == '__main__':
    main()
