#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_confusion_compare.py — manuscript Figure 3: theory route vs character-level
TF-IDF confusion matrices.
Layout matches the original paper figure: 2 side-by-side panels, Greys
colour scale, two lines per cell (count + row-normalised percentage).

Data route (final 2026-09-21, updated with the v9_3 patch):
  Theory features = feature_extractor_v9_4 (13 dims), C=1.0, within-fold
  StandardScaler
  Character-level TF-IDF = char_wb(2,4), max_features=5000, C=1.0
  StratifiedKFold(5, shuffle=True, random_state=42)

FIG_LANG environment variable switches labels: 'zh' (default) Chinese,
'en' English.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG, '01_feature_engineering'))
from feature_extractor_v9_4 import TextTypeFeatureExtractor

LANG = os.environ.get('FIG_LANG', 'zh')
# Chinese labels are the corpus label names (used by the zh variant)
LABELS_ZH = ['信息型', '表情型', '感染型']
LABELS_EN = ['Informative', 'Expressive', 'Infective']
LABEL_ORDER = ['informative', 'expressive', 'infective']


def load_corpus(path):
    data = json.load(open(path, encoding='utf-8'))
    texts, labels = [], []
    for item in data:
        text = item.get('data', {}).get('text', '').strip()
        if not text:
            continue
        annos = item.get('annotations', [])
        if not annos:
            continue
        res = annos[0].get('result', [])
        if not res:
            continue
        ch = res[0].get('value', {}).get('choices', [])
        if not ch:
            continue
        texts.append(text)
        labels.append(LABEL_ORDER.index(ch[0]))
    return texts, np.array(labels)


def oof_pred(X, y, C, scale):
    p = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        if scale:
            sc = StandardScaler().fit(X[tr])
            a, b = sc.transform(X[tr]), sc.transform(X[te])
        else:
            a, b = X[tr], X[te]
        m = LogisticRegression(C=C, max_iter=1000, class_weight='balanced', random_state=42)
        p[te] = m.fit(a, y[tr]).predict(b)
    return p


def main():
    texts, y = load_corpus(os.path.join(PKG, 'corpus', 'corpus_relabeled_v27.json'))
    n = len(y)
    ext = TextTypeFeatureExtractor(resource_dir=os.path.join(PKG, 'wordlists'))
    X13 = ext.extract_batch(texts)
    Xc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=5000).fit_transform(texts)

    cm_t = confusion_matrix(y, oof_pred(X13, y, 1.0, True))
    cm_c = confusion_matrix(y, oof_pred(Xc, y, 1.0, False))
    print('Theory:\n', cm_t, '\nChar-level:\n', cm_c)

    labels = LABELS_ZH if LANG == 'zh' else LABELS_EN
    t1 = '理论路线' if LANG == 'zh' else '(a) Theory-driven model'
    t2 = '字级 TF-IDF' if LANG == 'zh' else '(b) Character-level TF-IDF baseline'

    plt.rcParams['font.family'] = ['Times New Roman', 'Songti SC']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0))
    for ax, cm, title in [(axes[0], cm_t, t1), (axes[1], cm_c, t2)]:
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(pct, cmap='Greys', vmin=0, vmax=100, aspect='auto')
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_yticklabels(labels, fontsize=14)
        ax.set_xlabel('预测类别' if LANG == 'zh' else 'Predicted', fontsize=15, fontweight='bold')
        ax.set_ylabel('真实类别' if LANG == 'zh' else 'True', fontsize=15, fontweight='bold')
        ax.set_title(title, fontsize=17, fontweight='bold', pad=12)
        for i in range(3):
            for j in range(3):
                c = 'white' if pct[i, j] > 50 else 'black'
                ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                        fontsize=20, fontweight='bold', color=c)
                ax.text(j, i + 0.32, f'({pct[i, j]:.1f}%)', ha='center', va='center',
                        fontsize=12, color=c)
        ax.tick_params(top=False, bottom=False, left=False, right=False,
                       labeltop=False, labelbottom=True, labelleft=True, labelright=False)
        for sp in ax.spines.values():
            sp.set_linewidth(1.5); sp.set_color('black')

    plt.tight_layout()
    suffix = '' if LANG == 'zh' else '_en'
    out = os.path.join(PKG, f'fig3_confusion_compare{suffix}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'saved: {out}')


if __name__ == '__main__':
    main()
