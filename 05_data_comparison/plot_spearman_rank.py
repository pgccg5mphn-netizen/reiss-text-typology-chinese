#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_spearman_rank.py — paper figure: Spearman-correlation scatter plots of
theory-feature ranks vs TF-IDF predicted-probability ranks
(2x3 six panels, rank axes on both sides, greyscale + marker dual encoding,
solid regression line + dashed diagonal).

Layout is identical to the Chinese-version paper figure; FIG_LANG switches the
label language only.
  FIG_LANG=en (default) English labels, for international submission
  FIG_LANG=zh Chinese labels

Data route (final 2026-09-20):
  Theory features = feature_extractor_v9_4 (13 dims)
  Primary baseline (default) = character-level TF-IDF
  (analyzer='char_wb', ngram_range=(2,4), max_features=5000),
  LogisticRegression(C=1.0, class_weight='balanced'),
  StratifiedKFold(5, shuffle=True, random_state=42)
  BASELINE=word switches to the word-level (jieba) baseline.
  Correlation = scipy.stats.spearmanr (equivalent to Pearson on ranks).

NOTE: Chinese strings below (PAPER_RHO/PANELS/FEAT_ZH/CLS_ZH keys and zh-mode
labels) are the shared feature keys and the zh display variant; do not rename.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from scipy import stats

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG, '01_feature_engineering'))
from feature_extractor_v9_4 import TextTypeFeatureExtractor

LANG = os.environ.get('FIG_LANG', 'en')
# Baseline: 'char' (default, char_wb(2,4)) / 'word' (jieba word-level)
BASELINE = os.environ.get('BASELINE', 'char')
# rho shown on the figure: 'paper' uses the paper-final values (char-level, 2026-09-20), 'rerun' uses this run
RHO_MODE = os.environ.get('RHO_MODE', 'paper')

# Paper Section 4.2 final six rho values (char-level baseline, 2026-09-20)
PAPER_RHO = {
    ('F3_第三人称密度', 'expressive'):  0.630,
    ('F6_义务情态密度', 'infective'):   0.580,
    ('F4_事实认知密度', 'informative'): 0.448,
    ('F9_感叹句比例',   'expressive'):  0.328,
    ('F8_祈使句比例',   'infective'):   0.544,
    ('F3_第三人称密度', 'infective'):  -0.538,
}

LABEL_ORDER = ['informative', 'expressive', 'infective']

# (row, col, feature key, target class, Chinese title, English title)
PANELS = [
    (0, 0, 'F3_第三人称密度', 'expressive',  '表情型验证',      'Expressive type validation'),
    (0, 1, 'F6_义务情态密度', 'infective',   '感染型验证',      'Infective type validation'),
    (0, 2, 'F4_事实认知密度', 'informative', '信息型交叉验证',  'Informative type cross-check'),
    (1, 0, 'F9_感叹句比例',   'expressive',  '表情型交叉验证',  'Expressive type cross-check'),
    (1, 1, 'F8_祈使句比例',   'infective',   '感染型交叉验证',  'Infective type cross-check'),
    (1, 2, 'F3_第三人称密度', 'infective',   '感染型交叉验证',  'Infective type cross-check'),
]

FEAT_EN = {
    'F3_第三人称密度': 'F3 Third-person density',
    'F6_义务情态密度': 'F6 Deontic modality density',
    'F4_事实认知密度': 'F4 Factual-cognitive density',
    'F9_感叹句比例':   'F9 Exclamatory sentence ratio',
    'F8_祈使句比例':   'F8 Imperative sentence ratio',
}
FEAT_ZH = {
    'F3_第三人称密度': 'F3 第三人称密度',
    'F6_义务情态密度': 'F6 义务情态密度',
    'F4_事实认知密度': 'F4 事实认知密度',
    'F9_感叹句比例':   'F9 感叹句比例',
    'F8_祈使句比例':   'F8 祈使句比例',
}
CLS_EN = {'informative': 'informative', 'expressive': 'expressive', 'infective': 'infective'}
CLS_ZH = {'informative': '信息型', 'expressive': '表情型', 'infective': '感染型'}

# Marker-shape + greyscale dual encoding (no colour)
STYLE = {
    'expressive':  dict(marker='o', fc='#1A1A1A', ec='none',    s=11, lw=0.0, alpha=0.62, zorder=3),
    'infective':   dict(marker='s', fc='#808080', ec='none',    s=10, lw=0.0, alpha=0.62, zorder=2),
    'informative': dict(marker='^', fc='none',    ec='#9E9E9E', s=13, lw=0.7, alpha=0.75, zorder=1),
}
DRAW_ORDER = ['informative', 'infective', 'expressive']  # light shades drawn first, dark on top


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
        labels.append(ch[0])
    return texts, labels


def main():
    corpus = os.path.join(PKG, 'corpus', 'corpus_relabeled_v27.json')
    texts, labels = load_corpus(corpus)
    y = np.array([LABEL_ORDER.index(l) for l in labels])

    extractor = TextTypeFeatureExtractor()
    X_theory = extractor.extract_batch(texts)
    feat_names = extractor.FEATURE_NAMES

    if BASELINE == 'char':                              # char-level primary baseline (zero word-unit assumption)
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=5000)
        X_tfidf = vec.fit_transform(texts)
        C = 1.0
        ylab_base = '字级 TF-IDF' if LANG == 'zh' else 'char-level TF-IDF'
    else:                                              # word-level strong control (jieba segmentation)
        import jieba
        jieba.initialize()
        X_tfidf = TfidfVectorizer(max_features=5000).fit_transform(
            [' '.join(jieba.cut(t)) for t in texts])
        C = 1.0
        ylab_base = '词级 TF-IDF' if LANG == 'zh' else 'word-level TF-IDF'
    clf = LogisticRegression(C=C, solver='lbfgs', max_iter=1000,
                             class_weight='balanced', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(clf, X_tfidf, y, cv=cv, method='predict_proba')

    n = len(texts)
    print(f'n = {n}   LANG = {LANG}   BASELINE = {BASELINE}   RHO_MODE = {RHO_MODE}')

    rcParams['font.family'] = ['Times New Roman', 'Songti SC']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5))
    results = []

    for row, col, fname, cls, title_zh, title_en in PANELS:
        ax = axes[row, col]
        raw_x = X_theory[:, feat_names.index(fname)]
        raw_y = y_prob[:, LABEL_ORDER.index(cls)]

        rx = stats.rankdata(raw_x, method='average')
        ry = stats.rankdata(raw_y, method='average')
        r, p = stats.spearmanr(raw_x, raw_y)
        results.append((fname, cls, r, p))
        r_show = PAPER_RHO[(fname, cls)] if RHO_MODE == 'paper' else r

        for lab in DRAW_ORDER:
            m = y == LABEL_ORDER.index(lab)
            st = STYLE[lab]
            ax.scatter(rx[m], ry[m], marker=st['marker'], facecolors=st['fc'],
                       edgecolors=st['ec'], s=st['s'], linewidths=st['lw'],
                       alpha=st['alpha'], zorder=st['zorder'],
                       label=(CLS_EN if LANG == 'en' else CLS_ZH)[lab])

        # Linear trend line in rank space
        k, b = np.polyfit(rx, ry, 1)
        ax.plot([0, n], [b, k * n + b], color='#000000', lw=1.3, zorder=4)
        # y = x diagonal reference line
        ax.plot([0, n], [0, n], color='#AAAAAA', lw=0.9, ls='--', zorder=4)

        if LANG == 'en':
            xlab = f'{FEAT_EN[fname]} (rank)'
            ylab = f'{ylab_base} P({CLS_EN[cls]}) (rank)'
            title = f'{title_en}\nSpearman ρ = {r_show:+.3f} (p < 0.001)' if p < 0.001 else f'{title_en}\nSpearman ρ = {r_show:+.3f} (p = {p:.3f})'
        else:
            xlab = f'{FEAT_ZH[fname]} rank'
            ylab = f'{ylab_base} P({CLS_ZH[cls]}) rank'
            title = f'{title_zh}\nSpearman ρ = {r_show:+.3f} (p < 0.001)' if p < 0.001 else f'{title_zh}\nSpearman ρ = {r_show:+.3f} (p = {p:.3f})'

        ax.set_xlabel(xlab, fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=10.5, linespacing=1.5, pad=7)
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
        ax.set_yticks([0, 200, 400, 600, 800, 1000])
        ax.tick_params(labelsize=9, length=3, width=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color('#808080')
            ax.spines[sp].set_linewidth(0.6)
        ax.grid(True, color='#DDDDDD', lw=0.4, alpha=0.7)
        ax.set_axisbelow(True)

        if (row, col) == (0, 0):
            ax.legend(loc='upper left', fontsize=8.5, frameon=True,
                      framealpha=0.9, edgecolor='#DDDDDD', markerscale=1.4,
                      handletextpad=0.3, borderpad=0.35, labelspacing=0.25)

    plt.tight_layout(pad=1.0, h_pad=2.2)
    suffix = '' if BASELINE == 'char' else '_word'
    out = os.path.join(PKG, f'fig_spearman_rank_{LANG}{suffix}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'saved: {out}')
    print('\nPer-panel correlations (this run):')
    for fname, cls, r, p in results:
        print(f'  {fname:<18s} vs {cls:<12s}  r = {r:+.4f}   p = {p:.3g}')


if __name__ == '__main__':
    main()
