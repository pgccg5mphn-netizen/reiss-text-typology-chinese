#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_figures_en_pub.py — the three HSSC submission figures (publication-grade
unified style)
  Figure 1  factor-loading dot plot (data: outputs_reference/
            factor_analysis_12dim_v27.json, the final version)
  Figure 2  Spearman rank-correlation scatter (runs the corpus pipeline; same
            source as plot_spearman_rank.py)
  Figure 3  confusion-matrix comparison (runs the corpus pipeline; same source
            as plot_confusion_compare.py)

Style follows published HSSC articles (s41599-025-05562-9 / -06424-0 / -05814-8):
  white background + light-grey thin grid + no top/right spines + sans-serif
  Arial + low-saturation colour-blind-safe palette
  Fixed type colours: informative #6387A6 / expressive #D97676 / operative #E09C50
Output: PNG files in this directory (300 dpi); copy them to
HSSC_submission/figures/ for submission.

NOTE: Chinese keys below (ZH2EN, FEATURE_EN, PAPER_RHO, PANELS) are the Chinese
feature/label keys shared with the reference JSON; do not rename.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PKG, '01_feature_engineering'))

TYPE_COLOR = {'informative': '#6387A6', 'expressive': '#D97676', 'operative': '#E09C50'}
CMAP_SEQ = ['#FFFFFF', '#C7D6E4', '#4F7699']  # confusion matrices: white -> light steel blue -> deep steel blue (soft monochrome ramp)
ZH2EN = {'信息型': 'informative', '表情型': 'expressive', '感染型': 'operative'}
MARKER = {'informative': 'o', 'expressive': '^', 'operative': 's'}

FEATURE_EN = {
    'F1_第一人称密度':  'F1 First-person density',
    'F2_第二人称密度':  'F2 Second-person density',
    'F3_第三人称密度':  'F3 Third-person density',
    'F4_事实认知密度':  'F4 Factual-cognitive density',
    'F5_情感态度密度':  'F5 Affective-attitudinal density',
    'F6_义务情态密度':  'F6 Deontic modality density',
    'F8_祈使句比例':    'F8 Imperative sentence ratio',
    'F9_感叹句比例':    'F9 Exclamatory sentence ratio',
    'F10_连接词密度':   'F10 Connective density',
    'F11_平均句长':     'F11 Mean sentence length',
    'F12_词汇丰富度':   'F12 Lexical richness',
    'F13_句长变异系数': 'F13 Sentence-length CV',
}

def apply_style():
    from matplotlib import rcParams
    rcParams.update({
        'font.family': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'custom',
        'axes.unicode_minus': False,
        'axes.facecolor': 'white',
        'axes.edgecolor': '#9A9A9A', 'axes.linewidth': 0.7,
        'axes.grid': True, 'grid.color': '#E7E7E7', 'grid.linewidth': 0.55,
        'axes.axisbelow': True,
        'xtick.color': '#555555', 'ytick.color': '#555555',
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'axes.labelsize': 9.5, 'axes.labelcolor': '#222222',
        'axes.titlesize': 10, 'axes.titlecolor': '#1A1A1A',
        'legend.frameon': True, 'legend.edgecolor': '#DDDDDD',
        'legend.framealpha': 0.95, 'legend.fontsize': 8.5,
        'savefig.dpi': 300, 'savefig.facecolor': 'white',
    })
    def despine(ax):
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    return despine

# ---------------------------------------------------------------- Figure 1
def fig_loadings():
    despine = apply_style()
    FA_JSON = [os.path.join(PKG, 'outputs_reference', 'factor_analysis_12dim_v27.json'),
             os.path.join(PKG, '04_factor_analysis', 'factor_analysis_12dim_v27.json')]
    fa_path = next(p for p in FA_JSON if os.path.exists(p))
    d = json.load(open(fa_path, encoding='utf-8'))
    feats = d['features']
    TH = 0.40
    order = {'信息型': ('a', 'Informative'), '表情型': ('b', 'Expressive'), '感染型': ('c', 'Operative')}

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.7), sharey=True)
    for ax, (zh, res) in zip(axes, d['results'].items()):
        en = ZH2EN[zh]; tag, name = order[zh]
        A = np.array(res['loadings'])          # 12 x k
        dom_idx = np.array([int(np.argmax(np.abs(A[i]))) for i in range(len(feats))])
        dom = np.array([A[i][dom_idx[i]] for i in range(len(feats))])
        y = np.arange(len(feats))

        ax.axvspan(-TH, TH, color='#F1F1F1', zorder=0)
        for xv in (-TH, TH):
            ax.axvline(xv, color='#C9C9C9', lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.axvline(0, color='#B5B5B5', lw=0.8, zorder=1)

        sal = np.abs(dom) >= TH
        sub = ~sal
        ax.scatter(dom[sub], y[sub], s=24, facecolors='none', edgecolors='#B8B8B8',
                   linewidths=1.0, zorder=3)
        ax.scatter(dom[sal], y[sal], s=64, color=TYPE_COLOR[en], edgecolors='white',
                   linewidths=0.8, zorder=4)
        for v, yi, fi in zip(dom[sal], y[sal], dom_idx[sal]):
            off = 0.06 if v >= 0 else -0.06
            ha = 'left' if v >= 0 else 'right'
            ax.text(v + off, yi, f'{v:+.2f}', va='center', ha=ha,
                    fontsize=7.3, color='#444444', zorder=5)
            ax.text(v, yi - 0.34, f'f{fi + 1}', va='center', ha='center',
                    fontsize=7.5, style='italic', color='#666666', zorder=5)

        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(len(feats) - 0.5, -0.5)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_title(f'({tag}) {name}  (n = {res["n"]}; {res["n_factors"]} factors)', pad=8)
        ax.grid(False)
        ax.grid(axis='x', color='#E7E7E7', lw=0.55)
        despine(ax)
        if zh == '信息型':
            ax.set_yticks(y)
            ax.set_yticklabels([FEATURE_EN[f] for f in feats], fontsize=8.5)
            ax.set_ylabel('Varimax-rotated loading', fontsize=9.5)
    fig.text(0.5, -0.015,
             'Loading on the dominant factor of each feature (f-numbers denote factor membership); open grey dots: loadings below the |λ| = 0.40 reporting threshold',
             ha='center', fontsize=8.3, color='#555555')
    fig.tight_layout(pad=0.9, w_pad=1.6)
    out = os.path.join(HERE, 'fig1_factor_loadings_en.png')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print('saved:', out)
    # return salient values for verification
    salient = {}
    for zh, res in d['results'].items():
        A = np.array(res['loadings'])
        salient[zh] = {feats[i]: round(float(A[i][np.argmax(np.abs(A[i]))]), 2)
                       for i in range(len(feats)) if abs(A[i][np.argmax(np.abs(A[i]))]) >= TH}
    return salient

# ---------------------------------------------------------------- corpus pipeline
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

PAPER_RHO = {
    ('F3_第三人称密度', 'expressive'):  0.630,
    ('F6_义务情态密度', 'operative'):   0.580,
    ('F4_事实认知密度', 'informative'): 0.448,
    ('F9_感叹句比例',   'expressive'):  0.328,
    ('F8_祈使句比例',   'operative'):   0.544,
    ('F3_第三人称密度', 'operative'):  -0.538,
}
PANELS = [
    ('a', 'F3_第三人称密度', 'expressive',  'F3 × expressive'),
    ('b', 'F6_义务情态密度', 'operative',   'F6 × operative'),
    ('c', 'F4_事实认知密度', 'informative', 'F4 × informative'),
    ('d', 'F9_感叹句比例',   'expressive',  'F9 × expressive'),
    ('e', 'F8_祈使句比例',   'operative',   'F8 × operative'),
    ('f', 'F3_第三人称密度', 'operative',   'F3 × operative'),
]

def fig_spearman():
    despine = apply_style()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from scipy import stats
    from feature_extractor_v9_4 import TextTypeFeatureExtractor

    LABEL_ORDER = ['informative', 'expressive', 'infective']
    texts, labels = load_corpus(os.path.join(PKG, 'corpus', 'corpus_relabeled_v27.json'))
    y = np.array([LABEL_ORDER.index(l) for l in labels])
    ext = TextTypeFeatureExtractor()
    X = ext.extract_batch(texts)
    names = ext.FEATURE_NAMES
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=5000)
    Xc = vec.fit_transform(texts)
    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                             class_weight='balanced', random_state=42)
    y_prob = cross_val_predict(clf, Xc, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                               method='predict_proba')
    n = len(texts)
    print('corpus n =', n)
    internal = {'operative': 'infective'}  # display name -> corpus-internal label key

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2))
    for idx, (tag, fname, cls, short) in enumerate(PANELS):
        ax = axes[idx // 3, idx % 3]
        rx = stats.rankdata(X[:, names.index(fname)], method='average')
        ry = stats.rankdata(y_prob[:, LABEL_ORDER.index(internal.get(cls, cls))], method='average')
        r, p = stats.spearmanr(rx, ry)
        paper = PAPER_RHO[(fname, cls)]
        assert abs(r - paper) < 0.005, f'rho drift {fname}/{cls}: rerun {r:+.4f} vs paper {paper}'
        for lab_disp, lab in [('Informative', 'informative'), ('Operative', 'infective'),
                              ('Expressive', 'expressive')]:
            m = y == LABEL_ORDER.index(lab)
            ax.scatter(rx[m], ry[m], s=13, marker=MARKER[lab.replace('infective', 'operative')],
                       linewidths=0, color=TYPE_COLOR[lab.replace('infective', 'operative')],
                       alpha=0.5, zorder=3,
                       label=lab_disp if idx == 0 else None)
        k, b = np.polyfit(rx, ry, 1)
        ax.plot([0, n], [b, k * n + b], color='#333333', lw=1.2, zorder=4)
        ax.plot([0, n], [0, n], color='#B9B9B9', lw=0.9, ls='--', zorder=4)
        ax.set_xlim(0, n); ax.set_ylim(0, n)
        ax.set_xticks([0, 250, 500, 750, 1000]); ax.set_yticks([0, 250, 500, 750, 1000])
        ax.set_title(f'({tag}) {short}', pad=6)
        ax.text(0.045, 0.955, f'ρ = {paper:+.3f}', transform=ax.transAxes,
                ha='left', va='top', fontsize=9.5,
                bbox=dict(boxstyle='round,pad=0.32', fc='white', ec='#CCCCCC', lw=0.7))
        ax.set_xlabel('Feature value (rank)', fontsize=9)
        ax.set_ylabel('Class probability (rank)', fontsize=9)
        despine(ax)
    axes[0, 0].legend(loc='lower right', handletextpad=0.25, borderpad=0.4)
    fig.tight_layout(pad=0.9, h_pad=1.6, w_pad=1.4)
    out = os.path.join(HERE, 'fig2_spearman_en_pub.png')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print('saved:', out)

def fig_confusion():
    despine = apply_style()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix
    from feature_extractor_v9_4 import TextTypeFeatureExtractor

    LABEL_ORDER = ['informative', 'expressive', 'infective']
    texts, y = load_corpus(os.path.join(PKG, 'corpus', 'corpus_relabeled_v27.json'))
    y = np.array([LABEL_ORDER.index(l) for l in y])
    ext = TextTypeFeatureExtractor(resource_dir=os.path.join(PKG, 'wordlists'))
    X13 = ext.extract_batch(texts)
    Xc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=5000).fit_transform(texts)

    def oof_pred(X, C, scale):
        p = np.zeros(len(y), dtype=int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
            if scale:
                sc = StandardScaler().fit(X[tr]); a, b = sc.transform(X[tr]), sc.transform(X[te])
            else:
                a, b = X[tr], X[te]
            p[te] = LogisticRegression(C=C, max_iter=1000, class_weight='balanced',
                                       random_state=42).fit(a, y[tr]).predict(b)
        return p

    cm_t = confusion_matrix(y, oof_pred(X13, 1.0, True))
    cm_c = confusion_matrix(y, oof_pred(Xc, 1.0, False))
    expect_t = np.array([[345, 62, 67], [40, 330, 31], [50, 16, 224]])
    expect_c = np.array([[417, 41, 16], [48, 348, 5], [14, 20, 256]])
    assert (cm_t == expect_t).all(), f'theory CM drift:\n{cm_t}'
    assert (cm_c == expect_c).all(), f'char CM drift:\n{cm_c}'
    print('CM verified: theory 77.17% / char 87.64% reproduced exactly')

    from matplotlib.colors import LinearSegmentedColormap
    cmap_cm = LinearSegmentedColormap.from_list('soft_steel', CMAP_SEQ)
    labels = ['Informative', 'Expressive', 'Operative']
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for ax, cm, title in zip(axes, [cm_t, cm_c],
                             ['(a) Theory-driven model', '(b) Character-level TF-IDF baseline']):
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap=cmap_cm, vmin=0, vmax=100, aspect='auto')
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(labels, fontsize=9.5); ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_xlabel('Predicted type', fontsize=10)
        ax.set_ylabel('True type', fontsize=10)
        ax.set_title(title, pad=8)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                c = 'white' if pct[i, j] > 50 else '#1A1A1A'
                ax.text(j, i - 0.10, str(cm[i, j]), ha='center', va='center',
                        fontsize=15, fontweight='bold', color=c)
                ax.text(j, i + 0.26, f'({pct[i, j]:.1f}%)', ha='center', va='center',
                        fontsize=8.5, color=c)
        for sp in ax.spines.values():
            sp.set_color('#B5B5B5'); sp.set_linewidth(0.8)
        ax.tick_params(length=0)
    fig.tight_layout(pad=0.9, w_pad=2.2)
    out = os.path.join(HERE, 'fig3_confusion_en_pub.png')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print('saved:', out)

if __name__ == '__main__':
    sal = fig_loadings()
    print('--- salient loadings (|λ|>=0.40, dominant factor) ---')
    for zh, m in sal.items():
        print(' ', zh, m)
    CORPUS = os.path.join(PKG, 'corpus', 'corpus_relabeled_v27.json')
    if not os.path.exists(CORPUS):
        print('[note] real corpus not found; Figures 2-3 (corpus-based, paper-number asserted) are skipped; '
              'only Figure 1 (from reference JSON) is produced.')
    else:
        fig_spearman()
        fig_confusion()
