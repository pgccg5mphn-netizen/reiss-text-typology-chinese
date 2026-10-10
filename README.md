# Supplementary Code — Quantitative Test of Reiss's Text Typology in Chinese

This package contains the analysis code for the manuscript "From functional
definitions to linguistic features: a quantitative test of Reiss's text
typology in Chinese." It implements the full pipeline: 13 theory-driven
feature extraction, logistic-regression classification with stratified
five-fold cross-validation, TF-IDF baselines, source/genre-grouped (LOGO)
validation, ablation, and within-type factor analysis.

## Contents

```
code_v27/
├── 01_feature_engineering/          Feature extractor (13 dimensions, v9_4)
├── 02_training_evaluation/          Main classifier training and cascaded recomputation
├── 03_ablation/          Per-feature and per-group ablation (zeroing + retraining)
├── 04_factor_analysis/          Within-type factor analysis (12 dimensions, F7 removed)
├── 05_data_comparison/          Char/word-level TF-IDF baselines, LOGO validation, figures
├── wordlists/                 Eight lexical resources (modality, emotion lists)
├── corpus/               demo_corpus.json (synthetic sample, see Data note)
├── models/               Trained logistic regression and scaler (pickle)
├── outputs_reference/    Result files corresponding to the numbers in the paper
└── README.md
```

## Requirements

Python 3.9+ with: jieba, numpy, pandas, scipy, scikit-learn,
factor-analyzer, matplotlib (figures only). All scripts were run with
scikit-learn's default `LogisticRegression(C=1.0, class_weight='balanced',
max_iter=1000, solver='lbfgs')` unless stated otherwise.

## Data note

The annotated corpus (1,165 passages of contemporary Chinese) cannot be
redistributed: 60.8% of the passages derive from the ToRCH2014 corpus
(Beijing Foreign Studies University), and the remainder were collected from
public web sources whose copyright rests with third parties; a small number
of passages involve personal information.

Three provisions make the study verifiable without the raw text:

1. **Feature matrix.** The file `feature_matrix.csv` (supplied with this
   package) reports, for every passage, its identifier, type label, source
   category, and the values of the 13 features. All factor-analysis results
   in the paper can be reproduced from this file alone.
2. **Passage identifiers.** Identifiers of ToRCH-sourced passages follow the
   pattern `ToRCH2014_<file code>_<paragraph>`, e.g. `ToRCH2014_A10_06`.
   ToRCH2014 is freely downloadable from BFSU
   (https://corpus.bfsu.edu.cn), so these passages can be reconstructed from
   the official release using the identifiers.
3. **Synthetic demo corpus.** `corpus/demo_corpus.json` contains 18 passages
   written for this package (six per type). It is not research data. To
   smoke-test the pipeline, copy it over `corpus/corpus_relabeled_v27.json`
   and run any script; the full-data runs require the real corpus.

## Running the pipeline

Scripts are standalone; each resolves paths relative to the package root.

| Step | Script | Produces |
|---|---|---|
| 1 | `02_training_evaluation/train_classifier_v9.py` | 5-fold CV accuracy, confusion matrix, feature weights, model files |
| 2 | `02_training_evaluation/train_main_results.py` | Cascaded recomputation: accuracy, ablation, factor analysis (`main_results.json`) |
| 3 | `03_ablation/verify_ablation_v27.py` | Ablation verification |
| 4 | `04_factor_analysis/save_fa_v27_json.py` | Factor analysis (KMO, Bartlett, loadings) |
| 5 | `05_data_comparison/data_route_baselines_v27.py` | Char/word-level TF-IDF baselines, LOGO results |
| 6 | `05_data_comparison/feature_group_split_v27.py` | Feature-group split analysis |
| 7 | `05_data_comparison/genre_internal_tests_v27.py` | Within-genre robustness tests |
| 8 | `figures/plot_spearman_rank.py`, `figures/plot_confusion_compare.py` | Figures 1 and 2 (`FIG_LANG=en` gives English labels) |

`outputs_reference/` contains the result files generated from the full
corpus; the numbers reported in the paper can be checked against them. The
factor-analysis step requires at least several hundred passages per type and
will not converge on the synthetic sample; run it on `feature_matrix.csv`
instead.

### Paper-number verification scripts

Each `05_data_comparison/verify_*.py` script checks one specific number reported
in the manuscript and writes a `*_out.json` file next to it. They run
end-to-end on the synthetic corpus at demo scale; on the full corpus
(reconstruction path described in Data availability) they reproduce the
reported values.

| Script | Checks (paper section) |
|---|---|
| `05_data_comparison/verify_multiseed5.py` | Five-seed range 76.1–77.2 per cent (§4.1) |
| `05_data_comparison/verify_typical_purity.py` | Typical-passage shares 48.1/65.1/55.5; 650 passages (55.8%); purity 0.79/0.80 (§4.1) |
| `05_data_comparison/verify_spearman_39pairs.py` | 32 of 39 feature–probability pairs at p < .001 (§4.4) |
| `05_data_comparison/verify_random13_control.py` | 13 random features at chance level (≈33 per cent), vs majority baseline 40.69 per cent (§4.1) |
## Manuscript figures

`05_data_comparison/plot_figures_en_pub.py` regenerates the three submission figures:
Figure 1 (factor-loadings plot) is drawn from the reference JSON in
`outputs_reference/factor_analysis_12dim_v27.json` and reproduces every loading
value reported in the manuscript; Figures 2-3 (Spearman panels, confusion
matrices) re-run the full pipeline and assert agreement with the reported
numbers. On the demo corpus only Figure 1 is produced.


## Licence and citation

The code is released under the MIT licence (see `LICENSE`). A citable,
version-persistent DOI is available on Zenodo: https://doi.org/10.5281/zenodo.23273049.
Please cite both the manuscript and the code repository if you reuse the code.


## The 13 features

| Feature | Definition |
|---|---|
| F1–F3 | First-, second-, third-person pronoun density (per token) |
| F4–F6 | Factual-cognitive, emotive, and deontic modality density (lexical lists, per token) |
| F7–F9 | Declarative, imperative, exclamatory sentence proportions (mutually exclusive) |
| F10 | Connective density |
| F11 | Mean sentence length (tokens) |
| F12 | Guiraud's index (lexical richness) |
| F13 | Coefficient of variation of sentence length |
