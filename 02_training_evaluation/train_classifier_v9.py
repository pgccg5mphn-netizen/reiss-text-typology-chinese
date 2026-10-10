#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_classifier_v9.py
Trains the Reiss text-type three-way classifier with the 13 theory-driven
features + logistic regression, evaluated by 5-fold cross-validation.

Outputs:
- per-fold accuracy
- macro-averaged F1
- confusion matrix
- classification report
- feature-weight table
- model files: ../models/lr_v9_13features.pkl
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '01_feature_engineering')))
from feature_extractor_v9_4 import TextTypeFeatureExtractor


# ----------------------------------------------------------------------
# Corpus loading
# ----------------------------------------------------------------------
def load_corpus(path: str):
    """Read a Label Studio JSON export; return (texts, labels)."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts, labels = [], []
    for item in data:
        text = item.get('data', {}).get('text', '').strip()
        if not text:
            continue
        annos = item.get('annotations', [])
        if not annos:
            continue
        results = annos[0].get('result', [])
        if not results:
            continue
        choices = results[0].get('value', {}).get('choices', [])
        if not choices:
            continue
        label = choices[0]
        texts.append(text)
        labels.append(label)
    return texts, labels


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main():
    corpus_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "corpus", "corpus_relabeled_v27.json"))
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f'Corpus not found: {corpus_path}')

    texts, labels = load_corpus(corpus_path)
    print(f'Corpus size: {len(texts)}')
    print(f'Label distribution: {dict(Counter(labels))}')

    # Feature extraction
    extractor = TextTypeFeatureExtractor()
    X = extractor.extract_batch(texts)
    feature_names = extractor.FEATURE_NAMES
    print(f'Feature matrix: {X.shape}')

    # Label encoding, in a readable order
    label_order = ['informative', 'expressive', 'infective']
    y = np.array([label_order.index(l) for l in labels])

    # 5-fold stratified cross-validation (scaler fitted within each fold to avoid leakage)
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_f1_macro = []
    all_y_true = []
    all_y_pred = []

    print('\n=== 5-fold cross-validation ===')
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

        fold_accuracies.append(acc)
        fold_f1_macro.append(f1)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f'Fold {fold}: Accuracy={acc:.4f}, Macro-F1={f1:.4f}')

    # Summary
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_macro)

    print(f'\n=== Summary ===')
    print(f'5-fold accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
    print(f'Macro-averaged F1: {mean_f1:.4f}')

    # Classification report
    print('\n=== Classification report ===')
    print(classification_report(all_y_true, all_y_pred, target_names=label_order, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2])
    print('\n=== Confusion matrix ===')
    print(pd.DataFrame(cm, index=[f'True_{l}' for l in label_order], columns=[f'Pred_{l}' for l in label_order]))

    # Train on the full data and save the final model (scaler fitted on the full
    # data, for deployment/inference)
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_clf = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42
    )
    final_clf.fit(X_scaled, y)

    models_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'lr_v9_13features.pkl')
    scaler_path = os.path.join(models_dir, 'scaler_v9_13features.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(final_clf, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(final_scaler, f)
    print(f'\nModel saved: {model_path}')
    print(f'Scaler saved: {scaler_path}')

    # Feature weights
    coef_df = pd.DataFrame(final_clf.coef_, columns=feature_names, index=label_order)
    coef_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v9_feature_weights.csv')
    coef_df.to_csv(coef_path, encoding='utf-8-sig')
    print(f'Feature weights saved: {coef_path}')
    print('\n=== Feature weights (sorted by mean absolute value) ===')
    mean_abs_coef = coef_df.abs().mean(axis=0).sort_values(ascending=False)
    print(mean_abs_coef)

    # Save the result summary
    summary = {
        'corpus': corpus_path,
        'n_samples': len(texts),
        'label_distribution': dict(Counter(labels)),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'mean_accuracy': round(mean_acc, 4),
        'std_accuracy': round(std_acc, 4),
        'macro_f1': round(mean_f1, 4),
        'fold_accuracies': [round(a, 4) for a in fold_accuracies],
        'confusion_matrix': cm.tolist(),
    }
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v9_classification_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\nResult summary saved: {summary_path}')


if __name__ == '__main__':
    main()
