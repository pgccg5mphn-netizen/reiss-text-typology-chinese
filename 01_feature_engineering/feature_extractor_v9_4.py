#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_extractor_v9_4.py
13-dimensional feature extractor based on Reiss's three-way text typology and
systemic-functional linguistics (revised version).

Revisions in v9_4:
- F8 imperative-sentence detection no longer uses action-driving verbs, avoiding
  false imperatives in informative policy reports
- F14 removed to avoid overlap with F2/F8/F9

Feature system:
    1. Person reference:      F1 first-person density, F2 second-person density,
                              F3 third-person density
    2. Modality type:         F4 factual-cognitive density, F5 affective-attitudinal
                              density, F6 deontic modality density
    3. Sentence strategy:     F7 declarative ratio, F8 imperative ratio,
                              F9 exclamatory ratio
    4. Information organisation: F10 connective density, F11 mean sentence length
    5. Formal variation:      F12 lexical richness (Guiraud's R),
                              F13 sentence-length coefficient of variation
"""
import os
import re
import json
import jieba
import numpy as np
import pandas as pd
from typing import List, Optional


class TextTypeFeatureExtractor:
    """13-dimensional text-type feature extractor (revised version)."""

    # NOTE: feature names are Chinese because they serve as dictionary keys shared
    # with the reference JSON outputs and the analysis scripts; do not rename.
    FEATURE_NAMES = [
        'F1_第一人称密度',
        'F2_第二人称密度',
        'F3_第三人称密度',
        'F4_事实认知密度',
        'F5_情感态度密度',
        'F6_义务情态密度',
        'F7_陈述句比例',
        'F8_祈使句比例',
        'F9_感叹句比例',
        'F10_连接词密度',
        'F11_平均句长',
        'F12_词汇丰富度',
        'F13_句长变异系数',
    ]

    def __init__(self, resource_dir: Optional[str] = None):
        jieba.initialize()

        if resource_dir is None:
            resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wordlists')
        self.resource_dir = resource_dir

        # 1. Personal pronouns (Chinese word lists; the words themselves must stay Chinese)
        self.pronouns = {
            'first': ['我', '我们', '本人', '咱', '俺', '吾', '我校', '我院', '我国', '我司', '我社'],
            'second': ['你', '您', '你们', '大家', '各位', '同志', '同学们', '朋友们', '父老乡亲'],
            'third': ['他', '她', '它', '他们', '她们', '它们', '其', '其他', '其余'],
        }

        # 2. Modality word lists (loaded from the wordlists/ resource files)
        self.factual_cognitive_words = self._load_json('modality_factual_cognitive.json')
        self.deontic_words = self._load_json('modality_deontic.json')
        pos_words = self._load_json('emotion_pos.json')
        neg_words = self._load_json('emotion_neg.json')
        emotion_ext = self._load_json('modality_emotion_extension.json')
        self.emotion_words = sorted(set(pos_words + neg_words + emotion_ext))

        # 3. Connectives
        self.connectives = [
            '因此', '所以', '因为', '由于', '因而', '从而', '于是', '故', '据此', '由此可见',
            '而且', '并且', '此外', '另外', '同时', '以及', '及', '况且', '何况', '甚至',
            '然而', '但是', '但', '不过', '却', '虽然', '尽管', '固然', '即使',
            '总之', '综上所述', '总而言之',
            '首先', '其次', '再次', '最后', '第一', '第二', '第三',
        ]

        # 4. Imperative-sentence resources (strict version)
        self.imperative_markers = [
            '请', '务必', '必须', '应该', '应当', '需要', '让我们', '一起', '共同',
            '携手', '快来', '加入', '参与', '投身', '支持', '助力', '行动起来', '齐心协力',
        ]
        self.exclamatory_particles = [
            '啊', '呀', '哇', '哪', '呢', '吧', '吗', '么', '哦', '哟', '呦', '哩', '咧',
        ]

    def _load_json(self, filename: str) -> List[str]:
        path = os.path.join(self.resource_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing resource file: {path}. Run build_modality_dict_v9.py first.')
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _tokenize(self, text: str) -> List[str]:
        return list(jieba.cut(text.strip()))

    def _split_sentences(self, text: str) -> List[str]:
        # Keep sentence-final punctuation: it is required evidence for the
        # exclamatory/imperative classification.
        parts = re.split(r'(?<=[。！？!?；;])', text)
        sentences = [s.strip() for s in parts if s.strip()]
        return sentences

    @staticmethod
    def _strip_end(s: str) -> str:
        """Strip sentence-final punctuation; used for sentence length and particle checks."""
        return s.rstrip('。！？!?；;')

    def _is_imperative(self, sent: str) -> bool:
        """Imperative sentence (v9_4: same token-level criterion as F6, via jieba tokens).

        Up to v9_3 substring matching (`m in sent`) was used, which misclassified
        words containing the character 应 (e.g. 响应/适应/应急/应用, 'respond/adapt/
        emergency/apply') as imperatives. v9_4 switches to token-level matching,
        consistent with the token criterion of F6 deontic modality density: a
        sentence counts as imperative only if one of its jieba tokens hits the
        imperative-marker list or the deontic word list.
        """
        tokens = set(self._tokenize(sent))
        if tokens & set(self.imperative_markers):
            return True
        if tokens & set(self.deontic_words):
            return True
        return False

    def _is_exclamatory(self, sent: str) -> bool:
        """Exclamatory sentence: ends with '!', or contains an emotional intensifier
        or sentence-final particle, and is not imperative."""
        if self._is_imperative(sent):
            return False
        if sent.endswith(('！', '!')):
            return True
        if any(p in self._strip_end(sent)[-2:] for p in self.exclamatory_particles):
            return True
        emotional_intensifiers = ['多么', '何等', '太', '真', '好', '这么', '那么', '实在', '简直', '真的']
        if any(w in sent for w in emotional_intensifiers):
            return True
        return False

    def _is_declarative(self, sent: str) -> bool:
        return not self._is_imperative(sent) and not self._is_exclamatory(sent)

    def extract(self, text: str) -> np.ndarray:
        text = text.strip()
        tokens = self._tokenize(text)
        token_count = max(len(tokens), 1)
        token_set = set(tokens)

        sentences = self._split_sentences(text)
        sent_count = max(len(sentences), 1)
        sent_lengths = [len(self._strip_end(s)) for s in sentences]

        # F1-F3: person reference
        f1 = sum(1 for t in tokens if t in self.pronouns['first']) / token_count
        f2 = sum(1 for t in tokens if t in self.pronouns['second']) / token_count
        f3 = sum(1 for t in tokens if t in self.pronouns['third']) / token_count

        # F4-F6: modality type
        f4 = sum(1 for t in tokens if t in self.factual_cognitive_words) / token_count
        f5 = sum(1 for t in tokens if t in self.emotion_words) / token_count
        f6 = sum(1 for t in tokens if t in self.deontic_words) / token_count

        # F7-F9: sentence strategy
        imperative = sum(1 for s in sentences if self._is_imperative(s))
        exclamatory = sum(1 for s in sentences if self._is_exclamatory(s))
        declarative = max(sent_count - imperative - exclamatory, 0)
        f7 = declarative / sent_count
        f8 = imperative / sent_count
        f9 = exclamatory / sent_count

        # F10-F11: information organisation
        f10 = sum(1 for t in tokens if t in self.connectives) / token_count
        f11 = np.mean(sent_lengths) if sent_lengths else 0.0

        # F12-F13: formal variation
        # Guiraud's R removes the text-length confound of the plain TTR.
        f12 = len(token_set) / (token_count ** 0.5)
        f13 = np.std(sent_lengths) / np.mean(sent_lengths) if np.mean(sent_lengths) > 0 else 0.0

        return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13], dtype=np.float64)

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self.extract(t) for t in texts])

    def extract_dataframe(self, texts: List[str]) -> pd.DataFrame:
        X = self.extract_batch(texts)
        return pd.DataFrame(X, columns=self.FEATURE_NAMES)


if __name__ == '__main__':
    extractor = TextTypeFeatureExtractor()
    # Chinese demo sentences (one per intended type); the corpus itself is Chinese.
    samples = [
        '2024年我国国内生产总值1349084亿元，按不变价格计算，比上年增长5.0%。',
        '我靠着船舷，望着岸上的山一点一点变小，心里忽然酸酸的。',
        '让我们携手并肩，共同参与到这项光荣的志愿服务中来！',
        '扎实推进乡村全面振兴，加快建设农业强国。',
    ]
    df = extractor.extract_dataframe(samples)
    print(df.to_string(index=False))
