"""Shared corpus-line tokenization for training data construction.

Used by both src/baseline/train.py (n-gram counts) and src/model/dataset.py
(neural training examples), so the alpha-syllable-extraction regex only
lives in one place.
"""

import re
import unicodedata

_SYLLABLE_RE = re.compile(r"[^\W\d_]+")


def tokenize_sentence(line: str) -> list[str]:
    """NFC-normalize and extract lowercase alphabetic syllable runs from a
    corpus line, discarding punctuation, digits, and whitespace."""
    line = unicodedata.normalize("NFC", line)
    return [tok.lower() for tok in _SYLLABLE_RE.findall(line)]
