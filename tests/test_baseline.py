"""Tests for src/baseline/model.py's BaselineRestorer, against a small
synthetic dictionary + n-gram fixture — not the real (gitignored,
multi-tens-of-MB) trained artifacts, so these run instantly and don't
depend on data/build_dictionary.py or data/fetch_corpus.py having run."""

import json

import pytest

from src.baseline.model import BaselineRestorer
from src.baseline.ngram import NgramModel

DICTIONARY = {
    "toi": ["tôi"],
    "di": ["đi"],
    "hoc": ["học"],
    "o": ["ở"],
    "ha": ["hà"],
    "noi": ["nội"],
    "hom": ["hôm"],
    "nay": ["nay"],
    "troi": ["trời"],
    "dep": ["đẹp"],
    "qua": ["quá", "qua"],  # ambiguous: needs context to prefer "quá"
}

TRAINING_SENTENCE = "tôi đi học ở hà nội hôm nay trời đẹp quá".split()


@pytest.fixture
def restorer(tmp_path):
    dict_path = tmp_path / "dictionary.json"
    dict_path.write_text(json.dumps(DICTIONARY, ensure_ascii=False), encoding="utf-8")

    model = NgramModel(order=3)
    model.train([TRAINING_SENTENCE] * 20)
    ngram_path = tmp_path / "ngram.json"
    model.save(ngram_path)

    return BaselineRestorer(dict_path, ngram_path)


def test_claude_md_canonical_example(restorer):
    result = restorer.restore("toi di hoc o ha noi hom nay troi dep qua")
    assert result == "tôi đi học ở hà nội hôm nay trời đẹp quá"


def test_context_resolves_ambiguous_candidate(restorer):
    # "qua" has two candidates ("quá"/"qua") — strong training context
    # should pick "quá", not just the dictionary-order default.
    assert restorer.restore("troi dep qua") == "trời đẹp quá"


def test_idempotent_on_already_accented_text(restorer):
    text = "Tôi đi học."
    assert restorer.restore(text) == text


def test_casing_preserved(restorer):
    assert restorer.restore("Toi di hoc") == "Tôi đi học"
    assert restorer.restore("TOI DI HOC") == "TÔI ĐI HỌC"


def test_digits_and_punctuation_pass_through_losslessly(restorer):
    result = restorer.restore("Toi di hoc, 123!")
    assert result == "Tôi đi học, 123!"


def test_unknown_word_passes_through_unchanged(restorer):
    # "hello" isn't in the dictionary -> falls back to an identity candidate.
    assert restorer.restore("hello toi di hoc") == "hello tôi đi học"


def test_empty_input_does_not_crash(restorer):
    assert restorer.restore("") == ""


def test_whitespace_only_input_does_not_crash(restorer):
    assert restorer.restore("   ") == "   "
