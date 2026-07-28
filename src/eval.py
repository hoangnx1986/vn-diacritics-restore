"""Evaluate a diacritic restorer against held-out accented text.

Reports the metrics CLAUDE.md calls for:
- per-syllable accuracy (primary)
- per-character accuracy
- whole-sentence exact-match rate

Also reports a "most-frequent candidate, no context" baseline built from
the same dictionary (skip the n-gram/Viterbi step entirely) as a sanity
floor — the real model should clearly beat picking the single most common
spelling for every syllable regardless of context.

Usage:
    python src/eval.py --test-corpus data/test_corpus.txt \
        --dictionary data/dictionary.json --ngram data/ngram_counts.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strip import strip_diacritics  # noqa: E402
from baseline.model import BaselineRestorer, _apply_case, _tokenize  # noqa: E402


def naive_restore(text: str, dictionary: dict[str, list[str]]) -> str:
    """No-context baseline: always pick each syllable's single most
    frequent dictionary spelling."""
    tokens = _tokenize(text)
    out = list(tokens)
    for i, tok in enumerate(tokens):
        if not tok.isalpha() or strip_diacritics(tok) != tok:
            continue
        candidates = dictionary.get(tok.lower())
        if candidates:
            out[i] = _apply_case(tok, candidates[0])
    return "".join(out)


def score_sentence(prediction: str, truth: str) -> tuple[int, int, int, int, bool]:
    """Returns (correct_syllables, total_syllables, correct_chars, total_chars, exact_match)."""
    pred_tokens = _tokenize(prediction)
    truth_tokens = _tokenize(truth)

    correct_syl = total_syl = 0
    for p, t in zip(pred_tokens, truth_tokens):
        if t.isalpha():
            total_syl += 1
            correct_syl += int(p.lower() == t.lower())

    # Restoration never changes token/character count, so lengths should
    # always match; guard rather than crash if a sentence somehow doesn't.
    if len(prediction) == len(truth):
        correct_chars = sum(a == b for a, b in zip(prediction, truth))
    else:
        correct_chars = 0
    total_chars = len(truth)

    return correct_syl, total_syl, correct_chars, total_chars, prediction == truth


def evaluate(sentences: list[str], restore_fn: Callable[[str], str]) -> dict:
    sum_correct_syl = sum_total_syl = 0
    sum_correct_chars = sum_total_chars = 0
    exact_matches = 0

    for truth in sentences:
        prediction = restore_fn(strip_diacritics(truth))
        cs, ts, cc, tc, exact = score_sentence(prediction, truth)
        sum_correct_syl += cs
        sum_total_syl += ts
        sum_correct_chars += cc
        sum_total_chars += tc
        exact_matches += int(exact)

    n = len(sentences)
    return {
        "syllable_accuracy": sum_correct_syl / sum_total_syl if sum_total_syl else 0.0,
        "char_accuracy": sum_correct_chars / sum_total_chars if sum_total_chars else 0.0,
        "exact_match_rate": exact_matches / n if n else 0.0,
        "n_sentences": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-corpus", type=Path, default=Path("data/test_corpus.txt"))
    parser.add_argument("--dictionary", type=Path, default=Path("data/dictionary.json"))
    parser.add_argument("--ngram", type=Path, default=Path("data/ngram_counts.json"))
    args = parser.parse_args()

    sentences = [
        line.strip()
        for line in args.test_corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dictionary = json.loads(args.dictionary.read_text(encoding="utf-8"))
    restorer = BaselineRestorer(args.dictionary, args.ngram)

    results = {
        "Naive (most-frequent candidate, no context)": evaluate(
            sentences, lambda t: naive_restore(t, dictionary)
        ),
        "N-gram + Viterbi baseline": evaluate(sentences, restorer.restore),
    }

    print(f"Evaluated on {len(sentences)} held-out sentences\n")
    header = f"{'model':<45}{'syllable acc':>14}{'char acc':>12}{'exact match':>14}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<45}{m['syllable_accuracy']:>13.1%} {m['char_accuracy']:>11.1%} "
            f"{m['exact_match_rate']:>13.1%}"
        )


if __name__ == "__main__":
    main()
