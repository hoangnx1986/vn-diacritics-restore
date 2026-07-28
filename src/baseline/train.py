"""Fit the baseline's n-gram language model on a plain-text Vietnamese
corpus (one sentence per line — see data/fetch_corpus.py) and save the
counts for src/baseline/model.py to load at inference time.

Usage:
    python src/baseline/train.py --corpus data/corpus.txt --output data/ngram_counts.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngram import NgramModel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenize_utils import tokenize_sentence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/ngram_counts.json"))
    parser.add_argument("--order", type=int, default=3)
    args = parser.parse_args()

    sentences = []
    with args.corpus.open(encoding="utf-8") as f:
        for line in f:
            tokens = tokenize_sentence(line)
            if tokens:
                sentences.append(tokens)

    model = NgramModel(order=args.order)
    model.train(sentences)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(
        f"trained order-{args.order} model on {len(sentences)} sentences "
        f"({model.total_unigrams} tokens) -> {args.output}"
    )


if __name__ == "__main__":
    main()
