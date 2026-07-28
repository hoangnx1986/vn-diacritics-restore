"""Fetch a Vietnamese sentence corpus for training the n-gram baseline's
language model.

Pulls cleaned article text from the `wikimedia/wikipedia` dataset
(20231101.vi config) via Hugging Face's public datasets-server API —
this serves pre-parsed article text from HF's CDN, which sidesteps the
aggressive per-IP rate limiting on vi.wikipedia.org's live MediaWiki API.
Splits into sentences and writes one sentence per line.

This is a *starter* corpus sized for quickly validating the baseline
pipeline end-to-end, not a production-scale training set — CLAUDE.md's
data notes call for eventually mixing in news/OSCAR/CC-vi at larger
scale once the pipeline is proven out.

Usage:
    python data/fetch_corpus.py --articles 600 --output data/corpus.txt
"""

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

API_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.vi"
SPLIT = "train"
PAGE_SIZE = 100  # datasets-server's per-request row cap
MIN_ARTICLE_CHARS = 300  # skip near-empty stub/disambiguation articles

# Rough sentence boundary: '.', '!', or '?' followed by whitespace and a
# capital/quote. Not linguistically perfect, but good enough for n-gram
# training sequences.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỹ\"“])")


def fetch_page(offset: int, length: int) -> dict:
    url = (
        f"{API_URL}?dataset={DATASET}&config={CONFIG}&split={SPLIT}"
        f"&offset={offset}&length={length}"
    )
    result = subprocess.run(
        ["curl", "-s", "--max-time", "30", url], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def extract_sentences(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text)
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sent in _SENTENCE_SPLIT_RE.split(line):
            sent = sent.strip()
            if len(sent) >= 20:
                sentences.append(sent)
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=int, default=600, help="number of articles to pull")
    parser.add_argument("--output", type=Path, default=Path("data/corpus.txt"))
    args = parser.parse_args()

    all_sentences: list[str] = []
    kept_articles = 0
    offset = 0

    while offset < args.articles:
        length = min(PAGE_SIZE, args.articles - offset)
        try:
            data = fetch_page(offset, length)
        except Exception as exc:  # network hiccups shouldn't kill a long run
            print(f"page fetch failed at offset {offset}: {exc}", file=sys.stderr)
            time.sleep(2)
            continue

        for row in data.get("rows", []):
            text = row["row"]["text"]
            if len(text) < MIN_ARTICLE_CHARS:
                continue
            all_sentences.extend(extract_sentences(text))
            kept_articles += 1

        offset += length
        print(f"{offset}/{args.articles} articles fetched ({kept_articles} usable), {len(all_sentences)} sentences", end="\r")
        time.sleep(0.3)  # be polite to the API

    print()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(all_sentences) + "\n", encoding="utf-8")
    print(f"wrote {len(all_sentences)} sentences from {kept_articles} usable articles -> {args.output}")


if __name__ == "__main__":
    main()
