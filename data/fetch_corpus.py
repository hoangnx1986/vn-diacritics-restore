"""Fetch a Vietnamese sentence corpus from any row-based text dataset on
the Hugging Face Hub, via the public datasets-server API — this serves
pre-parsed rows from HF's CDN rather than live-scraping a source site
(which is how the training pull sidesteps vi.wikipedia.org's aggressive
per-IP rate limiting). Splits article/row text into sentences and writes
one sentence per line.

Defaults to `wikimedia/wikipedia` (20231101.vi) for the *training* corpus.
For a held-out *evaluation* set, point it at a different dataset entirely
so the eval domain doesn't overlap training — e.g. `tiennv/vietnamese-news`
for news-register text vs. Wikipedia's encyclopedic register:

    python data/fetch_corpus.py --articles 600 --output data/corpus.txt
    python data/fetch_corpus.py --dataset tiennv/vietnamese-news --config default \
        --articles 300 --output data/test_corpus.txt

This is a *starter* corpus sized for quickly validating the pipeline
end-to-end, not a production-scale training set — CLAUDE.md's data notes
call for eventually mixing in more sources at larger scale.
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
SPLIT = "train"
PAGE_SIZE = 100  # datasets-server's per-request row cap
MIN_ARTICLE_CHARS = 300  # skip near-empty stub/disambiguation articles

# Rough sentence boundary: '.', '!', or '?' followed by whitespace and a
# capital/quote. Not linguistically perfect, but good enough for n-gram
# training sequences.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỹ\"“])")


def fetch_page(dataset: str, config: str, offset: int, length: int) -> dict:
    url = (
        f"{API_URL}?dataset={dataset}&config={config}&split={SPLIT}"
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
    parser.add_argument("--dataset", default="wikimedia/wikipedia")
    parser.add_argument("--config", default="20231101.vi")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--articles", type=int, default=600, help="number of rows to pull")
    parser.add_argument("--output", type=Path, default=Path("data/corpus.txt"))
    args = parser.parse_args()

    all_sentences: list[str] = []
    kept_articles = 0
    offset = args.start_offset
    end_offset = args.start_offset + args.articles

    while offset < end_offset:
        length = min(PAGE_SIZE, end_offset - offset)
        try:
            data = fetch_page(args.dataset, args.config, offset, length)
        except Exception as exc:  # network hiccups shouldn't kill a long run
            print(f"page fetch failed at offset {offset}: {exc}", file=sys.stderr)
            time.sleep(2)
            continue

        for row in data.get("rows", []):
            text = row["row"][args.text_column]
            if len(text) < MIN_ARTICLE_CHARS:
                continue
            all_sentences.extend(extract_sentences(text))
            kept_articles += 1

        offset += length
        print(f"{offset - args.start_offset}/{args.articles} rows fetched ({kept_articles} usable), {len(all_sentences)} sentences", end="\r")
        time.sleep(0.3)  # be polite to the API

    print()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(all_sentences) + "\n", encoding="utf-8")
    print(f"wrote {len(all_sentences)} sentences from {kept_articles} usable articles -> {args.output}")


if __name__ == "__main__":
    main()
