"""Build the toneless -> [accented candidates] dictionary used by the
baseline model's candidate generation step.

Source: duyet/vietnamese-wordlist's Viet74K.txt (GPL-2.0, credited to Ho
Ngoc Duc's original Vietnamese wordlist). Each line is a word or phrase;
we split on whitespace/hyphens to get individual syllables, strip each to
its toneless form, and record every accented spelling observed for that
form. This gives us the closed set of legal candidates per syllable
without depending on the (separate) training corpus for coverage.

Usage:
    python data/build_dictionary.py --output data/dictionary.json
    python data/build_dictionary.py --input path/to/Viet74K.txt --output data/dictionary.json
"""

import argparse
import json
import re
import sys
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from strip import strip_diacritics  # noqa: E402

WORDLIST_URL = "https://vietnamese-wordlist.duyet.net/Viet74K.txt"

# A spelling that occurs only once across the whole wordlist is more likely
# a typo/rare-entry artifact than a real candidate worth searching over.
# Never applied if it would empty out a key entirely (see clean_candidate_table).
MIN_CANDIDATE_COUNT = 2


def fetch_wordlist(url: str = WORDLIST_URL) -> str:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def build_candidate_counts(wordlist_text: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for line in wordlist_text.splitlines():
        line = unicodedata.normalize("NFC", line.strip())
        if not line:
            continue
        for syllable in re.split(r"[\s\-]+", line):
            if not syllable.isalpha():
                continue
            syllable = syllable.lower()
            key = strip_diacritics(syllable)
            counts[key][syllable] += 1
    return counts


def clean_candidate_table(counts: dict[str, dict[str, int]]) -> dict[str, list[str]]:
    table = {}
    for key, forms in counts.items():
        kept = {form: n for form, n in forms.items() if n >= MIN_CANDIDATE_COUNT}
        if not kept:
            best = max(forms, key=forms.get)
            kept = {best: forms[best]}
        table[key] = sorted(kept, key=kept.get, reverse=True)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, help="local Viet74K.txt path; fetches from the web if omitted"
    )
    parser.add_argument("--output", type=Path, default=Path("data/dictionary.json"))
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8") if args.input else fetch_wordlist()
    table = clean_candidate_table(build_candidate_counts(text))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(table, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    n_ambiguous = sum(1 for forms in table.values() if len(forms) > 1)
    print(f"{len(table)} toneless syllables ({n_ambiguous} with >1 candidate) -> {args.output}")


if __name__ == "__main__":
    main()
