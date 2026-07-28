# Vietnamese Diacritic Restoration (thêm dấu)

Type Vietnamese without the accent marks, get them back:

```
input:   toi di hoc o Ha Noi hom nay troi dep qua
output:  tôi đi học ở Hà Nội hôm nay trời đẹp quá
```

**Live demo:** run it yourself — see [Try it](#try-it) below. (Hosting on a
Hugging Face Space is one command away via `app/deploy_space.py`; not yet
deployed from this checkout.)

**Headline number:** on a held-out test set from a *different domain* than
training (news articles vs. a Wikipedia-trained model — see
[Evaluation](#evaluation)):

| model | per-syllable accuracy | per-char accuracy | exact sentence match |
|---|---|---|---|
| naive: most-frequent spelling, no context | 52.5% | 89.2% | 0.2% |
| **n-gram + dictionary baseline** | **89.2%** | **97.5%** | **16.4%** |

That ~37-point jump is the whole point of this project: most toneless
Vietnamese syllables have several valid accented spellings (`ma` → `ma,
má, mà, mã, mả, mạ`), and only sentence context can tell them apart. A
plain dictionary lookup can't do that; this can.

(That exact number is from a corpus of ~98K Wikipedia sentences plus
~11.4K colloquial student-feedback sentences merged in — see
[Data sources](#data-sources--licenses). The quickstart below uses
Wikipedia alone for simplicity; numbers will be close but not identical.)

## Try it

```bash
pip install -r requirements.txt
python data/build_dictionary.py --output data/dictionary.json
python data/fetch_corpus.py --articles 800 --output data/corpus.txt
python src/baseline/train.py --corpus data/corpus.txt --output data/ngram_counts.json
python app/app.py   # -> http://127.0.0.1:7860
```

Or without the UI:

```python
import sys; sys.path.insert(0, "src")
from baseline import BaselineRestorer

restorer = BaselineRestorer("data/dictionary.json", "data/ngram_counts.json")
restorer.restore("toi di hoc o Ha Noi hom nay troi dep qua")
# 'tôi đi học ở Hà Nội hôm nay trời đẹp quá'
```

## How it works

This is constrained classification, not free text generation: output
length always equals input length, and each toneless syllable is resolved
to one of a small, finite set of candidate accented spellings — the model
can never hallucinate unrelated text.

1. **`src/strip.py`** — the single source of truth for toneless-stripping.
   NFC-normalizes, then collapses every diacritic *and* modifier letter
   (â/ă/ê/ô/ơ/ư/đ) to its plain ASCII base, matching how people actually
   type toneless Vietnamese (`Nội` → `Noi`, not `Nôi`).
2. **`data/build_dictionary.py`** — builds the toneless → candidate-spellings
   table from an open Vietnamese wordlist (not the training corpus itself),
   so legality of a spelling is a lexicon question, independent of whatever
   corpus happens to be on hand.
3. **`src/baseline/ngram.py` + `model.py`** — a trigram stupid-backoff
   language model scores candidate sequences; exact Viterbi decoding picks
   the best-scoring sequence for the whole sentence, which is where context
   actually resolves the ambiguity.

Every model (this baseline, and any future neural model) sits behind the
same `restore(text: str) -> str` interface, so the demo and eval code don't
care which one is loaded.

## Handling real input

- Already-accented text passes through unchanged (idempotent).
- English words, numbers, punctuation, and emoji pass through untouched
  (best-effort — see the code-switching caveat below).
- Original casing and whitespace are preserved exactly.
- Never crashes on empty input or unknown characters.

## Evaluation

```bash
python data/fetch_corpus.py --dataset tiennv/vietnamese-news --config default \
    --articles 250 --output data/test_corpus.txt
python src/eval.py --test-corpus data/test_corpus.txt \
    --dictionary data/dictionary.json --ngram data/ngram_counts.json
```

Metrics: per-syllable accuracy (primary), per-character accuracy,
whole-sentence exact match — plus a naive "always pick the dictionary's
most-frequent spelling, ignore context" floor, so the number reflects what
context-aware decoding actually buys you, not just dictionary coverage.

The held-out set is pulled from a Vietnamese *news* dataset
(`tiennv/vietnamese-news`), while training is Vietnamese *Wikipedia* —
deliberately different domains, so the number isn't inflated by in-domain
memorization.

## Known limitations

Kept honest on purpose — see **[`examples/hard_examples.md`](examples/hard_examples.md)**
for real, mined failure cases and the reasoning behind them. Short version:

- **Register mismatch.** Training data skews formal (Wikipedia, news), so
  casual words occasionally lose to formal homographs (`quá` → `qua`,
  `bạn` → `bản`). Adding a colloquial corpus fixed some of these and not
  others — it only helps the specific ambiguities it contains examples of,
  which is itself a useful thing to know before reaching for "more data."
- **Code-switching is out of scope, by design.** Vietnamese–English mixing
  isn't handled: an English word that happens to spell a valid Vietnamese
  syllable (`the` → `thể`) gets "corrected" anyway, since there's no
  language-ID step. Fixing this is a deliberately separate future project.

## Repo structure

```
data/            # fetch/preprocess scripts (fetch_corpus.py, build_dictionary.py) —
                  # generated artifacts (*.json, *.txt) are gitignored, not committed
src/
  strip.py       # canonical diacritic-stripping map
  baseline/      # n-gram + dictionary model (ngram.py, model.py, train.py)
  eval.py        # per-syllable / per-char / exact-match metrics
app/             # Gradio demo + deploy_space.py
examples/        # curated hard/ambiguous examples for spot-checks
tests/           # pytest suite (25 tests, all against synthetic fixtures —
                  # no network access or trained artifacts required to run)
```

## Data sources & licenses

- Dictionary: [`duyet/vietnamese-wordlist`](https://github.com/duyet/vietnamese-wordlist)
  (Viet74K, GPL-2.0, credited to Hồ Ngọc Đức's original wordlist).
- Training corpus: [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
  (`20231101.vi`, CC-BY-SA-3.0/GFDL) plus
  [`uitnlp/vietnamese_students_feedback`](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)
  (license unspecified upstream — used only to derive local n-gram
  statistics, not redistributed).
- Held-out eval set: [`tiennv/vietnamese-news`](https://huggingface.co/datasets/tiennv/vietnamese-news)
  (license unspecified upstream — same caveat: local eval only, not
  redistributed).

None of the generated data artifacts (`data/*.json`, `data/*.txt`) are
committed to this repo; regenerate them with the scripts above.

## Non-goals

- Not a translator, spell-checker, or grammar corrector.
- Not solving Vietnamese–English code-switching (see Known limitations).
- The neural token-classification model (ByT5 or similar) is a deliberate
  next step, not part of this baseline — CLAUDE.md's shipping order is
  baseline first, improve second.
