# CLAUDE.md

Guidance for AI assistants (and humans) working in this repo.

## What this project is

A Vietnamese **diacritic restoration** tool (thêm dấu). It takes toneless / un-accented
Vietnamese and restores the correct marks.

```
input:   toi di hoc o Ha Noi hom nay troi dep qua
output:  tôi đi học ở Hà Nội hôm nay trời đẹp quá
```

This is a real daily pain point (people type without marks and want them back) and a
non-trivial disambiguation problem, which is exactly why it makes a good portfolio piece.

## The bar: this is a *product*, not a notebook

The goal is something a stranger can use without the author present. Concretely:

- A **live demo** (Hugging Face Space or small hosted web UI): text in, restored text out.
- A **README** that shows the problem, an example, and the headline eval number in the
  first screen.
- It **handles messy real input** (mixed case, punctuation, numbers, already-accented text,
  English words left untouched), not just the clean example.
- It reports **honest eval numbers** and a note on failure cases.

If a reviewer can land on it cold and understand it in ~90 seconds, we've done our job.
Do not add scope that doesn't move that bar.

## Core framing (read this before designing anything)

**1. This is constrained classification, not free generation.**
Output length equals input length. Each toneless syllable maps to a *small, finite* set of
valid accented forms. Frame the task as: for each syllable, pick the correct variant from its
candidate set. This massively shrinks the output space vs. seq2seq generation, makes decoding
robust (the model can never hallucinate unrelated text), and makes evaluation clean.

Example candidate set for the toneless syllable `ma`:
`ma, má, mà, mã, mả, mạ` — only context resolves which one.

**2. Training data is effectively free.**
Take any corpus of *correct* accented Vietnamese, strip the diacritics to produce the input,
and keep the original as the target. Unlimited parallel (toneless → accented) pairs from
monolingual text. No manual labeling.

**3. Ambiguity is the whole problem.**
Deterministic dictionary lookup fails precisely where it matters (multiple valid forms per
toneless syllable). The model earns its keep by using context. This is the thing the project
is meant to demonstrate.

## Suggested approach (in shipping order)

Start simple, ship, then improve. Do not skip straight to the fanciest option.

1. **Baseline — n-gram + dictionary.** Map each toneless syllable to its candidate accented
   forms; use an n-gram language model to pick the most likely sequence. Fast, no GPU, gives a
   real accuracy number to beat. Ship this first so there's always a working demo.
2. **Neural — token classification.** Fine-tune a character/syllable-level model where each
   position is classified into its accented variant. `ByT5` (byte-level, no tokenizer headaches
   for Vietnamese) or a small custom Transformer are both reasonable. This should clearly beat
   the baseline; that delta is the story.
3. (Optional, later) Compare against a fine-tuned Vietnamese PLM (e.g. PhoBERT-style) if there's
   appetite, but only after 1 and 2 ship.

## Data notes

- Sources: Vietnamese Wikipedia dumps, news corpora, OSCAR/CC Vietnamese subsets, open song/
  lyric or subtitle corpora. Prefer clean, well-accented text.
- Preprocessing: normalize Unicode to a single canonical form (**NFC**) before stripping —
  Vietnamese has composed vs. decomposed forms (e.g. `ế` as one codepoint vs. `e` + combining
  marks). Inconsistent normalization silently corrupts both training and eval.
- The diacritic stripper must map every accented char to its base (à/á/ả/ã/ạ/â/ă → a, etc.,
  and đ → d). Keep this map in one place; everything depends on it.
- Hold out a test set from a *different* domain than training if possible, so the eval number
  isn't inflated by in-domain memorization.

## Evaluation

- Primary metric: **per-syllable accuracy** (fraction of syllables with the correct marks).
- Secondary: per-character accuracy, and a whole-sentence exact-match rate.
- Always report the **baseline vs. neural** comparison — the improvement is the narrative.
- Keep a small file of hard/ambiguous examples for qualitative spot-checks and for the README.

## Handling real input (don't skip)

- Leave already-correct accented text unchanged (idempotent on correct input).
- Pass through English words, numbers, punctuation, emoji untouched.
- Preserve original casing and whitespace.
- Never crash on empty input or unknown characters.

## Repo structure (target)

```
data/            # scripts to fetch + preprocess corpora (not the raw data itself)
src/
  strip.py       # canonical diacritic-stripping map (single source of truth)
  baseline/      # n-gram + dictionary model
  model/         # neural model: train, infer
  eval.py        # per-syllable / per-char / exact-match metrics
app/             # demo UI (Gradio/Streamlit) -> Hugging Face Space
tests/
README.md
```

## Commands

<!-- Fill these in as the project takes shape. -->

```bash
# setup
# train baseline
# train neural model
# run eval
# launch demo locally
```

## Conventions

- Python, type hints where they aid clarity.
- Keep the strip map and the syllable candidate logic centralized; do not duplicate.
- Every model must be runnable behind the same `restore(text: str) -> str` interface so the
  demo and eval don't care which model is loaded.
- Prefer a small, always-working demo over an unreleased better model.

## Non-goals (for now)

- Not a full translator, spell-checker, or grammar corrector.
- Not solving code-switching (Vietnamese-English mixing) — that's a possible *next* project,
  deliberately out of scope here to keep this one shippable.
