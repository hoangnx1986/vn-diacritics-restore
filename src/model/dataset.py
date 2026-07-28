"""Training-example construction for the neural diacritic restorer.

data/corpus.txt (real accented sentences) + data/dictionary.json (legal
candidate spellings per toneless syllable) -> per-syllable
(char ids, candidate global-class ids, local target index) examples,
batched with padding/masking so a whole sentence's context feeds the
Transformer encoder in one shot.
"""

import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strip import strip_diacritics  # noqa: E402
from tokenize_utils import tokenize_sentence  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocab import CharVocab, ClassVocab  # noqa: E402

MAX_SEQ_LEN = 128  # training-time cap on sentence length (syllables)
MAX_CHAR_LEN = 16  # per-syllable character cap
IGNORE_INDEX = -100  # matches torch's CrossEntropyLoss default sentinel


class Example:
    __slots__ = ("keys", "candidate_ids", "targets")

    def __init__(self, keys: list[str], candidate_ids: list[list[int]], targets: list[int]):
        self.keys = keys  # stripped-syllable strings -- the model's actual input
        self.candidate_ids = candidate_ids  # global class ids legal at each position
        self.targets = targets  # local index into candidate_ids[i], or IGNORE_INDEX


def build_examples(
    sentences: list[list[str]],
    dictionary: dict[str, list[str]],
    class_vocab: ClassVocab,
) -> list[Example]:
    """`sentences` are already-lowercased *accented* token sequences (from
    tokenize_sentence on corpus.txt). Each token's stripped form is what the
    model sees as input; the token itself is the training target."""
    examples = []
    for tokens in sentences:
        tokens = tokens[:MAX_SEQ_LEN]
        keys: list[str] = []
        candidate_ids: list[list[int]] = []
        targets: list[int] = []
        for tok in tokens:
            key = strip_diacritics(tok)
            keys.append(key)
            candidates = dictionary.get(key)
            if not candidates:
                # OOV syllable: no legal candidate set to classify over, so
                # this position contributes no loss -- it's still present
                # as context for its neighbors via `keys`.
                candidate_ids.append([])
                targets.append(IGNORE_INDEX)
                continue
            candidate_ids.append([class_vocab.id_of(c) for c in candidates])
            if tok in candidates:
                targets.append(candidates.index(tok))
            else:
                # Dictionary-coverage gap: true spelling isn't among this
                # key's candidates. Shared limitation with the baseline;
                # exclude from loss rather than supervise an impossible label.
                targets.append(IGNORE_INDEX)
        examples.append(Example(keys, candidate_ids, targets))
    return examples


class DiacriticDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def make_collate_fn(char_vocab: CharVocab):
    def collate_fn(batch: list[Example]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        seq_len = max(len(ex.keys) for ex in batch)
        max_candidates = max(
            (len(c) for ex in batch for c in ex.candidate_ids if c), default=1
        )
        max_candidates = max(max_candidates, 1)

        char_ids = torch.full(
            (batch_size, seq_len, MAX_CHAR_LEN), char_vocab.pad_id, dtype=torch.long
        )
        padding_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)  # True = PAD
        candidate_ids = torch.zeros((batch_size, seq_len, max_candidates), dtype=torch.long)
        candidate_valid = torch.zeros((batch_size, seq_len, max_candidates), dtype=torch.bool)
        targets = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)

        for b, ex in enumerate(batch):
            n = len(ex.keys)
            padding_mask[b, :n] = False
            for i in range(n):
                char_ids[b, i] = torch.tensor(char_vocab.encode(ex.keys[i], MAX_CHAR_LEN))
                cands = ex.candidate_ids[i]
                if cands:
                    k = len(cands)
                    candidate_ids[b, i, :k] = torch.tensor(cands)
                    candidate_valid[b, i, :k] = True
                targets[b, i] = ex.targets[i]

        return {
            "char_ids": char_ids,  # (B, S, MAX_CHAR_LEN)
            "padding_mask": padding_mask,  # (B, S), True = pad
            "candidate_ids": candidate_ids,  # (B, S, C) global class ids
            "candidate_valid": candidate_valid,  # (B, S, C) bool
            "targets": targets,  # (B, S) local index into that position's C candidates
        }

    return collate_fn


def load_sentences(corpus_path: Path) -> list[list[str]]:
    sentences = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            tokens = tokenize_sentence(line)
            if tokens:
                sentences.append(tokens)
    return sentences


def train_val_split(
    sentences: list[list[str]], val_fraction: float = 0.03, seed: int = 0
) -> tuple[list[list[str]], list[list[str]]]:
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]
