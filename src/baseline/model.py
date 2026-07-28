"""The baseline diacritic restorer: dictionary candidate lookup + n-gram
Viterbi decoding, behind the shared `restore(text: str) -> str` interface.
"""

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strip import strip_diacritics  # noqa: E402

from .ngram import NgramModel

# Matches every character in the input exactly once (letters / digits /
# underscores / whitespace as runs, everything else one char at a time),
# so "".join(tokens) always reconstructs the original string exactly.
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+|_+|\s+|[^\w\s]")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _apply_case(original: str, candidate: str) -> str:
    if original.isupper() and len(original) > 1:
        return candidate.upper()
    if original[:1].isupper():
        return candidate[:1].upper() + candidate[1:]
    return candidate


def viterbi_restore(candidates_per_slot: list[list[str]], model: NgramModel) -> list[str]:
    """Find the highest-scoring sequence of candidate choices under `model`.

    State = the last (order - 1) chosen tokens. Exact Viterbi, no beam
    pruning — candidate sets here are small enough (tens, not thousands)
    that this stays cheap even for long sentences.
    """
    order = model.order
    start_state = ("<s>",) * (order - 1)
    beams = {start_state: (0.0, [])}

    for candidates in candidates_per_slot:
        new_beams: dict[tuple, tuple[float, list[str]]] = {}
        for state, (log_score, seq) in beams.items():
            for cand in candidates:
                s = log_score + math.log(model.score(state, cand))
                new_state = (state + (cand,))[-(order - 1) :] if order > 1 else ()
                if new_state not in new_beams or s > new_beams[new_state][0]:
                    new_beams[new_state] = (s, seq + [cand])
        beams = new_beams

    best_state = max(beams, key=lambda st: beams[st][0])
    return beams[best_state][1]


class BaselineRestorer:
    def __init__(self, dictionary_path: Path, ngram_path: Path):
        self.dictionary: dict[str, list[str]] = json.loads(
            Path(dictionary_path).read_text(encoding="utf-8")
        )
        self.model = NgramModel.load(ngram_path)

    def restore(self, text: str) -> str:
        if not text:
            return text

        tokens = _tokenize(text)
        candidates_per_slot: list[list[str]] = []
        slot_indices: list[int] = []

        for i, tok in enumerate(tokens):
            if not tok.isalpha():
                continue
            if strip_diacritics(tok) != tok:
                continue  # already accented (or non-Vietnamese-diacritic) — leave as is
            key = tok.lower()
            candidates_per_slot.append(self.dictionary.get(key, [tok]))
            slot_indices.append(i)

        if not slot_indices:
            return text

        chosen = viterbi_restore(candidates_per_slot, self.model)
        out_tokens = list(tokens)
        for idx, cand in zip(slot_indices, chosen):
            out_tokens[idx] = _apply_case(tokens[idx], cand)
        return "".join(out_tokens)
