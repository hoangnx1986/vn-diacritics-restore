"""A small stupid-backoff n-gram language model over accented Vietnamese
syllables.

"Stupid backoff" (Brants et al., 2007) needs no held-out smoothing fit:
back off from the highest order that has seen the context, discounting
by a fixed weight per level dropped. It's not a normalized probability,
but it's a fine relative score for Viterbi decoding, and it degrades
gracefully on the modest corpus sizes this baseline starts with.
"""

import json
from collections import defaultdict
from pathlib import Path

START = "<s>"
BACKOFF_WEIGHT = 0.4
UNK_SCORE = 1e-6


class NgramModel:
    def __init__(self, order: int = 3):
        self.order = order
        # counts[k]: dict mapping a k-token tuple -> how often it occurred
        self.counts: list[dict] = [None] + [defaultdict(int) for _ in range(order)]
        self.total_unigrams = 0

    def train(self, sentences: list[list[str]]) -> None:
        for sent in sentences:
            padded = [START] * (self.order - 1) + sent
            for k in range(1, self.order + 1):
                for i in range(len(padded) - k + 1):
                    self.counts[k][tuple(padded[i : i + k])] += 1
            self.total_unigrams += len(sent)

    def score(self, context: tuple, word: str) -> float:
        """Stupid-backoff score for `word` following `context` (a tuple of
        the preceding tokens, most-recent last). Higher is better; not a
        normalized probability."""
        ctx = context[-(self.order - 1) :]
        return self._backoff(ctx, word)

    def _backoff(self, ctx: tuple, word: str) -> float:
        n = len(ctx) + 1
        ngram = ctx + (word,)
        c_ngram = self.counts[n].get(ngram, 0)
        if n == 1:
            return c_ngram / self.total_unigrams if c_ngram else UNK_SCORE
        c_ctx = self.counts[n - 1].get(ctx, 0)
        if c_ngram > 0 and c_ctx > 0:
            return c_ngram / c_ctx
        return BACKOFF_WEIGHT * self._backoff(ctx[1:], word)

    def save(self, path: Path) -> None:
        payload = {
            "order": self.order,
            "total_unigrams": self.total_unigrams,
            # tuple keys -> "␟"-joined strings so this round-trips through JSON
            "counts": [
                {"\x1f".join(k): v for k, v in level.items()} if level else None
                for level in self.counts
            ],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NgramModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(order=payload["order"])
        model.total_unigrams = payload["total_unigrams"]
        for k, level in enumerate(payload["counts"]):
            if level is None:
                continue
            model.counts[k] = defaultdict(int, {tuple(key.split("\x1f")): v for key, v in level.items()})
        return model
