"""Neural diacritic restorer: character-composed Transformer encoder +
masked syllable classification, behind the shared restore(text) -> str
interface. No Viterbi decoding needed here (unlike the n-gram baseline) --
bidirectional attention already integrates full-sentence context into each
position's hidden state before the classification head runs, so decoding
is plain per-position masked argmax.
"""

import json
import os
import sys
from pathlib import Path

# A couple of ops used by torch.nn.TransformerEncoder aren't yet implemented
# on Apple's MPS backend (e.g. aten::_nested_tensor_from_mask_left_aligned)
# and otherwise raise NotImplementedError instead of just falling back to
# CPU for that one op. Must be set before torch is imported anywhere in the
# process to take effect, so a caller doesn't have to know to set this env
# var themselves before running anything that imports NeuralRestorer.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strip import strip_diacritics  # noqa: E402
from baseline.model import _apply_case, _tokenize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from architecture import NeuralDiacriticModel  # noqa: E402
from dataset import MAX_CHAR_LEN  # noqa: E402
from vocab import CharVocab, ClassVocab  # noqa: E402


class NeuralRestorer:
    def __init__(self, dictionary_path: Path, checkpoint_path: Path, device: torch.device | None = None):
        self.dictionary: dict[str, list[str]] = json.loads(
            Path(dictionary_path).read_text(encoding="utf-8")
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.class_vocab = ClassVocab.from_list(checkpoint["class_vocab"])
        self.char_vocab = CharVocab()

        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )
        self.model = NeuralDiacriticModel(**checkpoint["config"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def restore(self, text: str) -> str:
        if not text:
            return text

        tokens = _tokenize(text)
        slot_indices: list[int] = []
        slot_keys: list[str] = []
        slot_candidates: list[list[str]] = []

        for i, tok in enumerate(tokens):
            if not tok.isalpha():
                continue
            if strip_diacritics(tok) != tok:
                continue  # already accented (or non-Vietnamese-diacritic) -- leave as is
            key = tok.lower()
            candidates = self.dictionary.get(key)
            if not candidates:
                continue  # OOV: no legal candidate set, no model call -- identity passthrough
            slot_indices.append(i)
            slot_keys.append(key)
            slot_candidates.append(candidates)

        if not slot_indices:
            return text

        chosen = self._predict(slot_keys, slot_candidates)
        out_tokens = list(tokens)
        for idx, cand in zip(slot_indices, chosen):
            out_tokens[idx] = _apply_case(tokens[idx], cand)
        return "".join(out_tokens)

    def _predict(self, keys: list[str], candidates_per_slot: list[list[str]]) -> list[str]:
        seq_len = len(keys)
        max_candidates = max(len(c) for c in candidates_per_slot)

        char_ids = torch.full((1, seq_len, MAX_CHAR_LEN), self.char_vocab.pad_id, dtype=torch.long)
        padding_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        candidate_ids = torch.zeros((1, seq_len, max_candidates), dtype=torch.long)
        candidate_valid = torch.zeros((1, seq_len, max_candidates), dtype=torch.bool)

        for i, (key, candidates) in enumerate(zip(keys, candidates_per_slot)):
            char_ids[0, i] = torch.tensor(self.char_vocab.encode(key, MAX_CHAR_LEN))
            ids = [self.class_vocab.id_of(c) for c in candidates]
            candidate_ids[0, i, : len(ids)] = torch.tensor(ids)
            candidate_valid[0, i, : len(ids)] = True

        char_ids = char_ids.to(self.device)
        padding_mask = padding_mask.to(self.device)
        candidate_ids = candidate_ids.to(self.device)
        candidate_valid = candidate_valid.to(self.device)

        with torch.no_grad():
            scores = self.model(char_ids, padding_mask, candidate_ids, candidate_valid)  # (1, S, C)

        chosen = []
        for i, candidates in enumerate(candidates_per_slot):
            local_idx = scores[0, i, : len(candidates)].argmax().item()
            chosen.append(candidates[local_idx])
        return chosen
