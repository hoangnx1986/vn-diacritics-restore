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
        # Every alpha token's stripped form goes into the model's input
        # sequence -- OOV and already-accented tokens still appear as
        # *context* for their neighbors, exactly as they did during
        # training (src/model/dataset.py never drops a token from the
        # sequence just because it has no loss target). Only a subset of
        # these positions are "predictable" and get a candidate list.
        all_keys: list[str] = []
        alpha_token_indices: list[int] = []
        predictable_positions: list[int] = []  # indices into all_keys
        candidates_per_slot: list[list[str]] = []

        for i, tok in enumerate(tokens):
            if not tok.isalpha():
                continue
            key = strip_diacritics(tok).lower()
            all_keys.append(key)
            alpha_token_indices.append(i)
            seq_pos = len(all_keys) - 1

            if strip_diacritics(tok) != tok:
                continue  # already accented -- context only, leave as is
            candidates = self.dictionary.get(key)
            if not candidates:
                continue  # OOV -- context only, identity passthrough
            predictable_positions.append(seq_pos)
            candidates_per_slot.append(candidates)

        if not predictable_positions:
            return text

        chosen = self._predict(all_keys, predictable_positions, candidates_per_slot)
        out_tokens = list(tokens)
        for seq_pos, cand in zip(predictable_positions, chosen):
            token_idx = alpha_token_indices[seq_pos]
            out_tokens[token_idx] = _apply_case(tokens[token_idx], cand)
        return "".join(out_tokens)

    def _predict(
        self,
        all_keys: list[str],
        predictable_positions: list[int],
        candidates_per_slot: list[list[str]],
    ) -> list[str]:
        """`all_keys` is the full alpha-token sequence (context for every
        position); `predictable_positions` indexes into it for the subset
        that actually needs a prediction, with `candidates_per_slot` in the
        same order."""
        seq_len = len(all_keys)
        max_candidates = max(len(c) for c in candidates_per_slot)

        char_ids = torch.full((1, seq_len, MAX_CHAR_LEN), self.char_vocab.pad_id, dtype=torch.long)
        padding_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        candidate_ids = torch.zeros((1, seq_len, max_candidates), dtype=torch.long)
        candidate_valid = torch.zeros((1, seq_len, max_candidates), dtype=torch.bool)

        for i, key in enumerate(all_keys):
            char_ids[0, i] = torch.tensor(self.char_vocab.encode(key, MAX_CHAR_LEN))

        for slot_i, pos in enumerate(predictable_positions):
            ids = [self.class_vocab.id_of(c) for c in candidates_per_slot[slot_i]]
            candidate_ids[0, pos, : len(ids)] = torch.tensor(ids)
            candidate_valid[0, pos, : len(ids)] = True

        char_ids = char_ids.to(self.device)
        padding_mask = padding_mask.to(self.device)
        candidate_ids = candidate_ids.to(self.device)
        candidate_valid = candidate_valid.to(self.device)

        with torch.no_grad():
            scores = self.model(char_ids, padding_mask, candidate_ids, candidate_valid)  # (1, S, C)

        chosen = []
        for slot_i, pos in enumerate(predictable_positions):
            candidates = candidates_per_slot[slot_i]
            local_idx = scores[0, pos, : len(candidates)].argmax().item()
            chosen.append(candidates[local_idx])
        return chosen
