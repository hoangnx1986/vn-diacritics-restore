"""Vocabularies for the neural diacritic restoration model.

CharVocab: a fixed, tiny ASCII alphabet for encoding toneless syllable
characters -- toneless tokens are always plain `[a-z]+` runs (they've
already been through strip_diacritics()/lower()), so this never needs to
grow.

ClassVocab: the global classification vocabulary -- every distinct
accented spelling that appears anywhere in data/dictionary.json's
candidate lists, in a deterministic (sorted) order. Deterministic so a
freshly-built ClassVocab matches a checkpoint's bundled class list as
long as the same dictionary was used; `from_list` reconstructs one
directly from a checkpoint's bundled list without needing dictionary.json
for that purpose (dictionary.json is still needed separately, for
per-position candidate lookup/masking).
"""

import string


class CharVocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self):
        chars = [self.PAD, self.UNK] + list(string.ascii_lowercase)
        self.id_to_char = chars
        self.char_to_id = {c: i for i, c in enumerate(chars)}

    def __len__(self) -> int:
        return len(self.id_to_char)

    @property
    def pad_id(self) -> int:
        return self.char_to_id[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.char_to_id[self.UNK]

    def encode(self, syllable: str, max_len: int) -> list[int]:
        """Character ids for a lowercase toneless syllable, padded/truncated
        to exactly `max_len`."""
        ids = [self.char_to_id.get(c, self.unk_id) for c in syllable[:max_len]]
        ids += [self.pad_id] * (max_len - len(ids))
        return ids


class ClassVocab:
    def __init__(self, classes: list[str]):
        self.classes = classes
        self.class_to_id = {c: i for i, c in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.classes)

    def id_of(self, spelling: str) -> int:
        return self.class_to_id[spelling]

    def spelling_of(self, class_id: int) -> str:
        return self.classes[class_id]

    def to_list(self) -> list[str]:
        return list(self.classes)

    @classmethod
    def from_dictionary(cls, dictionary: dict[str, list[str]]) -> "ClassVocab":
        classes = sorted({spelling for candidates in dictionary.values() for spelling in candidates})
        return cls(classes)

    @classmethod
    def from_list(cls, classes: list[str]) -> "ClassVocab":
        return cls(classes)
