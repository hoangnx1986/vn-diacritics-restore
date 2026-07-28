"""Canonical Vietnamese diacritic-stripping map.

Single source of truth for turning accented Vietnamese into its toneless
form. Training data prep, dictionary building, and inference must all
strip through this module so a syllable never strips two different ways
in two different places.
"""

import unicodedata

# Tone marks (sắc/huyền/hỏi/ngã/nặng) *and* the modifier letters
# (ă, â, ê, ô, ơ, ư) are all collapsed to their plain ASCII base, matching
# how people actually type toneless Vietnamese (e.g. "Nội" -> "Noi", not "Nôi").
_MODIFIER_AND_TONE_GROUPS = {
    "a": "áàảãạăắằẳẵặâấầẩẫậ",
    "e": "éèẻẽẹêếềểễệ",
    "i": "íìỉĩị",
    "o": "óòỏõọôốồổỗộơớờởỡợ",
    "u": "úùủũụưứừửữự",
    "y": "ýỳỷỹỵ",
}

STRIP_MAP: dict[str, str] = {"đ": "d", "Đ": "D"}
for _base, _chars in _MODIFIER_AND_TONE_GROUPS.items():
    for _c in _chars:
        STRIP_MAP[_c] = _base
        STRIP_MAP[_c.upper()] = _base.upper()


def strip_diacritics(text: str) -> str:
    """Map every diacritic-bearing character in `text` to its base letter.

    NFC-normalizes first so composed and decomposed Unicode input
    (e.g. "ế" as one codepoint vs. "e" + combining marks) strip identically.
    Casing, whitespace, punctuation, digits, and non-Vietnamese characters
    pass through unchanged.
    """
    text = unicodedata.normalize("NFC", text)
    return "".join(STRIP_MAP.get(ch, ch) for ch in text)
