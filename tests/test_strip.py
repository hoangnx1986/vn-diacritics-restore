import unicodedata

from src.strip import strip_diacritics


def test_claude_md_example():
    assert strip_diacritics("Hà Nội") == "Ha Noi"


def test_full_sentence():
    assert (
        strip_diacritics("tôi đi học ở Hà Nội hôm nay trời đẹp quá")
        == "toi di hoc o Ha Noi hom nay troi dep qua"
    )


def test_modifier_letters_collapse_not_just_tones():
    # â/ă/ê/ô/ơ/ư must collapse to their plain base, not stay distinct —
    # this is the "toneless as people actually type it" behavior CLAUDE.md
    # calls for, not just tone-mark removal.
    assert strip_diacritics("â ă ê ô ơ ư") == "a a e o o u"


def test_dd_stroke():
    assert strip_diacritics("đ") == "d"
    assert strip_diacritics("Đ") == "D"


def test_case_preserved():
    assert strip_diacritics("VIỆT NAM") == "VIET NAM"
    assert strip_diacritics("Việt Nam") == "Viet Nam"


def test_non_vietnamese_characters_pass_through():
    assert strip_diacritics("hello world 123!? 😀") == "hello world 123!? 😀"


def test_empty_string_does_not_crash():
    assert strip_diacritics("") == ""


def test_idempotent_on_already_toneless_text():
    plain = "toi di hoc o Ha Noi"
    assert strip_diacritics(plain) == plain


def test_nfc_and_nfd_input_strip_identically():
    composed = unicodedata.normalize("NFC", "Việt")
    decomposed = unicodedata.normalize("NFD", "Việt")
    assert composed != decomposed  # sanity: these really are different byte sequences
    assert strip_diacritics(composed) == strip_diacritics(decomposed) == "Viet"


def test_every_vietnamese_letter_maps_to_a_plain_ascii_base():
    accented = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    stripped = strip_diacritics(accented)
    assert stripped.isascii()
    assert len(stripped) == len(accented)
