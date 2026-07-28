"""Tests for data/build_dictionary.py's candidate-table logic, using small
synthetic wordlist text — no network access, no dependency on the real
(gitignored, multi-hundred-KB) Viet74K download."""

from data.build_dictionary import build_candidate_counts, clean_candidate_table


def test_single_candidate_syllable():
    table = clean_candidate_table(build_candidate_counts("chuẩn\nchuẩn bị"))
    assert table["chuan"] == ["chuẩn"]


def test_ambiguous_syllable_keeps_all_candidates_sorted_by_frequency():
    text = "\n".join(["ma"] * 5 + ["má"] * 4 + ["mà"] * 3 + ["mã"] * 2 + ["mả"] * 2 + ["mạ"] * 2)
    table = clean_candidate_table(build_candidate_counts(text))
    assert set(table["ma"]) == {"ma", "má", "mà", "mã", "mả", "mạ"}
    assert table["ma"][0] == "ma"  # most frequent spelling first


def test_singleton_noise_dropped_when_a_dominant_candidate_exists():
    # "tơì" is the kind of stray/typo entry the real wordlist has alongside
    # a dominant real spelling — should be filtered out.
    text = "\n".join(["toi"] * 10 + ["tơì"])
    table = clean_candidate_table(build_candidate_counts(text))
    assert "tơì" not in table["toi"]
    assert table["toi"] == ["toi"]


def test_singleton_survives_when_it_is_the_only_candidate():
    table = clean_candidate_table(build_candidate_counts("hiếm"))
    assert table["hiem"] == ["hiếm"]


def test_multi_syllable_phrase_splits_into_individual_syllables():
    counts = build_candidate_counts("Hà Nội")
    assert "ha" in counts and "noi" in counts


def test_hyphenated_entries_split_on_hyphen_too():
    counts = build_candidate_counts("a-ba-toa")
    assert set(counts) == {"a", "ba", "toa"}


def test_no_empty_candidate_lists():
    table = clean_candidate_table(build_candidate_counts("ma\nchuẩn\ntoi"))
    assert all(len(candidates) > 0 for candidates in table.values())
