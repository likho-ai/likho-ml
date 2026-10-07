"""Error rates: no database, no event bus."""

import pytest

from likho_ml.scoring import Scores, count, edit_distance, normalise_roman, normalise_script, score


def test_edit_distance() -> None:
    assert edit_distance([], []) == 0
    assert edit_distance(["a"], []) == 1
    assert edit_distance([], ["a", "b"]) == 2
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance(["aap", "ka", "order"], ["aap", "order", "kal"]) == 2


def test_script_normalising_ignores_what_is_not_an_error() -> None:
    assert normalise_script("नमस्ते जी। आपका ऑर्डर, कल आएगा॥") == "नमस्ते जी आपका ऑर्डर कल आएगा"
    # A zero-width joiner and NFD input are the same text.
    assert normalise_script("क्‍ष") == normalise_script("क्ष")
    assert normalise_script("Order  NO. 42!") == "order no 42"


def test_roman_normalising_forgives_spelling_variants() -> None:
    for a, b in [
        ("aapka order", "apka order"),
        ("kyaa hua", "kya hua"),
        ("theek hai", "thik hai"),
        ("wala", "vala"),
        ("zaroor", "jaroor"),
        ("acchha", "achha"),
        ("zaroor", "zarur"),
    ]:
        assert normalise_roman(a) == normalise_roman(b), (a, b)
    # Different words stay different.
    assert normalise_roman("hai") != normalise_roman("hain")
    assert normalise_roman("kal") != normalise_roman("kab")


def test_rates_of_one_recording() -> None:
    perfect = score("नमस्ते जी।", "नमस्ते जी", "Namaste ji.", "namaste ji")
    assert perfect.rates() == {"wer_script": 0.0, "cer_script": 0.0, "wer_roman": 0.0, "cer_roman": 0.0}

    one_wrong = score("आपका ऑर्डर कल आएगा", "आपका ऑर्डर कल आया", "aapka order kal aayega", "apka order kal aaya")
    assert one_wrong.script.wer == pytest.approx(1 / 4)
    assert one_wrong.roman.wer == pytest.approx(1 / 4), "aapka/apka is not an error, aayega/aaya is"
    assert 0 < one_wrong.script.cer < one_wrong.script.wer


def test_corpus_rates_weigh_long_recordings_more() -> None:
    total = Scores()
    total.add(score("एक दो तीन चार", "एक दो तीन चार", "", ""))  # 4 words, 0 errors
    total.add(score("पाँच", "छह", "", ""))  # 1 word, 1 error
    assert total.script.wer == pytest.approx(1 / 5), "not the mean of 0 and 1"


def test_an_empty_reference_scores_zero_not_a_division_error() -> None:
    empty = count("", "")
    assert empty.wer == 0.0 and empty.cer == 0.0
