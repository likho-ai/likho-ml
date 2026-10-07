"""Word and character error rates of a hypothesis against a reference, for both layers.

Both texts are normalised first, so what is not a transcription error does not count as one:
Unicode NFC, zero-width joiners removed, punctuation (the Devanagari danda too) removed, lower
case, whitespace collapsed. The roman layer (Hinglish) also forgives spelling variants that
people write either way - "aapka"/"apka", "kyaa"/"kya", "hain"/"hai" stay distinct, but doubled
vowels and a few common letter swaps are folded - since Hinglish has no one spelling.

Rates are corpus rates: the edits of many recordings summed, divided by their summed reference
length, so a long call weighs more than a short one (Scores.add, Scores.rates).
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)
# Any character that is neither a letter, a combining mark nor a digit (punctuation, symbols, ।, ॥).
_NOT_WORDISH = re.compile(r"[^\wऀ-෿]+", re.UNICODE)
_DANDA = re.compile("[।॥]")

# Hinglish spellings people write either way, folded to one form (applied in order, per word).
_ROMAN_FOLDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"ee+"), "i"),  # "theek" / "thik" (before doubled vowels fold, or it would be "thek")
    (re.compile(r"oo+"), "u"),  # "zaroor" / "zarur"
    (re.compile(r"([aeiou])\1+"), r"\1"),  # aa -> a ("aapka" / "apka")
    (re.compile(r"w"), "v"),  # "wala" / "vala"
    (re.compile(r"z"), "j"),  # "zaroor" / "jaroor"
    (re.compile(r"q"), "k"),
    (re.compile(r"ph"), "f"),
    (re.compile(r"([^aeiou])h\b"), r"\1"),  # a trailing aspirate: "sath" / "sat"
    (re.compile(r"(.)\1+"), r"\1"),  # any doubled letter: "acchha" / "achha"
)


def normalise_script(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    text = _DANDA.sub(" ", text)
    text = _NOT_WORDISH.sub(" ", text.lower())
    return " ".join(text.replace("_", " ").split())


def normalise_roman(text: str) -> str:
    words = normalise_script(text).split()
    folded = []
    for word in words:
        for pattern, replacement in _ROMAN_FOLDS:
            word = pattern.sub(replacement, word)
        folded.append(word)
    return " ".join(folded)


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance (substitutions, insertions, deletions all cost 1), in O(n) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i]
        for j, hyp in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


@dataclass
class Counts:
    """Edits and reference length of one layer, at word and character level."""

    word_edits: int = 0
    words: int = 0
    char_edits: int = 0
    chars: int = 0

    def add(self, other: "Counts") -> None:
        self.word_edits += other.word_edits
        self.words += other.words
        self.char_edits += other.char_edits
        self.chars += other.chars

    @property
    def wer(self) -> float:
        return self.word_edits / self.words if self.words else 0.0

    @property
    def cer(self) -> float:
        return self.char_edits / self.chars if self.chars else 0.0


def count(reference: str, hypothesis: str) -> Counts:
    """Counts of already normalised texts."""
    ref_words, hyp_words = reference.split(), hypothesis.split()
    return Counts(
        word_edits=edit_distance(ref_words, hyp_words),
        words=len(ref_words),
        char_edits=edit_distance(list(reference), list(hypothesis)),
        chars=len(reference),
    )


@dataclass
class Scores:
    """Both layers; add recordings one by one, read corpus rates."""

    script: Counts = field(default_factory=Counts)
    roman: Counts = field(default_factory=Counts)

    def add(self, other: "Scores") -> None:
        self.script.add(other.script)
        self.roman.add(other.roman)

    def rates(self) -> dict[str, float]:
        return {
            "wer_script": self.script.wer,
            "cer_script": self.script.cer,
            "wer_roman": self.roman.wer,
            "cer_roman": self.roman.cer,
        }


def score(reference_script: str, hypothesis_script: str, reference_roman: str, hypothesis_roman: str) -> Scores:
    """One recording: whole texts (all lines joined), both layers."""
    return Scores(
        script=count(normalise_script(reference_script), normalise_script(hypothesis_script)),
        roman=count(normalise_roman(reference_roman), normalise_roman(hypothesis_roman)),
    )
