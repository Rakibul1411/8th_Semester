"""Unicode-aware text normalization and tokenization using the standard library."""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol


_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u02bc": "'",  # modifier letter apostrophe
        "\u0060": "'",  # grave accent used as an apostrophe in some sources
        "\u00b4": "'",  # acute accent used as an apostrophe in some sources
    }
)
_PERCENT_SIGNS = {"%", "\u066a"}


class VocabularyEncoder(Protocol):
    """Small protocol needed by summary-boundary preprocessing."""

    sos_id: int
    eos_id: int

    def encode(
        self,
        text: str,
        *,
        add_sos: bool,
        add_eos: bool,
        max_length: int,
    ) -> list[int]: ...


def _is_word_character(character: str) -> bool:
    """Return whether a character belongs inside a Unicode word or number."""
    return unicodedata.category(character)[0] in {"L", "M", "N"}


def _is_symbol_token(character: str) -> bool:
    """Keep currency and percent signs as independent vocabulary tokens."""
    return unicodedata.category(character) == "Sc" or character in _PERCENT_SIGNS


def clean_text(text: str) -> str:
    """Normalize text while retaining Unicode words, apostrophes, and key symbols.

    Currency and percent signs are surrounded with spaces so values such as
    ``"£10"`` and ``"25%"`` become the token pairs ``("£", "10")`` and
    ``("25", "%")``. Other punctuation is treated as a word boundary.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = (
        unicodedata.normalize("NFKC", text)
        .translate(_APOSTROPHE_TRANSLATION)
        .casefold()
    )
    characters: list[str] = []
    for character in normalized:
        if _is_word_character(character) or character == "'":
            characters.append(character)
        elif _is_symbol_token(character):
            characters.extend((" ", character, " "))
        else:
            characters.append(" ")
    return re.sub(r"\s+", " ", "".join(characters)).strip()


def tokenize(text: str) -> list[str]:
    """Return lowercase Unicode word/currency/percent tokens.

    Apostrophes are retained only within a word. For example,
    ``"Müller's £20 offer (10%)"`` becomes
    ``["müller's", "£", "20", "offer", "10", "%"]``.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    tokens: list[str] = []
    for candidate in cleaned.split():
        if len(candidate) == 1 and _is_symbol_token(candidate):
            tokens.append(candidate)
            continue

        # Remove unmatched apostrophes without disturbing contractions and names.
        word_parts = [part for part in candidate.strip("'").split("'") if part]
        if word_parts:
            tokens.append("'".join(word_parts))
    return tokens


def prepare_summary_sequences(
    summary: str,
    vocabulary: VocabularyEncoder,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Create aligned teacher-forcing IDs from one summary.

    The returned lists both have shape ``(T,)``.  The first is
    ``<SOS>, w1, ..., w_(T-1)`` and the second is ``w1, ..., w_(T-1), <EOS>``.
    ``max_length`` counts target positions and the boundary-aware encoding uses
    one extra slot for the leading ``<SOS>``.
    """
    if not isinstance(summary, str):
        raise TypeError("summary must be a string")
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length must be an integer")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    complete = vocabulary.encode(
        summary,
        add_sos=True,
        add_eos=True,
        max_length=max_length + 1,
    )
    if len(complete) < 2:
        raise ValueError("summary must produce at least <SOS> and <EOS>")
    return complete[:-1], complete[1:]
