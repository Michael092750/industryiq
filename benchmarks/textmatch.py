"""Parser-tolerant gold-needle matching, shared by both benchmarks.

Gold needles are verbatim phrases copied out of the corpus, so they are coupled
to whatever the *PDF parser* produced. Switching parser (pypdf -> Docling/OCR)
reflows the same fact: different whitespace around a number ("17 .1" vs "17.1"),
soft-hyphenation across a line break ("emis-\\nsions"), unicode punctuation
(curly quotes, en/em dashes, non-breaking spaces). A raw ``substring in text``
then breaks even though the underlying fact is unchanged -- "benchmark drift".

:func:`normalize` folds both sides to a whitespace- and punctuation-insensitive
form so *pure surface drift* no longer breaks a needle. It deliberately stops
there: a genuinely **reworded** sentence still changes the normalized string and
still fails to match, so it surfaces for re-anchoring rather than silently
resolving to the wrong chunk. That boundary is the point -- the benchmark must
keep catching facts that were corrupted or dropped, not paper over them.

The ``*_anchor*`` helpers power the runner's re-anchor assist: given a needle
that no longer resolves, they locate its numeric core in the current corpus so a
fresh verbatim phrase can be copied out (the manual hunt, automated).
"""

import html
import re
import unicodedata

# Curly quotes/apostrophes -> ASCII, and every unicode dash/minus -> "-", so OCR
# punctuation variants collapse to one form before matching.
_TRANSLATE = str.maketrans(
    {
        **{ord(c): "'" for c in "‘’‚‛′"},
        **{ord(c): '"' for c in "“”„‟″"},
        **{ord(c): "-" for c in "‐‑‒–—―−"},
    }
)
# Join a word hyphenated across a line break: "emis-\nsions" -> "emissions".
_SOFT_HYPHEN_BREAK = re.compile(r"-\s*\n\s*")
_WHITESPACE = re.compile(r"\s+")
# A numeric anchor: an optional $, a digit, then digits/decimals/commas, optional %.
_NUMERIC = re.compile(r"\$?\d[\d.,]*%?")


def normalize(text: str) -> str:
    """Fold ``text`` to a whitespace/punctuation-insensitive form for matching.

    NFKC (ligatures, full-width, non-breaking space -> space) -> unescape HTML
    entities (Docling emits escaped markdown, so "R&amp;D" -> "R&D") -> unify
    quotes/dashes -> casefold -> join line-break hyphenation -> drop *all*
    whitespace. After this, "reaching 17 .1 million" and "reaching 17.1 million"
    are identical, and "business R&amp;D" matches a "business R&D" needle, but a
    reworded sentence is not.
    """
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = text.translate(_TRANSLATE)
    text = text.casefold()
    text = _SOFT_HYPHEN_BREAK.sub("", text)
    text = _WHITESPACE.sub("", text)
    return text


def contains_needle(haystack: str, needle: str) -> bool:
    """True if ``needle`` occurs in ``haystack`` under :func:`normalize`."""
    return normalize(needle) in normalize(haystack)


def contains_any(haystack: str, needles: list[str]) -> bool:
    """True if any of ``needles`` occurs in ``haystack`` (normalized once)."""
    hay = normalize(haystack)
    return any(normalize(n) in hay for n in needles)


def numeric_anchors(text: str) -> list[str]:
    """Distinctive numeric tokens in ``text``, longest first.

    Numbers ("$285.9", "17.6", "150") are the parser-invariant core of these
    stat needles, so they locate the fact even when the sentence around them was
    reworded. Longer tokens are the most specific, so they lead.
    """
    toks = {m.group(0).rstrip(".,") for m in _NUMERIC.finditer(text)}
    toks = {t for t in toks if any(c.isdigit() for c in t)}
    return sorted(toks, key=len, reverse=True)


def ws_tolerant_pattern(token: str) -> re.Pattern[str]:
    """A regex matching ``token`` with any whitespace allowed between characters,
    so "17.1" also finds the pypdf-mangled "17 .1" in the *original* text."""
    chars = [c for c in token if not c.isspace()]
    return re.compile(r"\s*".join(re.escape(c) for c in chars), re.IGNORECASE)
