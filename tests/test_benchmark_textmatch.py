"""Unit tests for the benchmarks' parser-tolerant needle matcher.

The benchmark's gold needles were authored against the old pypdf parse. After the
switch to Docling/OCR, the same fact reflows (spacing around numbers, hyphenation,
unicode punctuation). ``textmatch.normalize`` must bridge that *surface* drift while
still refusing a genuinely reworded sentence, so real content gaps keep failing.
"""

import sys
from pathlib import Path

# The benchmarks dir isn't a package (runners do a bare ``import metrics``); put it
# on the path the same way so ``import textmatch`` resolves here too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import textmatch  # noqa: E402


def test_pypdf_spaced_decimal_matches_clean_docling_text():
    # Bucket A: "17 .1" (pypdf artifact) vs "17.1" (Docling) — the fact is identical.
    assert textmatch.contains_needle(
        "global compute capacity, reaching 17.1 million H100-equivalents in 2025",
        "reaching 17 .1 million H100-equivalents",
    )


def test_soft_hyphenation_across_line_break_matches():
    assert textmatch.contains_needle(
        "total unrealized los-\nses on securities of $56.0 billion",
        "unrealized losses on securities",
    )


def test_unicode_dash_quote_and_nbsp_are_folded():
    # en-dash vs hyphen, curly quotes vs straight, non-breaking space vs space.
    assert textmatch.contains_needle(
        "the “H100–equivalent” rose by 29.4 percent",
        '"H100-equivalent" rose by 29.4 percent',
    )


def test_html_escaped_ampersand_matches_plain_needle():
    # Docling emits escaped markdown: corpus has "R&amp;D", needle has "R&D".
    assert textmatch.contains_needle(
        "accounts for 17.6 percent of all domestic U.S. business R&amp;D performance in 2021",
        "business R&D performance in 2021",
    )


def test_reworded_sentence_still_misses():
    # Bucket B must NOT auto-resolve — it has to surface for re-anchoring, so the
    # matcher can't silently point at the wrong chunk.
    assert not textmatch.contains_needle(
        "primary credit climbed to more than $150 billion within the first week",
        "increased from less than $5 billion to more than $150 billion",
    )


def test_contains_any_matches_second_needle():
    assert textmatch.contains_any(
        "US biopharma output exceeded $802 billion in direct output in 2022",
        ["$999 billion", "$802 billion"],
    )


def test_numeric_anchors_are_longest_first_and_digit_bearing():
    anchors = textmatch.numeric_anchors("reached $285.9 billion, up from 233 in 2024.")
    assert anchors[0] == "$285.9"  # longest, most distinctive number leads
    assert "2024" in anchors and "233" in anchors
    assert all(any(c.isdigit() for c in a) for a in anchors)


def test_ws_tolerant_pattern_finds_spaced_number_in_original_text():
    # Used by the re-anchor assist to locate a clean needle's number in mangled text.
    pattern = textmatch.ws_tolerant_pattern("17.1")
    assert pattern.search("reaching 17 .1 million") is not None
