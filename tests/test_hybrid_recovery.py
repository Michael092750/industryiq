"""Tests for the pypdf hybrid text-recovery net (loaders._recover_dropped_pages).

Docling drops the text inside picture regions (chart footnotes, annotations, boxed
callouts); pypdf reads it off the text layer. Recovery appends the pypdf lines Docling
dropped, deduped against the Docling text and against running headers/footers.
"""

from industryiq.core.loaders import _recover_dropped_pages


def test_recovers_a_footnote_docling_dropped():
    # The real 587835eb case: a chart footnote pypdf has but Docling rasterized away.
    docling = ["## Corporate Bond Issuers\n\n<!-- image -->\n\nSome clean prose Docling kept."]
    pypdf = [
        "Some clean prose Docling kept.\n"
        "For a sample of 518 North American and 157 European high-yield corporate bond "
        "issuers, the average share of variable rate debt is 29.4 percent, at the end of 2022."
    ]
    (recovered,) = _recover_dropped_pages(docling, pypdf)
    assert "29.4 percent" in recovered
    assert "average share of variable rate debt" in recovered


def test_does_not_readd_text_docling_kept_even_if_respaced():
    # Dedup is whitespace/punctuation-insensitive: "17.1" in docling covers pypdf "17 .1".
    docling = ["global compute reached 17.1 million H100-equivalents in 2025"]
    pypdf = ["global compute reached 17 .1 million H100-equivalents in 2025"]
    (recovered,) = _recover_dropped_pages(docling, pypdf)
    assert recovered == ""


def test_drops_running_headers_and_footers():
    banner = "International Monetary Fund Global Financial Stability Report April 2024"
    docling = ["clean body one", "clean body two", "clean body three", "clean body four"]
    # The banner is on every pypdf page (and absent from docling) -> must NOT be recovered.
    pypdf = [f"{banner}\nunique footnote alpha beta gamma delta on page {i}" for i in range(4)]
    recovered = _recover_dropped_pages(docling, pypdf)
    assert all(banner not in block for block in recovered)
    assert "unique footnote alpha beta gamma delta" in recovered[0]


def test_skips_numeric_table_rows_leaves_them_to_the_figure_vlm():
    docling = ["## Table intro\n\n<!-- image -->"]
    pypdf = ["| 29.4 | 44.30 | 36.56 | 40.71 |\n| 1.2 | 3.4 | 5.6 | 7.8 |"]
    (recovered,) = _recover_dropped_pages(docling, pypdf)
    assert recovered == ""  # digit-soup rows have too few letters; not prose to recover


def test_page_alignment_keeps_recovered_text_on_its_own_page():
    docling = ["page one prose", "page two prose"]
    pypdf = [
        "page one prose\nfootnote for page one with several words here",
        "page two prose\nfootnote for page two with several words here",
    ]
    recovered = _recover_dropped_pages(docling, pypdf)
    assert "page one" in recovered[0] and "page two" not in recovered[0]
    assert "page two" in recovered[1] and "page one" not in recovered[1]
