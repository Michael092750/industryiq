"""Tests for the figure-VLM ingest pass (industryiq.core.figure_vlm).

Covers the pure, provider-independent parts: placeholder injection, the count-mismatch
fallback, the figure-iteration loop (with a fake doc + fake annotator), and the key
integration guarantee -- a transcribed table survives chunking whole.
"""

from types import SimpleNamespace

from PIL import Image

from industryiq.core.chunking import chunk_markdown, split_sections
from industryiq.core.figure_vlm import annotate_document_figures, inject_figures

PLACEHOLDER = "<!-- image -->"
TABLE = "| metric | value |\n|---|---|\n| variable rate debt | 29.4 percent |"


def _fake_doc(pictures: list[object]) -> object:
    return SimpleNamespace(pictures=pictures)


def _fake_picture(page_no: int, image: Image.Image | None) -> object:
    return SimpleNamespace(
        prov=[SimpleNamespace(page_no=page_no)],
        get_image=lambda _doc: image,
    )


# --- inject_figures ---------------------------------------------------------- #


def test_inject_replaces_each_placeholder_in_order() -> None:
    page = f"Intro.\n\n{PLACEHOLDER}\n\nMiddle.\n\n{PLACEHOLDER}\n\nEnd."
    out = inject_figures(page, ["Figure: a chart.", TABLE])
    assert PLACEHOLDER not in out
    assert "Figure: a chart." in out
    assert TABLE in out
    # order preserved: first figure lands before the second
    assert out.index("Figure: a chart.") < out.index(TABLE)


def test_inject_empty_text_drops_the_placeholder() -> None:
    page = f"before {PLACEHOLDER} after"
    out = inject_figures(page, [""])
    assert PLACEHOLDER not in out
    assert "before" in out and "after" in out


def test_inject_count_mismatch_appends_at_end_without_losing_content() -> None:
    # One placeholder but two figure texts -> fall back to append-at-end.
    page = f"Body text.\n\n{PLACEHOLDER}"
    out = inject_figures(page, [TABLE, "Figure: second."])
    assert PLACEHOLDER not in out
    assert TABLE in out
    assert "Figure: second." in out


def test_inject_no_placeholders_is_a_noop_when_no_figures() -> None:
    assert inject_figures("plain page", []) == "plain page"


# --- annotate_document_figures ---------------------------------------------- #


def test_annotate_returns_texts_per_page_in_order() -> None:
    doc = _fake_doc(
        [
            _fake_picture(1, Image.new("RGB", (400, 300))),
            _fake_picture(1, Image.new("RGB", (400, 300))),
            _fake_picture(2, Image.new("RGB", (400, 300))),
        ]
    )
    calls: list[int] = []

    def annotator(_img: Image.Image) -> str:
        calls.append(1)
        return f"fig-{len(calls)}"

    result = annotate_document_figures(doc, annotator, min_pixels=200)
    assert result == {1: ["fig-1", "fig-2"], 2: ["fig-3"]}


def test_annotate_skips_small_figures_but_keeps_the_slot() -> None:
    doc = _fake_doc(
        [
            _fake_picture(1, Image.new("RGB", (50, 50))),  # below min_pixels -> skipped
            _fake_picture(1, Image.new("RGB", (400, 300))),
        ]
    )
    result = annotate_document_figures(doc, lambda _img: "T", min_pixels=200)
    # slot preserved for the skipped figure so positional injection still aligns
    assert result == {1: ["", "T"]}


def test_annotate_isolates_a_failing_figure() -> None:
    def annotator(_img: Image.Image) -> str:
        raise RuntimeError("VLM down")

    doc = _fake_doc([_fake_picture(1, Image.new("RGB", (400, 300)))])
    # a failed call must not raise -- it yields an empty slot (one figure lost)
    assert annotate_document_figures(doc, annotator, min_pixels=200) == {1: [""]}


def test_annotate_respects_max_figures_cap() -> None:
    doc = _fake_doc([_fake_picture(1, Image.new("RGB", (400, 300))) for _ in range(5)])
    sent = 0

    def annotator(_img: Image.Image) -> str:
        nonlocal sent
        sent += 1
        return "T"

    result = annotate_document_figures(doc, annotator, min_pixels=200, max_figures=2)
    assert sent == 2  # only two figures actually sent to the VLM
    assert result[1] == ["T", "T", "", "", ""]  # remaining slots empty


# --- integration: transcribed table survives chunking whole ----------------- #


def test_injected_table_is_kept_atomic_by_chunk_markdown() -> None:
    page = f"## Debt Overview\nSome prose about leverage.\n\n{PLACEHOLDER}\n\nMore prose."
    injected = inject_figures(page, [TABLE])

    chunks: list[str] = []
    for _section, block in split_sections(injected):
        chunks.extend(chunk_markdown(block, chunk_size=100, overlap=10))

    # The whole table survives verbatim in exactly one chunk -- never split.
    assert any(c == TABLE for c in chunks)
    # And no chunk holds only a fragment (a lone data row without the header).
    assert not any("29.4 percent" in c and "| metric | value |" not in c for c in chunks)
