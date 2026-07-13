"""Figure understanding at ingest: transcribe charts/figures with a vision model.

Docling's layout model detects figures but drops their *content*: a chart's numbers
live in the picture region, which the Markdown export emits only as an
``<!-- image -->`` placeholder. Docling's own chart-extraction / picture-description
stages recover that content, but they run heavy local vision models *inside* the
fragile ``StandardPdfPipeline`` -- which OOM/crash on this environment, silently
falling the whole document back to pypdf (see ``docs/figure-ingestion.md``).

This module runs the recovery as a **separate pass over the finished
``DoclingDocument``**, not inside the pipeline. For each detected picture we call a
provider-agnostic :data:`Annotator` (image -> Markdown) and splice the result into the
page Markdown in place of the picture's placeholder. Charts come back as GFM tables
(kept whole by :func:`industryiq.core.chunking.chunk_markdown`); other figures as a
one-line description. Because it is outside the pipeline, a VLM failure loses one
figure, not the whole document's parse -- and it needs no local models, so it sidesteps
the torch.compile / transformers breakage the local stages hit.

The module is provider-agnostic: :func:`annotate_document_figures` takes any callable
``image -> str``, so it is testable offline with a fake and swappable per provider.
"""

import base64
import io
import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anthropic.types import MessageParam
    from PIL.Image import Image

logger = logging.getLogger(__name__)

# Docling's Markdown export emits this for each picture (its default image placeholder).
_IMAGE_PLACEHOLDER = "<!-- image -->"

# Output cap for one figure transcription (a dense chart table stays well under this).
FIGURE_MAX_TOKENS = 2048

# Prompt kept deliberately strict so the output is either a clean GFM table or a single
# line -- both land cleanly in the page Markdown and chunk predictably.
FIGURE_PROMPT = (
    "You are transcribing a single figure extracted from an industry-analysis report so "
    "its content becomes searchable as text.\n"
    "- If it is a data chart (bar, line, pie, scatter, area, ...) or a table rendered as an "
    "image: output ONLY a GitHub-flavored Markdown table of its data -- every series, "
    "category, axis label, and numeric value you can read, with units (%, $, bn, ...). If "
    "the figure has a title, put it on one line above the table.\n"
    "- If it is a diagram, photo, logo, map, or decorative image with no readable data: "
    "output ONE short sentence describing it, prefixed with 'Figure: '.\n"
    "- Output only the table or the one-line description. No preamble, no commentary, no "
    "code fences."
)

#: A figure annotator turns one figure image into Markdown (a GFM table or a one-liner).
Annotator = Callable[["Image"], str]


def encode_png_b64(image: "Image") -> str:
    """Encode a figure image as a base64 (ascii) PNG -- the wire form for a vision call."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def figure_user_content(png_b64: str) -> list[dict[str, Any]]:
    """The Messages ``content`` blocks for one figure: the image, then the prompt.

    Shared by the synchronous annotator and the Batch-API path so both send an
    identical request -- the only difference between them is the transport.
    """
    return [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": png_b64},
        },
        {"type": "text", "text": FIGURE_PROMPT},
    ]


class SupportsFigureVlmSettings(Protocol):  # pragma: no cover - typing only
    # Read-only members (properties) so a frozen Settings dataclass -- whose attributes
    # are read-only -- structurally satisfies the protocol; plain mutable attributes on
    # other implementers match a read-only member too.
    @property
    def figure_vlm(self) -> str: ...
    @property
    def figure_vlm_model(self) -> str: ...
    @property
    def figure_vlm_min_pixels(self) -> int: ...
    @property
    def figure_vlm_max_figures(self) -> int: ...
    @property
    def anthropic_api_key(self) -> str | None: ...


class AnthropicFigureAnnotator:
    """Transcribe a figure image via the Anthropic Messages API (vision)."""

    def __init__(self, *, model: str, api_key: str, max_tokens: int = FIGURE_MAX_TOKENS) -> None:
        import anthropic  # local import: keep the heavy SDK off the module import path

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def __call__(self, image: "Image") -> str:
        # figure_user_content returns plain dicts (shared with the Batch path, which
        # serializes them as JSON); cast to the SDK's param type for the live call.
        messages = cast(
            "list[MessageParam]",
            [{"role": "user", "content": figure_user_content(encode_png_b64(image))}],
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()


def build_annotator(settings: SupportsFigureVlmSettings) -> Annotator:
    """Return the configured figure annotator, or raise if it can't be built.

    Raising here (at ingest start) rather than per-figure means a misconfiguration fails
    fast and loudly instead of silently producing a corpus with no figure data.
    """
    if settings.figure_vlm == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("FIGURE_VLM=anthropic requires ANTHROPIC_API_KEY to be set")
        return AnthropicFigureAnnotator(
            model=settings.figure_vlm_model, api_key=settings.anthropic_api_key
        )
    raise ValueError(f"Unknown FIGURE_VLM={settings.figure_vlm!r} (expected 'off' or 'anthropic')")


def _picture_image(picture: Any, doc: Any) -> "Image | None":
    """The cropped figure image for a docling ``PictureItem``, or ``None``.

    Needs ``generate_picture_images=True`` at conversion time; best-effort so a picture
    with no recoverable image is simply skipped rather than failing the pass.
    """
    try:
        # ``picture`` is an untyped docling item, so ``get_image`` returns Any; pin it to
        # the declared return type rather than leaking Any out of the function.
        return cast("Image | None", picture.get_image(doc))
    except Exception:  # noqa: BLE001 -- a missing/unreadable image just skips this figure
        return None


def iter_figure_slots(
    doc: Any,
    *,
    min_pixels: int = 200,
    max_figures: int = 0,
) -> Iterator[tuple[int, "Image | None"]]:
    """Yield ``(page_no, image | None)`` for each picture in ``doc``, in reading order.

    ``image`` is the crop to transcribe, or ``None`` when this slot should be left empty
    (too small, unreadable, or past the ``max_figures`` cap). Every picture yields exactly
    one tuple, so callers can align the results positionally with the page's placeholders.
    This is the single source of truth for figure selection, shared by the synchronous
    annotator and the Batch-API collector -- a figure skipped here is skipped identically
    by both. A yielded image counts toward ``max_figures`` (the cap bounds attempts, so a
    later failure doesn't hand its budget to another figure).
    """
    sent = 0
    for picture in getattr(doc, "pictures", []):
        prov = getattr(picture, "prov", None)
        if not prov:
            continue
        page_no = prov[0].page_no
        image = _picture_image(picture, doc)
        capped = max_figures > 0 and sent >= max_figures
        if image is not None and max(image.size) >= min_pixels and not capped:
            sent += 1
            yield page_no, image
        else:
            yield page_no, None


def annotate_document_figures(
    doc: Any,
    annotator: Annotator,
    *,
    min_pixels: int = 200,
    max_figures: int = 0,
) -> dict[int, list[str]]:
    """Transcribe each figure in ``doc``; return ``{page_no: [markdown, ...]}``.

    Iterates ``doc.pictures`` in reading order (via :func:`iter_figure_slots`). The returned
    per-page lists keep one slot per picture in order (an empty string for a skipped or
    failed figure), so the caller can align them positionally with the page's
    ``<!-- image -->`` placeholders.

    * Figures whose long edge is under ``min_pixels`` are skipped (empty slot).
    * ``max_figures`` (> 0) caps how many figures are actually sent to the annotator --
      a cost bound for test runs; remaining figures get empty slots.
    * A failed annotator call logs and yields an empty slot -- one figure lost, never the
      document. This is the whole point of running outside docling's pipeline.
    """
    figures_by_page: dict[int, list[str]] = {}
    for page_no, image in iter_figure_slots(doc, min_pixels=min_pixels, max_figures=max_figures):
        text = ""
        if image is not None:
            try:
                text = annotator(image).strip()
            except Exception as exc:  # noqa: BLE001 -- lose one figure, not the document
                logger.warning(
                    "Figure VLM failed on page %s (%s); leaving it untranscribed.", page_no, exc
                )
        figures_by_page.setdefault(page_no, []).append(text)
    return figures_by_page


def inject_figures(page_md: str, figure_texts: list[str]) -> str:
    """Splice ``figure_texts`` into ``page_md`` in place of its ``<!-- image -->`` markers.

    Placeholders and ``figure_texts`` are both in reading order, so the k-th placeholder
    is replaced by ``figure_texts[k]``. An empty text drops that placeholder. Each figure's
    Markdown is surrounded by blank lines so a transcribed GFM table is a standalone block
    that :func:`chunk_markdown` keeps whole.

    If the counts disagree (docling emitted a different number of placeholders than we have
    pictures for this page), fall back to stripping the placeholders and appending all
    non-empty texts at the end of the page -- still on the correct page, just not perfectly
    inline. A mismatch therefore never drops or misplaces figure content.
    """
    parts = page_md.split(_IMAGE_PLACEHOLDER)
    if len(parts) - 1 == len(figure_texts):
        out = [parts[0]]
        for text, tail in zip(figure_texts, parts[1:], strict=True):
            out.append(f"\n\n{text}\n\n" if text else "")
            out.append(tail)
        return "".join(out)
    # Count mismatch: strip markers, append what we have so nothing is lost.
    stripped = page_md.replace(_IMAGE_PLACEHOLDER, "")
    extra = "\n\n".join(text for text in figure_texts if text)
    return f"{stripped}\n\n{extra}" if extra else stripped
