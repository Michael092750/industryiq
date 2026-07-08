"""Tests for the checkpointed Batch-API figure re-ingest (industryiq.core.figure_batch).

Covers the offline, provider-independent machinery: the collect phase's figure selection +
crop saving, request building, batch submit (with the id persisted before return), result
polling/fetch, the pure result->pages splice, and the write phase's resume (skip already-
written docs). The docling parse and the live Anthropic calls are faked.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from industryiq.config import get_settings
from industryiq.core import figure_batch
from industryiq.core.figure_batch import (
    DocPlan,
    FigureSlot,
    JobState,
    build_requests,
    collect_all,
    collect_document,
    fetch_results,
    load_doc_plan,
    load_job,
    plan_to_pages,
    poll_until_ended,
    save_doc_plan,
    submit_batch,
    write_all,
)
from industryiq.core.ingestion.adapters.store_memory import InMemoryIngestStateStore
from industryiq.core.ingestion.models import FileState

PLACEHOLDER = "<!-- image -->"
TABLE = "| metric | value |\n|---|---|\n| variable rate debt | 29.4 percent |"


def _fake_doc(pictures: list[object], page_count: int = 2) -> object:
    return SimpleNamespace(pictures=pictures, pages={n: object() for n in range(1, page_count + 1)})


def _fake_picture(page_no: int, image: Image.Image | None) -> object:
    return SimpleNamespace(prov=[SimpleNamespace(page_no=page_no)], get_image=lambda _doc: image)


# --- plan_to_pages (pure splice) -------------------------------------------- #


def test_plan_to_pages_injects_results_positionally() -> None:
    plan = DocPlan(
        source="AI/a.pdf",
        content_hash="h",
        size=1,
        title="a",
        metadata={},
        pages=[f"Intro.\n\n{PLACEHOLDER}\n\nEnd.", f"Second {PLACEHOLDER} page."],
        slots=[
            FigureSlot(page_no=1, custom_id="k-0000", image="figs/k/0000.png"),
            FigureSlot(page_no=2, custom_id="k-0001", image="figs/k/0001.png"),
        ],
    )
    # second figure errored in the batch -> empty string
    pages = plan_to_pages(plan, {"k-0000": TABLE, "k-0001": ""})
    assert TABLE in pages[0]
    assert PLACEHOLDER not in pages[0]
    assert PLACEHOLDER not in pages[1]  # empty result still consumes/drops its placeholder


def test_plan_to_pages_empty_slot_and_missing_result_are_blank() -> None:
    plan = DocPlan(
        source="a.pdf",
        content_hash="h",
        size=1,
        title="a",
        metadata={},
        pages=[f"{PLACEHOLDER} then {PLACEHOLDER}"],
        slots=[
            FigureSlot(page_no=1),  # skipped-small: no custom_id -> blank
            FigureSlot(page_no=1, custom_id="k-0001", image="figs/k/0001.png"),  # missing result
        ],
    )
    pages = plan_to_pages(plan, {})  # no results at all
    assert PLACEHOLDER not in pages[0]


# --- collect phase ----------------------------------------------------------- #


def test_collect_document_selects_saves_and_slots(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "AI" / "a.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4 dummy")
    work = tmp_path / "work"

    pages = [f"Body.\n\n{PLACEHOLDER}\n\nMore.", f"Small fig {PLACEHOLDER} here."]
    doc = _fake_doc(
        [
            _fake_picture(1, Image.new("RGB", (400, 300))),  # selected
            _fake_picture(2, Image.new("RGB", (50, 50))),  # below min_pixels -> empty slot
        ]
    )
    monkeypatch.setattr(figure_batch, "_docling_pages_and_doc", lambda _p: (pages, doc))

    plan = collect_document(
        pdf,
        source="AI/a.pdf",
        metadata={"category": "AI"},
        title="a",
        work=work,
        settings=get_settings(),
    )

    assert plan.pages == pages
    assert [s.custom_id is not None for s in plan.slots] == [True, False]
    # the selected figure's crop was written to disk
    saved = plan.slots[0].image
    assert saved is not None and (work / saved).is_file()
    # and the empty slot carries its page but no request
    assert plan.slots[1].page_no == 2 and plan.slots[1].image is None


def test_collect_all_skips_already_collected(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "corpus"
    (root / "AI").mkdir(parents=True)
    (root / "AI" / "x.pdf").write_bytes(b"%PDF dummy")
    work = tmp_path / "work"
    # Pre-seed the plan so collect_all treats this file as done.
    save_doc_plan(
        work,
        DocPlan(source="AI/x.pdf", content_hash="h", size=1, title="x", metadata={}, pages=["p"]),
    )
    # If it tried to parse, this would blow up -- proving it skipped instead.
    monkeypatch.setattr(
        figure_batch,
        "_docling_pages_and_doc",
        lambda _p: (_ for _ in ()).throw(AssertionError("should not parse a collected doc")),
    )
    collected, failures = collect_all(root, work, get_settings())
    assert collected == 0
    assert failures == []


def test_collect_document_non_pdf_captures_text(tmp_path: Path, monkeypatch) -> None:
    txt = tmp_path / "notes.txt"
    txt.write_text("hello world\nsecond line", encoding="utf-8")
    work = tmp_path / "work"
    # A non-PDF must not touch docling -- this would blow up if it tried.
    monkeypatch.setattr(
        figure_batch,
        "_docling_pages_and_doc",
        lambda _p: (_ for _ in ()).throw(AssertionError("no docling for non-pdf")),
    )
    plan = collect_document(
        txt, source="notes.txt", metadata={}, title="notes", work=work, settings=get_settings()
    )
    assert plan.slots == []  # no figures to batch
    assert any("hello world" in page for page in plan.pages)


def test_docling_pages_and_doc_applies_hybrid_recovery(tmp_path: Path, monkeypatch) -> None:
    # Regression: the batch parse must run the SAME pypdf hybrid net as the loader, or a
    # bulk re-ingest silently loses chart footnotes/annotations Docling drops. This mirrors
    # tests/test_loaders_docling.py::test_docling_applies_hybrid_recovery for the batch path.
    from industryiq.core import loaders

    class _Doc:
        pages = {1: object()}

        def export_to_markdown(self, page_no: int | None = None) -> str:
            return "# Page one\n\n<!-- image -->\n\ndocling kept this prose"

    def _fake_converter() -> object:
        return SimpleNamespace(convert=lambda _s: SimpleNamespace(document=_Doc()))

    monkeypatch.setattr(loaders, "_get_docling_converter", _fake_converter)
    monkeypatch.setattr(
        loaders,
        "_load_pdf_pages_pypdf",
        lambda _p: ["docling kept this prose\nrecovered figure footnote with plenty of words"],
    )
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    pages, _doc = figure_batch._docling_pages_and_doc(pdf)
    joined = "\n".join(pages)

    assert "recovered figure footnote with plenty of words" in joined  # dropped text restored
    assert joined.count("docling kept this prose") == 1  # kept text not duplicated
    assert "<!-- image -->" in joined  # placeholder preserved for the batch write phase


def test_collect_all_incremental_skips_unchanged_via_manifest(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "old.txt").write_text("already ingested", encoding="utf-8")
    (root / "new.txt").write_text("brand new report", encoding="utf-8")
    work = tmp_path / "work"
    store = InMemoryIngestStateStore()
    store.upsert_file_state(
        FileState(
            source="old.txt",
            size=1,
            content_hash=figure_batch._file_hash(root / "old.txt"),
            chunk_count=3,
        )
    )
    collected, failures = collect_all(root, work, get_settings(), store=store)
    assert failures == []
    assert collected == 1  # only the new file
    assert load_doc_plan(work, "new.txt") is not None
    assert load_doc_plan(work, "old.txt") is None  # unchanged -> skipped


# --- submit phase ------------------------------------------------------------ #


class _FakeBatches:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []  # one entry per create() (one per batch chunk)

    def create(self, *, requests: list[dict]) -> object:
        self.calls.append(requests)
        return SimpleNamespace(id=f"msgbatch_{len(self.calls)}")


def _fake_client(batches: _FakeBatches) -> object:
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def _seed_one_figure(work: Path) -> None:
    """A collected plan with a single selected figure crop on disk."""
    figs = work / "figs" / "k"
    figs.mkdir(parents=True)
    Image.new("RGB", (400, 300)).save(figs / "0000.png", format="PNG")
    save_doc_plan(
        work,
        DocPlan(
            source="AI/a.pdf",
            content_hash="h",
            size=1,
            title="a",
            metadata={"category": "AI"},
            pages=[f"Body {PLACEHOLDER}."],
            slots=[FigureSlot(page_no=1, custom_id="k-0000", image="figs/k/0000.png")],
        ),
    )


def test_build_requests_reads_each_saved_crop(tmp_path: Path) -> None:
    _seed_one_figure(tmp_path)
    reqs = build_requests(tmp_path)
    assert len(reqs) == 1
    assert reqs[0]["custom_id"] == "k-0000"
    assert reqs[0]["png_b64"]  # base64 of the crop


def test_build_requests_min_pixels_prunes_small_crops(tmp_path: Path) -> None:
    figs = tmp_path / "figs" / "k"
    figs.mkdir(parents=True)
    Image.new("RGB", (250, 180)).save(figs / "0000.png", format="PNG")  # small -> pruned at 300
    Image.new("RGB", (420, 300)).save(figs / "0001.png", format="PNG")  # kept at 300
    save_doc_plan(
        tmp_path,
        DocPlan(
            source="AI/a.pdf",
            content_hash="h",
            size=1,
            title="a",
            metadata={},
            pages=[f"{PLACEHOLDER} {PLACEHOLDER}"],
            slots=[
                FigureSlot(page_no=1, custom_id="k-0000", image="figs/k/0000.png"),
                FigureSlot(page_no=1, custom_id="k-0001", image="figs/k/0001.png"),
            ],
        ),
    )
    assert len(build_requests(tmp_path)) == 2  # no filter
    reqs = build_requests(tmp_path, min_pixels=300)
    assert [r["custom_id"] for r in reqs] == ["k-0001"]  # only the >=300px crop survives


def test_submit_batch_persists_id_and_shapes_request(tmp_path: Path) -> None:
    _seed_one_figure(tmp_path)
    job = JobState(root=str(tmp_path), model="claude-sonnet-5", phase="submitting")
    batches = _FakeBatches()

    updated = submit_batch(tmp_path, job, _fake_client(batches))

    assert updated.phase == "submitted"
    assert updated.batch_ids == ["msgbatch_1"]  # one chunk for the single small crop
    # the id is persisted to disk (so a restart re-attaches, not resubmits)
    assert load_job(tmp_path).batch_ids == ["msgbatch_1"]
    # request carries model + the shared image+prompt content
    req = batches.calls[0][0]
    assert req["custom_id"] == "k-0000"
    assert req["params"]["model"] == "claude-sonnet-5"
    kinds = [block["type"] for block in req["params"]["messages"][0]["content"]]
    assert kinds == ["image", "text"]


def test_submit_batch_splits_when_over_byte_cap(tmp_path: Path) -> None:
    # Three crops with a tiny byte cap -> one batch each (packing is size-driven).
    figs = tmp_path / "figs" / "k"
    figs.mkdir(parents=True)
    slots = []
    for i in range(3):
        Image.new("RGB", (400, 300)).save(figs / f"{i:04d}.png", format="PNG")
        slots.append(FigureSlot(page_no=1, custom_id=f"k-{i:04d}", image=f"figs/k/{i:04d}.png"))
    save_doc_plan(
        tmp_path,
        DocPlan(
            source="a.pdf",
            content_hash="h",
            size=1,
            title="a",
            metadata={},
            pages=["p"],
            slots=slots,
        ),
    )
    job = JobState(root=str(tmp_path), model="m", phase="submitting")
    batches = _FakeBatches()
    updated = submit_batch(tmp_path, job, _fake_client(batches), max_batch_bytes=1)
    assert len(batches.calls) == 3  # split into three batches
    assert updated.batch_ids == ["msgbatch_1", "msgbatch_2", "msgbatch_3"]


def test_submit_batch_resumes_without_resubmitting_done_chunks(tmp_path: Path) -> None:
    figs = tmp_path / "figs" / "k"
    figs.mkdir(parents=True)
    slots = []
    for i in range(3):
        Image.new("RGB", (400, 300)).save(figs / f"{i:04d}.png", format="PNG")
        slots.append(FigureSlot(page_no=1, custom_id=f"k-{i:04d}", image=f"figs/k/{i:04d}.png"))
    save_doc_plan(
        tmp_path,
        DocPlan(
            source="a.pdf",
            content_hash="h",
            size=1,
            title="a",
            metadata={},
            pages=["p"],
            slots=slots,
        ),
    )
    # Pretend the first two chunks already submitted on a prior run.
    job = JobState(root=str(tmp_path), model="m", phase="submitting", batch_ids=["old-0", "old-1"])
    batches = _FakeBatches()
    updated = submit_batch(tmp_path, job, _fake_client(batches), max_batch_bytes=1)
    assert len(batches.calls) == 1  # only the third chunk is (re)sent
    assert updated.batch_ids == ["old-0", "old-1", "msgbatch_1"]


def test_submit_batch_with_no_figures_skips_to_writing(tmp_path: Path) -> None:
    save_doc_plan(
        tmp_path,
        DocPlan(
            source="a.pdf", content_hash="h", size=1, title="a", metadata={}, pages=["no figs"]
        ),
    )
    job = JobState(root=str(tmp_path), model="m", phase="submitting")
    updated = submit_batch(tmp_path, job, _fake_client(_FakeBatches()))
    assert updated.phase == "writing"
    assert updated.batch_ids == []


# --- poll / fetch ------------------------------------------------------------ #


def test_poll_until_ended_waits_then_returns(monkeypatch) -> None:
    statuses = iter(["in_progress", "in_progress", "ended"])

    class _Batches:
        def retrieve(self, _id: str) -> object:
            return SimpleNamespace(processing_status=next(statuses), request_counts=None)

    monkeypatch.setattr(figure_batch.time, "sleep", lambda _s: None)
    poll_until_ended(_fake_client_from(_Batches()), ["msgbatch_test"], poll_seconds=0)
    # returning at all means it saw "ended"; the iterator is now exhausted
    with pytest.raises(StopIteration):
        next(statuses)


def _fake_client_from(batches: object) -> object:
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def test_fetch_results_maps_success_and_drops_failures() -> None:
    def _entry(cid: str, type_: str, text: str | None = None) -> object:
        if type_ == "succeeded":
            message = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
            result = SimpleNamespace(type="succeeded", message=message)
            return SimpleNamespace(custom_id=cid, result=result)
        return SimpleNamespace(custom_id=cid, result=SimpleNamespace(type=type_))

    class _Batches:
        def results(self, _id: str):
            return iter(
                [
                    _entry("k-0000", "succeeded", TABLE),
                    _entry("k-0001", "errored"),
                    _entry("k-0002", "expired"),
                ]
            )

    out = fetch_results(_fake_client_from(_Batches()), ["msgbatch_test"])
    assert out == {"k-0000": TABLE, "k-0001": "", "k-0002": ""}


# --- write phase (resume) ---------------------------------------------------- #


class _FakePipeline:
    def __init__(self) -> None:
        self.ingested: list[str] = []
        self.deleted: list[str] = []

    def delete_source(self, source: str) -> int:
        self.deleted.append(source)
        return 0

    def ingest_pages(self, pages, *, source, metadata, title) -> list[str]:
        self.ingested.append(source)
        return ["id1", "id2"]


class _FakeStore:
    def __init__(self) -> None:
        self.states: list[object] = []

    def upsert_file_state(self, state: object) -> None:
        self.states.append(state)


def test_write_all_skips_written_and_is_idempotent(tmp_path: Path) -> None:
    save_doc_plan(
        tmp_path,
        DocPlan(
            source="AI/done.pdf",
            content_hash="h1",
            size=1,
            title="d",
            metadata={},
            pages=["p"],
            written=True,
        ),
    )
    save_doc_plan(
        tmp_path,
        DocPlan(
            source="AI/new.pdf",
            content_hash="h2",
            size=2,
            title="n",
            metadata={"category": "AI"},
            pages=["p2"],
        ),
    )
    pipeline, store = _FakePipeline(), _FakeStore()

    written = write_all(tmp_path, {}, pipeline, store)

    assert written == 1
    assert pipeline.ingested == ["AI/new.pdf"]  # the already-written doc is skipped
    assert pipeline.deleted == ["AI/new.pdf"]  # delete-then-reindex
    assert len(store.states) == 1
    # the doc is now flagged written on disk -> a re-run does nothing
    assert load_doc_plan(tmp_path, "AI/new.pdf").written is True
    assert write_all(tmp_path, {}, _FakePipeline(), _FakeStore()) == 0


def test_write_all_skips_failed_plan_without_deleting(tmp_path: Path) -> None:
    # A doc that crashed the parser (failed=True) must be left entirely alone -- in
    # particular NOT delete_source'd, or its existing chunks would be wiped.
    save_doc_plan(
        tmp_path,
        DocPlan(
            source="AI/crash.pdf",
            content_hash="h",
            size=1,
            title="crash",
            metadata={},
            pages=[],
            slots=[],
            failed=True,
        ),
    )
    pipeline, store = _FakePipeline(), _FakeStore()
    assert write_all(tmp_path, {}, pipeline, store) == 0
    assert pipeline.ingested == []
    assert pipeline.deleted == []  # no delete -> existing chunks preserved
    assert store.states == []
