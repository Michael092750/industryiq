"""Unit tests for manifest-derived provenance metadata.

Covers the domain -> source_type classifier and the ``manifest.csv`` reader /
sibling-join used to stamp publisher / published_date / source_type onto chunks.
"""

from pathlib import Path

from industryiq.core.ingestion.manifest import (
    classify_source_type,
    load_manifest,
    manifest_metadata,
)

_HEADER = "status,url,filename,domain,detected_year,size_bytes,sha256,error"


def _write_manifest(path: Path, *rows: str) -> None:
    path.write_text("\n".join((_HEADER, *rows)) + "\n", encoding="utf-8")


def test_classify_consultancy() -> None:
    assert classify_source_type("www.mckinsey.com") == "consultancy"


def test_classify_government_and_igo() -> None:
    assert classify_source_type("agency.gov") == "government"
    assert classify_source_type("www.worldbank.org") == "government"


def test_classify_association_and_company() -> None:
    assert classify_source_type("some-trade-body.org") == "association"
    assert classify_source_type("www.rabobankna.com") == "company"


def test_classify_unknown_is_empty() -> None:
    assert classify_source_type("weird.xyz") == ""
    assert classify_source_type("") == ""


def test_load_manifest_maps_filename_to_provenance(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        "downloaded,https://www.rabobankna.com/x.pdf,report.pdf,www.rabobankna.com,2025,1,abc,",
    )
    assert load_manifest(manifest)["report.pdf"] == {
        "publisher": "rabobankna.com",
        "source_type": "company",
        "published_date": "2025",
    }


def test_load_manifest_omits_absent_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    # No domain and no year -> nothing to attach for this file.
    _write_manifest(manifest, "downloaded,https://x/y.pdf,bare.pdf,,,1,abc,")
    assert load_manifest(manifest)["bare.pdf"] == {}


def test_load_manifest_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_manifest(tmp_path / "nope.csv") == {}


def test_manifest_metadata_joins_by_filename_and_caches(tmp_path: Path) -> None:
    category = tmp_path / "AI"
    category.mkdir()
    _write_manifest(
        category / "manifest.csv",
        "downloaded,https://www.mckinsey.com/x.pdf,paper.pdf,www.mckinsey.com,2024,1,abc,",
    )
    pdf = category / "paper.pdf"
    pdf.write_bytes(b"%PDF-")

    cache: dict[Path, dict[str, dict[str, object]]] = {}
    meta = manifest_metadata(pdf, cache)
    assert meta == {
        "publisher": "mckinsey.com",
        "source_type": "consultancy",
        "published_date": "2024",
    }
    # Second call is served from the cache (still exactly one manifest parsed).
    assert manifest_metadata(pdf, cache) == meta
    assert len(cache) == 1


def test_manifest_metadata_no_manifest_returns_empty(tmp_path: Path) -> None:
    pdf = tmp_path / "loose.pdf"
    pdf.write_bytes(b"%PDF-")
    assert manifest_metadata(pdf, {}) == {}


def test_manifest_metadata_file_not_listed_returns_empty(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "manifest.csv", "downloaded,u,other.pdf,www.x.com,2024,1,a,")
    pdf = tmp_path / "missing.pdf"
    pdf.write_bytes(b"%PDF-")
    assert manifest_metadata(pdf, {}) == {}
