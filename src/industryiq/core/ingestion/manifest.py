"""Provenance metadata from a category's ``manifest.csv``.

The report collector drops a ``manifest.csv`` next to the PDFs in each category
folder (see the ``report-library`` layout). Its columns
``status,url,filename,domain,detected_year,size_bytes,sha256,error`` record where
each file came from. This module turns those rows into the document-level chunk
metadata the manifest can supply:

* ``publisher``       <- ``domain`` (the registrable host, ``www.`` stripped)
* ``published_date``  <- ``detected_year`` (a year; ISO-sortable as the stores want)
* ``source_type``     <- ``domain``, via the heuristic :func:`classify_source_type`

Files with no manifest row (or no manifest at all) simply get none of these --
absent provenance is left unset, never guessed.
"""

import csv
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.csv"

# Per-scan memo of parsed manifests, keyed by manifest path -> (filename -> meta).
ManifestCache = dict[Path, dict[str, dict[str, Any]]]

# Heuristic domain -> source_type table. Deliberately small and explicit: it is
# the obvious starting point, easy to extend, and any unrecognised domain is left
# unclassified rather than mislabeled. Match is on the second-level domain (e.g.
# "mckinsey" in "www.mckinsey.com").
_CONSULTANCIES = frozenset(
    {
        "mckinsey",
        "bcg",
        "bain",
        "deloitte",
        "pwc",
        "kpmg",
        "ey",
        "accenture",
        "gartner",
        "forrester",
        "idc",
        "oliverwyman",
        "rolandberger",
        "kearney",
        "strategyand",
        "capgemini",
    }
)
# Inter-governmental / multilateral bodies (treated as government for filtering;
# the project's source preference flags these as *not* industry analysis).
_IGO = frozenset(
    {"worldbank", "imf", "oecd", "un", "wto", "weforum", "europa", "iea", "who", "unctad"}
)


def classify_source_type(domain: str) -> str:
    """Best-effort publisher *type* for a host, or ``""`` when unrecognised.

    Returns one of ``consultancy`` / ``government`` / ``academic`` /
    ``association`` / ``company``. The classification is intentionally coarse and
    heuristic -- enough to facet retrieval by who published a report -- and errs
    toward ``""`` (unset) over a wrong label for hosts it doesn't recognise.
    """
    labels = domain.lower().strip().removeprefix("www.").split(".")
    labels = [label for label in labels if label]
    if not labels:
        return ""
    tld = labels[-1]
    sld = labels[-2] if len(labels) >= 2 else labels[0]
    if sld in _CONSULTANCIES:
        return "consultancy"
    if sld in _IGO or tld in {"gov", "mil", "int"}:
        return "government"
    if tld == "edu" or "ac" in labels:
        return "academic"
    if tld == "org":
        return "association"
    if tld in {"com", "net", "io", "ai", "co", "biz"}:
        return "company"
    return ""


def _row_metadata(row: dict[str, str]) -> dict[str, Any]:
    """Map one manifest row to the doc-level metadata keys it can supply."""
    meta: dict[str, Any] = {}
    domain = (row.get("domain") or "").strip()
    if domain:
        publisher = domain.lower().removeprefix("www.")
        if publisher:
            meta["publisher"] = publisher
        source_type = classify_source_type(domain)
        if source_type:
            meta["source_type"] = source_type
    year = (row.get("detected_year") or "").strip()
    if year:
        meta["published_date"] = year
    return meta


def load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Read ``manifest_path``, returning ``filename -> doc-level metadata``.

    Unreadable or malformed manifests yield ``{}`` -- a missing manifest must
    never abort an ingest, only leave provenance unset.
    """
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        filename = (row.get("filename") or "").strip()
        if filename:
            result[filename] = _row_metadata(row)
    return result


def manifest_metadata(file_path: Path, cache: ManifestCache) -> dict[str, Any]:
    """Doc-level metadata for ``file_path`` from its sibling ``manifest.csv``.

    ``cache`` memoizes each manifest by path across a scan, so a category's
    manifest is parsed once however many files it lists. Returns ``{}`` when the
    file has no manifest or no row in it.
    """
    manifest_path = file_path.parent / MANIFEST_NAME
    if manifest_path not in cache:
        cache[manifest_path] = load_manifest(manifest_path) if manifest_path.is_file() else {}
    return dict(cache[manifest_path].get(file_path.name, {}))
