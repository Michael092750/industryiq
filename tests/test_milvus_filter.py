"""Unit tests for the pure Milvus filter compiler + publisher normalisation.

No live Milvus -- only the pure module helpers are exercised (importing the module
does not open a connection).
"""

from industryiq.core.milvusvectorstore import _compile_filter_expr, _publisher_like_value
from industryiq.core.vectorstore import MetadataFilter

# --- publisher normalisation --------------------------------------------------


def test_publisher_name_reduces_to_lowercase_token() -> None:
    assert _publisher_like_value("Deloitte") == "deloitte"


def test_publisher_domain_is_kept() -> None:
    assert _publisher_like_value("deloitte.com") == "deloitte.com"


def test_publisher_strips_leading_www_and_lowercases() -> None:
    assert _publisher_like_value("WWW.Deloitte.com") == "deloitte.com"


def test_publisher_drops_like_wildcards_and_quotes() -> None:
    # A like-wildcard (% or _), quote, or space can't survive -> injection-safe.
    assert _publisher_like_value('a"% _b') == "ab"


# --- filter compilation -------------------------------------------------------


def test_publisher_compiles_to_a_contains_like() -> None:
    got = _compile_filter_expr(MetadataFilter(publisher="Deloitte"))
    assert got == 'publisher like "%deloitte%"'


def test_domain_publisher_matches_subdomains_via_contains() -> None:
    # "deloitte.com" as a substring is what hits www2.deloitte.com / mkto.deloitte.com.
    got = _compile_filter_expr(MetadataFilter(publisher="deloitte.com"))
    assert got == 'publisher like "%deloitte.com%"'


def test_source_type_and_category_stay_exact_equality() -> None:
    assert _compile_filter_expr(MetadataFilter(source_type="consultancy")) == (
        'source_type == "consultancy"'
    )
    assert _compile_filter_expr(MetadataFilter(category="AI")) == 'category == "AI"'


def test_dates_are_an_inclusive_range_that_admits_undated() -> None:
    # A date bound narrows dated chunks but never drops undated ones (published_date
    # == ""): publication year != content year, and ~37% of the corpus is undated.
    got = _compile_filter_expr(MetadataFilter(published_from="2024", published_to="2025"))
    assert got == (
        '((published_date >= "2024" && published_date <= "2025") || published_date == "")'
    )


def test_single_date_bound_also_admits_undated() -> None:
    got = _compile_filter_expr(MetadataFilter(published_from="2024"))
    assert got == '((published_date >= "2024") || published_date == "")'


def test_date_bound_is_grouped_when_anded_with_other_clauses() -> None:
    # The date OR must be parenthesised so it binds within the && chain, not across it.
    got = _compile_filter_expr(MetadataFilter(category="AI", published_from="2024"))
    assert got == ('category == "AI" && ((published_date >= "2024") || published_date == "")')


def test_clauses_are_anded() -> None:
    got = _compile_filter_expr(MetadataFilter(publisher="Deloitte", category="AI"))
    assert got == 'publisher like "%deloitte%" && category == "AI"'


def test_empty_filter_compiles_to_empty_string() -> None:
    assert _compile_filter_expr(MetadataFilter()) == ""


def test_publisher_that_sanitizes_to_empty_is_dropped() -> None:
    # No safe chars -> no clause, rather than `like "%%"` which would match everything.
    assert _compile_filter_expr(MetadataFilter(publisher="!!!")) == ""
