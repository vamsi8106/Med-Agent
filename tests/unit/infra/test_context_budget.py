from medagent.infra.context_budget import approx_token_count, truncate_text


def test_approx_token_count_estimates_from_character_length() -> None:
    assert approx_token_count("") == 0
    assert approx_token_count("x" * 350) == 100


def test_approx_token_count_is_not_fooled_by_long_unspaced_tokens() -> None:
    """Regression: the old whitespace-word estimate counted this as 1 token."""
    url = "https://doi.org/10.1016/j.example.2024.01.001/" * 10
    assert len(url.split()) == 1
    assert approx_token_count(url) > 100


def test_truncate_text_returns_unchanged_when_within_budget() -> None:
    text = "one two three"
    assert truncate_text(text, max_tokens=10, source="test") == text


def test_truncate_text_cuts_and_marks_when_over_budget() -> None:
    text = " ".join(f"word{i}" for i in range(200))

    result = truncate_text(text, max_tokens=10, source="test")

    assert result.startswith("word0 word1 word2")
    assert "truncated" in result
    assert "word199" not in result
    assert len(result) < len(text)


def test_truncate_text_never_ends_mid_word() -> None:
    text = " ".join(["abcdefghij"] * 50)

    result = truncate_text(text, max_tokens=10, source="test")

    body = result.split("\n")[0]
    assert all(word == "abcdefghij" for word in body.split())


def test_truncate_text_boundary_exact_length_is_not_truncated() -> None:
    text = "x" * 35
    assert truncate_text(text, max_tokens=10, source="test") == text


def test_truncations_are_counted_by_source() -> None:
    from tests.conftest import metric_value

    before = metric_value("medagent_context_truncations_total", source="test-source")

    truncate_text("word " * 500, max_tokens=10, source="test-source")
    truncate_text("short", max_tokens=10, source="test-source")  # untouched: not counted

    assert metric_value("medagent_context_truncations_total", source="test-source") - before == 1
