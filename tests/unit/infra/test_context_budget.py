from medagent.infra.context_budget import approx_token_count, truncate_text


def test_approx_token_count_counts_whitespace_separated_words() -> None:
    assert approx_token_count("one two three") == 3
    assert approx_token_count("") == 0


def test_truncate_text_returns_unchanged_when_within_budget() -> None:
    text = "one two three"
    assert truncate_text(text, max_tokens=10, source="test") == text


def test_truncate_text_cuts_and_marks_when_over_budget() -> None:
    text = " ".join(f"word{i}" for i in range(20))

    result = truncate_text(text, max_tokens=5, source="test")

    assert result.startswith("word0 word1 word2 word3 word4")
    assert "truncated" in result
    assert "word19" not in result


def test_truncate_text_boundary_exact_length_is_not_truncated() -> None:
    text = "one two three"
    assert truncate_text(text, max_tokens=3, source="test") == text
