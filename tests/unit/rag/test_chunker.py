from medagent.rag.chunker import chunk_text, is_reference_list


def test_chunk_text_single_section_no_headers() -> None:
    text = "word " * 100
    chunks = chunk_text(text, chunk_size_tokens=50, overlap_tokens=10)
    assert len(chunks) == 3
    assert all(c.section is None for c in chunks)


def test_chunk_text_splits_by_section_header() -> None:
    text = "# Diagnosis\nSome diagnosis text.\n\n# Treatment\nSome treatment text."
    chunks = chunk_text(text, chunk_size_tokens=512, overlap_tokens=64)
    sections = {c.section for c in chunks}
    assert "Diagnosis" in sections
    assert "Treatment" in sections


def test_chunk_text_overlap_between_windows() -> None:
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size_tokens=10, overlap_tokens=5)
    assert len(chunks) == 3
    assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]


def _sections(text: str) -> list[str | None]:
    return [c.section for c in chunk_text(text)]


def test_numbered_headings_become_sections() -> None:
    text = "2.2 Risk prediction in people with CKD\nSome text.\n3.2.2 Physical activity\nMore text."
    assert _sections(text) == ["2.2 Risk prediction in people with CKD", "3.2.2 Physical activity"]


def test_all_caps_multi_word_title_is_a_section() -> None:
    assert _sections("INTRODUCTION AND METHODOLOGY\nBody text here.") == [
        "INTRODUCTION AND METHODOLOGY"
    ]


def test_table_labels_and_single_words_are_not_sections() -> None:
    for junk in ("CKD G1", "NA/NA", "A C A C", "CITATION", "HTN"):
        assert junk not in {s for s in _sections(f"Intro words.\n{junk}\nBody words follow.") if s}


def test_doses_and_reference_numbers_are_not_sections() -> None:
    text = "Take.\n2.5 mg once daily\n1. Smith J, et al. Some paper\n3 Other\nEnd."
    assert _sections(text) == [None]


def _two_tokens_per_word(words: list[str]) -> list[int]:
    return [2] * len(words)


def test_chunks_fit_the_token_budget_including_the_context_prefix() -> None:
    text = "# Blood pressure\n" + " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text(
        text,
        chunk_size_tokens=40,
        overlap_tokens=8,
        count_tokens=_two_tokens_per_word,
        title="kdigo",
    )

    assert {c.context for c in chunks} == {"kdigo › Blood pressure"}
    for chunk in chunks:
        assert chunk.embedding_text == f"{chunk.context}\n{chunk.text}"
        # 4 context words + the chunk's words, 2 tokens each
        assert 2 * len(chunk.embedding_text.split()) <= 40
    # 40 - 8 (context) = 32 tokens = 16 words per chunk
    assert len(chunks[0].text.split()) == 16


def test_overlap_is_measured_in_tokens() -> None:
    text = " ".join(f"w{i}" for i in range(60))
    chunks = chunk_text(
        text, chunk_size_tokens=20, overlap_tokens=6, count_tokens=_two_tokens_per_word
    )

    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]  # 6 tokens = 3 words


def test_a_word_longer_than_the_budget_becomes_its_own_chunk() -> None:
    def huge_first_word(words: list[str]) -> list[int]:
        return [100 if w == "giant" else 1 for w in words]

    chunks = chunk_text(
        "giant a b c", chunk_size_tokens=10, overlap_tokens=2, count_tokens=huge_first_word
    )

    assert chunks[0].text == "giant"
    assert chunks[-1].text.endswith("c")


def test_without_a_token_counter_windows_are_words_as_before() -> None:
    words = [f"w{i}" for i in range(1000)]
    chunks = chunk_text(" ".join(words), chunk_size_tokens=512, overlap_tokens=64)

    assert [len(c.text.split()) for c in chunks] == [512, 512, 104]
    assert chunks[1].text.split()[0] == "w448"
    assert all(c.context is None and c.embedding_text == c.text for c in chunks)


def test_reference_lists_are_dropped_on_request() -> None:
    body = "Blood pressure should be measured in a standardized way before treatment. " * 5
    references = " ".join(
        f"{i}. Smith J, et al. A trial of something. Kidney Int. 2023;{i}:1{i}-2{i}."
        for i in range(8)
    )
    text = f"# Blood pressure\n{body}\n# Citations\n{references}"

    kept = chunk_text(text, chunk_size_tokens=512, overlap_tokens=64, drop_reference_lists=True)
    everything = chunk_text(text, chunk_size_tokens=512, overlap_tokens=64)

    assert [c.section for c in kept] == ["Blood pressure"]
    assert len(everything) == 2


def test_body_text_with_one_citation_is_not_a_reference_list() -> None:
    paragraph = (
        "As Smith et al. showed, lowering blood pressure reduced cardiovascular events. "
        + "Treatment should be adjusted to tolerance and reviewed at each visit. " * 4
    )
    assert not is_reference_list(paragraph)
    assert is_reference_list(
        "Lee K, et al. JAMA. 2023;330:1-9. Ito M, et al. 2021;99:5-8. Roe P. 2020;1:2"
    )
