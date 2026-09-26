from medagent.rag.chunker import chunk_text


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
