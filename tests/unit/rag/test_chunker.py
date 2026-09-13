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
