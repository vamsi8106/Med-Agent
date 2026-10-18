from pathlib import Path
from unittest.mock import AsyncMock, patch

from medagent.cli import build_arg_parser, ingest_guideline, main


def test_build_arg_parser_parses_ingest_guideline() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        ["ingest-guideline", "--file", "guideline.pdf", "--source", "ada-2024"]
    )
    assert args.command == "ingest-guideline"
    assert args.file == Path("guideline.pdf")
    assert args.source == "ada-2024"


async def test_ingest_guideline_missing_file_returns_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.pdf"
    result = await ingest_guideline(missing, "ada-2024")
    assert result == 1


async def test_ingest_guideline_text_file_calls_pipeline(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("Metformin is first-line therapy.")

    fake_pipeline = AsyncMock()
    fake_pipeline.ingest_text.return_value = 3

    with (
        patch("medagent.cli.EmbeddingModel"),
        patch("medagent.cli.VectorStore"),
        patch("medagent.cli.IngestionPipeline", return_value=fake_pipeline),
    ):
        result = await ingest_guideline(text_file, "internal-notes")

    assert result == 0
    fake_pipeline.ingest_text.assert_awaited_once_with(
        "Metformin is first-line therapy.", "internal-notes"
    )


async def test_ingest_guideline_pdf_file_calls_pipeline(tmp_path: Path) -> None:
    pdf_file = tmp_path / "guideline.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    fake_pipeline = AsyncMock()
    fake_pipeline.ingest_pdf.return_value = 5

    with (
        patch("medagent.cli.EmbeddingModel"),
        patch("medagent.cli.VectorStore"),
        patch("medagent.cli.IngestionPipeline", return_value=fake_pipeline),
    ):
        result = await ingest_guideline(pdf_file, "ada-2024")

    assert result == 0
    fake_pipeline.ingest_pdf.assert_awaited_once_with(str(pdf_file), "ada-2024")


def test_main_dispatches_to_ingest_guideline(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("Some guideline text.")

    def _run_and_close(coro: object) -> int:
        coro.close()  # type: ignore[attr-defined]
        return 0

    with patch("medagent.cli.asyncio.run", side_effect=_run_and_close) as fake_run:
        result = main(["ingest-guideline", "--file", str(text_file), "--source", "test-source"])

    assert result == 0
    fake_run.assert_called_once()


def test_build_arg_parser_parses_purge_expired() -> None:
    parser = build_arg_parser()
    assert parser.parse_args(["purge-expired"]).execute is False
    assert parser.parse_args(["purge-expired", "--execute"]).execute is True


def test_purge_without_retention_configured_is_a_noop(capsys: object) -> None:
    from medagent.core.config import Settings

    with patch("medagent.cli.get_settings", return_value=Settings(_env_file=None)):
        assert main(["purge-expired", "--execute"]) == 0
