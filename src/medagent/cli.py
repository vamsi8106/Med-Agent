"""CLI entrypoint for one-off admin tasks, starting with guideline ingestion.

Usage:
    uv run python -m medagent.cli ingest-guideline --file path/to/guideline.pdf --source ada-2024
    uv run python -m medagent.cli ingest-guideline --file path/to/notes.txt --source internal-notes
"""

import argparse
import asyncio
import sys
from pathlib import Path

from medagent.core.config import get_settings
from medagent.infra.logging import configure_logging, get_logger
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.pipeline import IngestionPipeline
from medagent.rag.vector_store import VectorStore

logger = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medagent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest-guideline", help="Ingest a PDF or text guideline into the RAG vector store."
    )
    ingest.add_argument("--file", required=True, type=Path, help="Path to a .pdf or .txt file")
    ingest.add_argument(
        "--source", required=True, help="Source label stored with each chunk, e.g. 'ada-2024'"
    )

    return parser


async def ingest_guideline(file_path: Path, source_name: str) -> int:
    if not file_path.exists():
        logger.error("ingest_file_not_found", file=str(file_path))
        return 1

    settings = get_settings()
    pipeline = IngestionPipeline(
        embeddings=EmbeddingModel(),
        vector_store=VectorStore(settings.chroma_host, settings.chroma_port),
    )

    if file_path.suffix.lower() == ".pdf":
        chunk_count = await pipeline.ingest_pdf(str(file_path), source_name)
    else:
        text = file_path.read_text()
        chunk_count = await pipeline.ingest_text(text, source_name)

    logger.info(
        "guideline_ingestion_complete",
        file=str(file_path),
        source=source_name,
        chunk_count=chunk_count,
    )
    print(f"Ingested {chunk_count} chunks from {file_path} (source={source_name})")
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest-guideline":
        return asyncio.run(ingest_guideline(args.file, args.source))

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
