"""CLI entrypoint for one-off admin tasks, starting with guideline ingestion.

Usage:
    uv run python -m medagent.cli ingest-guideline --file path/to/guideline.pdf --source ada-2024
    uv run python -m medagent.cli ingest-guideline --file path/to/notes.txt --source internal-notes
    uv run python -m medagent.cli purge-expired            # dry run: counts only
    uv run python -m medagent.cli purge-expired --execute  # delete past retention
"""

import argparse
import asyncio
import sys
from pathlib import Path

from medagent.core.config import get_settings
from medagent.infra.logging import configure_logging, get_logger
from medagent.memory.persistent import PersistentStore
from medagent.memory.retention import purge_expired
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

    purge = subparsers.add_parser(
        "purge-expired",
        help="Delete patient records and audit entries past RETENTION_DAYS / "
        "AUDIT_RETENTION_DAYS (dry run unless --execute).",
    )
    purge.add_argument(
        "--execute", action="store_true", help="Actually delete (default: only count)"
    )

    return parser


async def purge_expired_records(execute: bool) -> int:
    settings = get_settings()
    if settings.retention_days is None and settings.audit_retention_days is None:
        print("Retention is not configured (RETENTION_DAYS / AUDIT_RETENTION_DAYS); nothing to do.")
        return 0
    store = PersistentStore(settings.postgres_dsn)
    try:
        result = await purge_expired(
            store,
            retention_days=settings.retention_days,
            audit_retention_days=settings.audit_retention_days,
            execute=execute,
        )
    finally:
        await store.close()
    logger.info(
        "retention_purge",
        executed=result.executed,
        patients=result.patients,
        audit_entries=result.audit_entries,
    )
    verb = "Deleted" if execute else "Would delete (dry run, pass --execute)"
    print(
        f"{verb}: {result.patients if result.patients is not None else 'n/a'} patient records, "
        f"{result.audit_entries if result.audit_entries is not None else 'n/a'} audit entries"
    )
    return 0


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
    # Deliberate stdout confirmation for the human running this CLI
    # interactively -- structlog's JSON log line above is for log aggregation,
    # not for reading in a terminal.
    print(f"Ingested {chunk_count} chunks from {file_path} (source={source_name})")
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest-guideline":
        return asyncio.run(ingest_guideline(args.file, args.source))
    if args.command == "purge-expired":
        return asyncio.run(purge_expired_records(args.execute))

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
