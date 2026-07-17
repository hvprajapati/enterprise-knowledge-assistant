"""Offline indexing CLI.

Builds a FAISS index and SQLite metadata database from a directory of documents.

Usage::

    python scripts/build_index.py ^
        --input data/raw ^
        --index storage/index.faiss ^
        --database storage/metadata.db
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``app`` is importable.
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.indexing.index_builder import IndexBuilder  # noqa: E402
from app.storage.repository import ChunkRepository  # noqa: E402
from app.storage.schema import create_schema  # noqa: E402
from app.storage.sqlite import SQLiteConnection  # noqa: E402

logger = logging.getLogger("build_index")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from a folder of documents.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing source documents (.pdf, .docx, .txt, .md).",
    )
    parser.add_argument(
        "--index",
        required=True,
        type=Path,
        help="Output path for the FAISS index (e.g. storage/index.faiss).",
    )
    parser.add_argument(
        "--database",
        required=True,
        type=Path,
        help="Output path for the SQLite metadata database (e.g. storage/metadata.db).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks to embed per batch (default: 32).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # validate inputs
    # ------------------------------------------------------------------
    if not args.input.is_dir():
        print(f"ERROR: --input is not a directory: {args.input}")
        sys.exit(1)

    # Ensure output directories exist
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.database.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------
    # wire everything together
    # ------------------------------------------------------------------
    logger.info("Connecting to database: %s", args.database)
    sqlite = SQLiteConnection(args.database)
    connection = sqlite.connect()

    try:
        logger.info("Creating schema …")
        create_schema(connection)

        repository = ChunkRepository(connection)

        builder = IndexBuilder(
            repository,
            batch_size=args.batch_size,
        )

        logger.info("Starting indexing from: %s", args.input)
        logger.info("FAISS index will be saved to: %s", args.index)

        stats = builder.build(
            input_folder=args.input,
            index_path=args.index,
        )

        # ------------------------------------------------------------------
        # summary
        # ------------------------------------------------------------------
        print()
        print("=" * 56)
        print("  Indexing Complete")
        print("=" * 56)
        print(f"  Files processed:      {stats.files_processed:>6d}")
        print(f"  Chunks created:       {stats.chunks_created:>6d}")
        print(f"  Embeddings generated: {stats.embeddings_generated:>6d}")
        print(f"  FAISS index:          {args.index}")
        print(f"  Metadata database:    {args.database}")
        print("=" * 56)

    finally:
        connection.close()
        logger.info("Database connection closed.")


if __name__ == "__main__":
    main()
