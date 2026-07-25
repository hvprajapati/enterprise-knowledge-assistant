import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.models import (
    DocumentChunk,
    ParsedDocument,
)

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits parsed documents into semantic chunks.

    Parameters
    ----------
    chunk_size:
        Maximum characters per chunk (default from Settings: 800).
    chunk_overlap:
        Overlap between adjacent chunks (default from Settings: 100).
    min_chunk_size:
        Chunks shorter than this are skipped (default from Settings: 10).
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_size: int = 10,
    ) -> None:
        self.min_chunk_size = min_chunk_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def chunk(
        self,
        document: ParsedDocument,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        chunk_index = 0

        for page in document.pages:
            split_chunks = self.splitter.split_text(page.text)

            for chunk_text in split_chunks:
                # Skip empty / near-empty chunks
                if len(chunk_text.strip()) < self.min_chunk_size:
                    continue

                chunks.append(
                    DocumentChunk(
                        document_id=document.metadata.document_id,
                        chunk_index=chunk_index,
                        page_number=page.page_number,
                        text=chunk_text,
                        metadata=document.metadata,
                    )
                )

                chunk_index += 1

        skipped = sum(1 for p in document.pages
                      for t in self.splitter.split_text(p.text)
                      if len(t.strip()) < self.min_chunk_size)
        if skipped:
            logger.debug("Skipped %d chunks below min_chunk_size (%d chars)",
                         skipped, self.min_chunk_size)

        return chunks
