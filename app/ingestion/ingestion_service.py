from pathlib import Path

from app.ingestion.chunker import DocumentChunker
from app.ingestion.cleaner import TextCleaner
from app.ingestion.models import DocumentChunk, DocumentPage, ParsedDocument
from app.ingestion.parser_factory import ParserFactory


class IngestionService:
    """Coordinates the document ingestion pipeline."""

    def __init__(self) -> None:
        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker()

    def ingest(self, file_path: Path) -> list[DocumentChunk]:
        parser = ParserFactory.get_parser(file_path)

        document = parser.parse(file_path)

        cleaned_pages = [
            DocumentPage(
                page_number=page.page_number,
                text=self.cleaner.clean(page.text),
            )
            for page in document.pages
        ]

        cleaned_document = ParsedDocument(
            pages=cleaned_pages,
            metadata=document.metadata,
        )

        return self.chunker.chunk(cleaned_document)
