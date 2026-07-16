from abc import ABC, abstractmethod
from pathlib import Path

import fitz
from docx import Document

from app.ingestion.models import (
    DocumentMetadata,
    DocumentPage,
    ParsedDocument,
)


class BaseParser(ABC):
    """Base interface for all document parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        raise NotImplementedError


def build_metadata(file_path: Path) -> DocumentMetadata:
    return DocumentMetadata(
        filename=file_path.name,
        file_path=file_path.resolve(),
        extension=file_path.suffix.lower(),
        file_size=file_path.stat().st_size,
    )


class PDFParser(BaseParser):
    def parse(self, file_path: Path) -> ParsedDocument:
        pdf = fitz.open(file_path)

        pages = [
            DocumentPage(
                page_number=index + 1,
                text=page.get_text("text"),
            )
            for index, page in enumerate(pdf)
        ]

        pdf.close()

        return ParsedDocument(
            pages=pages,
            metadata=build_metadata(file_path),
        )


class DOCXParser(BaseParser):
    def parse(self, file_path: Path) -> ParsedDocument:
        document = Document(file_path)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        return ParsedDocument(
            pages=[
                DocumentPage(
                    page_number=1,
                    text=text,
                )
            ],
            metadata=build_metadata(file_path),
        )


class TXTParser(BaseParser):
    def parse(self, file_path: Path) -> ParsedDocument:
        text = file_path.read_text(encoding="utf-8")

        return ParsedDocument(
            pages=[
                DocumentPage(
                    page_number=1,
                    text=text,
                )
            ],
            metadata=build_metadata(file_path),
        )


class MarkdownParser(BaseParser):
    def parse(self, file_path: Path) -> ParsedDocument:
        text = file_path.read_text(encoding="utf-8")

        return ParsedDocument(
            pages=[
                DocumentPage(
                    page_number=1,
                    text=text,
                )
            ],
            metadata=build_metadata(file_path),
        )