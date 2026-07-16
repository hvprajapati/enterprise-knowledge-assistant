from pathlib import Path

from app.ingestion.parser import (
    BaseParser,
    DOCXParser,
    MarkdownParser,
    PDFParser,
    TXTParser,
)


class ParserFactory:
    """Returns the appropriate parser based on file extension."""

    _parsers: dict[str, BaseParser] = {
        ".pdf": PDFParser(),
        ".docx": DOCXParser(),
        ".txt": TXTParser(),
        ".md": MarkdownParser(),
    }

    @classmethod
    def get_parser(cls, file_path: Path) -> BaseParser:
        extension = file_path.suffix.lower()

        parser = cls._parsers.get(extension)

        if parser is None:
            raise ValueError(f"Unsupported file type: {extension}")

        return parser
