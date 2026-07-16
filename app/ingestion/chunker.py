from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.models import (
    DocumentChunk,
    ParsedDocument,
)


class DocumentChunker:
    """Splits parsed documents into semantic chunks."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
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

            for chunk in split_chunks:
                chunks.append(
                    DocumentChunk(
                        document_id=document.metadata.document_id,
                        chunk_index=chunk_index,
                        page_number=page.page_number,
                        text=chunk,
                    )
                )

                chunk_index += 1

        return chunks