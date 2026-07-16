from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_serializer


class DocumentMetadata(BaseModel):
    document_id: UUID = Field(default_factory=uuid4)

    filename: str
    file_path: Path
    extension: str
    file_size: int

    source: str | None = None
    document_type: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_serializer("file_path")
    def serialize_file_path(self, value: Path) -> str:
        return str(value)


class DocumentPage(BaseModel):
    page_number: int
    text: str


class ParsedDocument(BaseModel):
    pages: list[DocumentPage]
    metadata: DocumentMetadata


class DocumentChunk(BaseModel):
    chunk_id: UUID = Field(default_factory=uuid4)

    document_id: UUID
    chunk_index: int
    page_number: int | None = None

    text: str

    metadata: DocumentMetadata


class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
