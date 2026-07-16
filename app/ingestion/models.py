from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: UUID = Field(default_factory=uuid4)
    filename: str
    file_path: Path
    extension: str
    file_size: int

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