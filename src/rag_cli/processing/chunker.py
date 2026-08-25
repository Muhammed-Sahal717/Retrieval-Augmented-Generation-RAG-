import hashlib
from dataclasses import dataclass
from typing import List

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_document: str
    chunk_index: int


class TextChunker:
    """Splits text into smaller chunks with configurable size and overlap."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, text: str, source_document: str) -> List[DocumentChunk]:
        """
        Splits the document text into character-based chunks.
        
        Args:
            text: The extracted text from the document.
            source_document: The name or path of the source document.
            
        Returns:
            A list of DocumentChunk objects.
        """
        if not text:
            return []

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        
        # Ensure we always make progress
        if step <= 0:
            step = 1

        for i in range(0, len(text), step):
            chunk_text = text[i : i + self.chunk_size]
            
            # Generate a unique ID for the chunk
            chunk_id = self._generate_chunk_id(source_document, i // step)
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_document=source_document,
                chunk_index=i // step,
            )
            chunks.append(chunk)
            
        return chunks

    def _generate_chunk_id(self, source_document: str, index: int) -> str:
        """Generates a stable, unique ID for a chunk based on its source and index."""
        raw_id = f"{source_document}_{index}"
        return hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:16]
