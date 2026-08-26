import os

from google import genai

from rag_cli.processing.chunker import DocumentChunk


class GeminiEmbedder:
    """Generates vector embeddings for text using the Gemini API."""
    
    # Using gemini-embedding-2 as standard for Gemini embeddings
    DEFAULT_MODEL = "gemini-embedding-2"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        """
        Initialize the embedder.
        If api_key is None, it relies on the GEMINI_API_KEY environment variable.
        """
        # Ensure we have the API key available, either explicitly or via env
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
             raise ValueError("GEMINI_API_KEY must be provided or set in environment variables.")
        
        self.client = genai.Client(api_key=key)
        self.model = model

    def embed_text(self, text: str) -> list[float]:
        """
        Generates an embedding for a single string of text.
        Useful for generating embeddings for user queries.
        """
        if not text.strip():
            raise ValueError("Text cannot be empty.")
            
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        # In google-genai, the embedding is stored in the response.embeddings list
        # We assume response.embeddings[0].values contains the floats
        return response.embeddings[0].values

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        """
        Generates embeddings for a list of DocumentChunks.
        Returns a list of embedding vectors corresponding to each chunk.
        """
        if not chunks:
            return []
            
        # In google-genai, to embed multiple texts we can iterate or use a batch mechanism if supported.
        # For simplicity in this MVP, we will iterate.
        embeddings = []
        for chunk in chunks:
            response = self.client.models.embed_content(
                model=self.model,
                contents=chunk.text
            )
            embeddings.append(response.embeddings[0].values)
            
        return embeddings
