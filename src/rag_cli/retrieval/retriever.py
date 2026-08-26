from typing import List, Dict, Any

from rag_cli.embeddings.gemini_embedder import GeminiEmbedder
from rag_cli.vector_store.chroma_store import ChromaStore

class Retriever:
    """Handles the semantic retrieval of relevant document chunks for a given query."""
    
    def __init__(self, embedder: GeminiEmbedder, vector_store: ChromaStore):
        """
        Initializes the Retriever with an embedder and a vector store.
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the most semantically relevant chunks for the given query.
        
        Args:
            query: The natural language question from the user.
            top_k: The number of results to return (default is 3).
            
        Returns:
            A list of result dictionaries containing chunk information.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
            
        # 1. Generate an embedding for the question
        query_embedding = self.embedder.embed_text(query)
        
        # 2. Search the vector database for the top K results
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k
        )
        
        return results
