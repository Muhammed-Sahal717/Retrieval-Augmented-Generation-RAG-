import os
import chromadb
from typing import List, Dict, Any

from rag_cli.processing.chunker import DocumentChunk


class ChromaStore:
    """Manages local storage and retrieval of vector embeddings using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "rag_documents"):
        """
        Initializes the ChromaDB client with persistent storage.
        """
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def store_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """
        Stores document chunks and their corresponding embeddings into ChromaDB.
        
        Args:
            chunks: A list of DocumentChunk objects.
            embeddings: A list of embedding vectors (list of floats).
        """
        if not chunks or not embeddings:
            return
            
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")
            
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            
            # Metadata allows us to filter or display sources later
            metadata = {
                "source_document": chunk.source_document,
                "chunk_index": chunk.chunk_index
            }
            metadatas.append(metadata)
            
        # Add to collection
        # We pass embeddings explicitly because we generated them via Gemini, not a Chroma built-in function
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query_embedding: List[float], n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the vector database for the most relevant chunks.
        
        Args:
            query_embedding: The embedding of the user's question.
            n_results: Top K results to return.
            
        Returns:
            A list of result dictionaries containing chunk information.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        
        # Results are lists of lists since query can take multiple embeddings
        if not results['ids'] or not results['ids'][0]:
            return formatted_results
            
        for idx in range(len(results['ids'][0])):
            res = {
                "id": results['ids'][0][idx],
                "document": results['documents'][0][idx],
                "metadata": results['metadatas'][0][idx],
                "distance": results['distances'][0][idx] if 'distances' in results and results['distances'] else None
            }
            formatted_results.append(res)
            
        return formatted_results
