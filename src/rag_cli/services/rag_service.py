import os
from typing import Dict, Any

from rag_cli.loaders.text_loader import TextLoader
from rag_cli.processing.chunker import TextChunker
from rag_cli.embeddings.gemini_embedder import GeminiEmbedder
from rag_cli.vector_store.chroma_store import ChromaStore
from rag_cli.retrieval.retriever import Retriever
from rag_cli.generation.generator import Generator


class RAGService:
    """Orchestrates the entire RAG pipeline: ingestion and question answering."""
    
    def __init__(self):
        """Initializes all underlying components of the RAG pipeline."""
        # Ensure we have the API key available
        if not os.getenv("GEMINI_API_KEY"):
             raise ValueError("GEMINI_API_KEY must be set in environment variables.")
             
        self.chunker = TextChunker()
        self.embedder = GeminiEmbedder()
        self.vector_store = ChromaStore()
        self.retriever = Retriever(embedder=self.embedder, vector_store=self.vector_store)
        self.generator = Generator()

    def ingest_document(self, file_path: str) -> None:
        """
        Executes the Document Ingestion Flow.
        Loads the document, splits it into chunks, generates embeddings, and stores them.
        """
        # 1. Load document
        text = TextLoader.load(file_path)
        
        # 2. Split into chunks
        filename = os.path.basename(file_path)
        chunks = self.chunker.chunk_document(text, source_document=filename)
        
        if not chunks:
            raise ValueError(f"No text could be extracted or chunked from {file_path}")
            
        # 3. Generate embeddings
        embeddings = self.embedder.embed_chunks(chunks)
        
        # 4. Store chunks and embeddings
        self.vector_store.store_chunks(chunks, embeddings)

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Executes the Question Answering Flow.
        Retrieves relevant context and generates an answer.
        
        Returns:
            A dictionary containing the generated 'answer' and the 'sources' used.
        """
        # 1 & 2. Embed query and Retrieve top relevant chunks
        results = self.retriever.retrieve(query=question)
        
        if not results:
            return {
                "answer": "I don't have enough information in my database to answer that. Please ingest some documents first.",
                "sources": []
            }
            
        # 3. Generate answer
        answer = self.generator.generate_answer(question=question, context_chunks=results)
        
        # Format sources to return
        sources = []
        for res in results:
            metadata = res.get("metadata", {})
            sources.append({
                "source_document": metadata.get("source_document", "Unknown"),
                "chunk_index": metadata.get("chunk_index", -1)
            })
            
        return {
            "answer": answer,
            "sources": sources
        }
