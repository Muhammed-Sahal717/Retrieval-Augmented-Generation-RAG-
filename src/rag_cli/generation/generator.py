import os
from typing import Any

from google import genai


class Generator:
    """Handles the generation of answers using Gemini based on retrieved context."""
    
    DEFAULT_MODEL = "gemini-3.6-flash"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        """
        Initializes the generator with the Gemini client.
        """
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
             raise ValueError("GEMINI_API_KEY must be provided or set in environment variables.")
             
        self.client = genai.Client(api_key=key)
        self.model = model

    def build_prompt(self, question: str, context_chunks: list[dict[str, Any]]) -> str:
        """
        Builds the RAG prompt by combining the question, instructions, and context.
        """
        # Format the retrieved chunks into a single text block
        formatted_context = ""
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("metadata", {}).get("source_document", "Unknown")
            chunk_text = chunk.get("document", "")
            formatted_context += f"--- Chunk {i} (Source: {source}) ---\n{chunk_text}\n\n"
            
        prompt = f"""You are a helpful assistant answering questions based on the provided context.

Context:
{formatted_context.strip()}

Question:
{question}

Instructions:
- Answer using the provided context.
- If the answer is not available in the context, say that you do not have enough information.
- Do not make up information.

Answer:
"""
        return prompt

    def generate_answer(self, question: str, context_chunks: list[dict[str, Any]]) -> str:
        """
        Builds the prompt and sends it to Gemini to generate the grounded answer.
        
        Args:
            question: The user's question.
            context_chunks: The chunks retrieved from the vector database.
            
        Returns:
            The generated answer string.
        """
        prompt = self.build_prompt(question, context_chunks)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        return response.text
