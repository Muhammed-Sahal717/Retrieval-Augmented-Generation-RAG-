# RAG CLI

A minimal, educational command-line application that demonstrates the complete **Retrieval-Augmented Generation (RAG)** pipeline.

The goal of this project is to help developers learn how a RAG system works internally without obscuring the core concepts behind high-level orchestration frameworks.

## Architecture & Technology Stack

The pipeline is explicitly modularized so that each core concept can be understood independently:

- **Language:** Python
- **LLM & Embeddings:** [Gemini API](https://aistudio.google.com/) (`gemini-3.6-flash` and `gemini-embedding-2`)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Local persistent storage)
- **Dependency Management:** `uv` (or pip)

### Data Flows

**Document Ingestion:**
1. **Load:** Reads local `.txt` or `.md` files.
2. **Chunk:** Splits the text into character-based chunks with overlap.
3. **Embed:** Generates a vector embedding for each chunk using Gemini.
4. **Store:** Persists the chunks, metadata, and embeddings locally using ChromaDB.

**Question Answering:**
1. **Query:** User submits a natural language question.
2. **Embed:** Generates an embedding for the user's question.
3. **Retrieve:** Searches the ChromaDB for the top K most semantically relevant chunks.
4. **Generate:** Combines the retrieved context with the question in a strict RAG prompt to Gemini to produce a grounded answer.

## Prerequisites

- Python 3.14+
- `uv` (Recommended) or `pip`
- A [Google Gemini API Key](https://aistudio.google.com/app/apikey)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd RAG-CLI
   ```

2. **Set up your environment variables:**
   Copy the example environment file and add your API key.
   ```bash
   cp .env.example .env
   # Edit .env and insert your GEMINI_API_KEY
   ```

3. **Install dependencies:**
   Using `uv`:
   ```bash
   uv sync
   ```
   Or using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This project natively uses `uv.lock` and `pyproject.toml`)*

## Usage

You interact with the RAG CLI through the `main.py` entry point.

### 1. Ingest a Document
Add a text or markdown document into your local vector database:

```bash
uv run python main.py ingest ./path/to/your/document.txt
```

### 2. Ask a Question
Query the ingested documents. The system will retrieve relevant context and generate an answer with sources:

```bash
uv run python main.py ask "What is the main topic of the document?"
```

## Project Structure

```
RAG-CLI/
├── main.py                          # CLI entry point wrapper
├── src/rag_cli/
│   ├── __init__.py                  # CLI arguments and routing
│   ├── config.py                    # Environment configuration
│   ├── loaders/text_loader.py       # Reads and validates files
│   ├── processing/chunker.py        # Splits text into chunks
│   ├── embeddings/gemini_embedder.py# Generates vector embeddings
│   ├── vector_store/chroma_store.py # Local ChromaDB integration
│   ├── retrieval/retriever.py       # Semantic search orchestration
│   └── generation/generator.py      # LLM prompt builder & generator
│   └── services/rag_service.py      # High-level pipeline controller
├── chroma_db/                       # (Auto-generated) Vector database storage
├── pyproject.toml                   # Project metadata and dependencies
└── .env                             # Secrets configuration
```

## Future Enhancements
As defined in the PRD, potential future additions include:
- PDF and DOCX support
- Recursive or semantic chunking strategies
- Metadata filtering & Hybrid Search
- Streaming responses
- Conversation history

## License
MIT
