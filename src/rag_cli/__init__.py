import argparse
import sys
from rag_cli.services.rag_service import RAGService

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG CLI - Command-line Retrieval-Augmented Generation application"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document into the RAG system")
    ingest_parser.add_argument("file_path", type=str, help="Path to the document (e.g., .txt or .md)")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question based on ingested documents")
    ask_parser.add_argument("question", type=str, help="Your natural-language question")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    try:
        service = RAGService()
    except Exception as e:
        print(f"Error initializing RAG Service: {e}")
        sys.exit(1)
        
    if args.command == "ingest":
        try:
            service.ingest_document(args.file_path)
            print("Successfully ingested the document.")
        except Exception as e:
            print(f"Error: {e}")
            print("Please check the file path and try again.")
            
    elif args.command == "ask":
        try:
            print(f"Question:\n{args.question}\n")
            result = service.ask_question(args.question)
            print("Answer:")
            print(result["answer"])
            print("\nSources:")
            if result["sources"]:
                for i, source in enumerate(result["sources"], 1):
                    print(f"{i}. {source['source_document']} — Chunk {source['chunk_index']}")
            else:
                print("No sources found.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
