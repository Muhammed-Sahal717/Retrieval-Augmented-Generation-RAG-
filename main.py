from dotenv import load_dotenv
from rag_cli import main as cli_main

def main():
    load_dotenv()
    cli_main()

if __name__ == "__main__":
    main()
