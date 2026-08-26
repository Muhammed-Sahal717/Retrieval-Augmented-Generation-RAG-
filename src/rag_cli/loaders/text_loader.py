from pathlib import Path
from typing import ClassVar


class TextLoader:
    """Loads and extracts text from local .txt and .md files."""
    
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".txt", ".md"}

    @classmethod
    def load(cls, file_path: str | Path) -> str:
        """
        Validates and loads text from the given file path.
        
        Args:
            file_path: Path to the document.
            
        Returns:
            Extracted and cleaned text.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported or the file is empty.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Error: Document not found at {path}")
            
        if not path.is_file():
            raise ValueError(f"Error: {path} is not a file.")
            
        if path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Error: Unsupported file format '{path.suffix}'. Supported formats are {', '.join(cls.SUPPORTED_EXTENSIONS)}.")
            
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Clean text by stripping leading/trailing whitespace
        cleaned_text = text.strip()
        
        if not cleaned_text:
            raise ValueError(f"Error: Document {path} is empty.")
            
        return cleaned_text
