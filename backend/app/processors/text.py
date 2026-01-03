"""Plain text document processor.

Handles .txt, .md, and other plain text files.
"""

from typing import List, Optional
from app.processors.base import DocumentProcessor, ProcessorConfig
from app.processors.registry import ProcessorRegistry


@ProcessorRegistry.register
class PlainTextProcessor(DocumentProcessor):
    """Processor for plain text files."""
    
    name = "Plain Text Processor"
    supported_extensions = [".txt", ".md", ".markdown", ".rst", ".text"]
    supported_mimetypes = [
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "text/x-rst",
    ]
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize the plain text processor."""
        super().__init__(config)
    
    def extract(self, content: bytes, filename: str) -> str:
        """Extract text from plain text files.
        
        Args:
            content: Raw bytes of the file
            filename: Original filename
            
        Returns:
            Decoded text content
            
        Raises:
            ValueError: If the file cannot be decoded as UTF-8
        """
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            # Try other common encodings
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            raise ValueError(
                f"File '{filename}' could not be decoded. "
                "Tried UTF-8, Latin-1, CP1252, and ISO-8859-1."
            )


# Set as default processor for unknown file types
ProcessorRegistry.set_default(PlainTextProcessor)
