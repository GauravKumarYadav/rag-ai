"""Base classes for document processors.

This module defines the abstract interface that all document processors must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProcessorConfig:
    """Configuration options for document processors."""
    
    # OCR settings
    use_ocr: bool = True
    fast_mode: bool = False
    
    # Performance settings (throttled so uploads do not saturate CPU)
    num_threads: int = 4
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    # Docling-specific settings
    do_table_structure: bool = True
    do_picture_classification: bool = False
    do_picture_description: bool = False
    do_code_enrichment: bool = False
    do_formula_enrichment: bool = False
    
    # Batch sizes
    ocr_batch_size: int = 8
    layout_batch_size: int = 8
    table_batch_size: int = 8


class DocumentProcessor(ABC):
    """Abstract base class for document processors.
    
    Each processor handles specific file types and implements the extract method
    to convert document content to plain text.
    
    Attributes:
        name: Human-readable name for the processor
        supported_extensions: List of file extensions this processor handles (e.g., [".pdf", ".PDF"])
        supported_mimetypes: List of MIME types this processor handles
    """
    
    name: str = "Base Processor"
    supported_extensions: List[str] = []
    supported_mimetypes: List[str] = []
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize the processor with optional configuration.
        
        Args:
            config: Processor configuration options
        """
        self.config = config or ProcessorConfig()
    
    @abstractmethod
    def extract(self, content: bytes, filename: str) -> str:
        """Extract text content from a document.
        
        Args:
            content: Raw bytes of the document
            filename: Original filename (used for format detection)
            
        Returns:
            Extracted text content as a string
            
        Raises:
            ValueError: If extraction fails or content is invalid
        """
        pass
    
    def can_process(self, filename: str, mimetype: Optional[str] = None) -> bool:
        """Check if this processor can handle the given file.
        
        Args:
            filename: Name of the file to check
            mimetype: Optional MIME type of the file
            
        Returns:
            True if this processor can handle the file
        """
        # Check by extension
        ext = self._get_extension(filename)
        if ext in [e.lower() for e in self.supported_extensions]:
            return True
        
        # Check by mimetype if provided
        if mimetype and mimetype in self.supported_mimetypes:
            return True
        
        return False
    
    def _get_extension(self, filename: str) -> str:
        """Get the lowercase file extension including the dot."""
        if "." in filename:
            return "." + filename.rsplit(".", 1)[-1].lower()
        return ""
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(extensions={self.supported_extensions})"
