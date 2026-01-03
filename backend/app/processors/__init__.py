"""Document Processor Plugin Registry.

This module provides a plugin-based architecture for document text extraction.
Each processor handles specific file types and can be registered dynamically.

Usage:
    from app.processors import ProcessorRegistry
    
    processor = ProcessorRegistry.get_processor("document.pdf")
    text = processor.extract(file_content, "document.pdf")
"""

from app.processors.base import DocumentProcessor, ProcessorConfig
from app.processors.registry import ProcessorRegistry
from app.processors.factory import get_processor, extract_text

__all__ = [
    "DocumentProcessor",
    "ProcessorConfig",
    "ProcessorRegistry",
    "get_processor",
    "extract_text",
]
