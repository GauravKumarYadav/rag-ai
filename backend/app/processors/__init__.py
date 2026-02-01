"""Document Processor Module.

This module provides document text extraction using Docling as the unified processor.

Usage:
    from app.processors import get_docling_processor
    
    processor = get_docling_processor()
    text = processor.extract(file_content, "document.pdf")
    
    # Or for async:
    text = await processor.extract_async(file_content, "document.pdf")
"""

from app.processors.base import DocumentProcessor, ProcessorConfig
from app.processors.docling_processor import DoclingProcessor, get_docling_processor
from app.processors.registry import ProcessorRegistry
from app.processors.chunking import (
    Chunk,
    ChunkMetadata,
    chunk_document,
    chunk_text_simple,
    generate_doc_id,
    generate_chunk_id,
)


def extract_text(
    content: bytes, 
    filename: str, 
    mimetype: str = None,
    config: "ProcessorConfig" = None,
) -> str:
    """
    Extract text from document content using Docling.
    
    This is a convenience function that uses the unified Docling processor.
    
    Args:
        content: The raw file content as bytes
        filename: The filename (used to determine file type)
        mimetype: Optional MIME type (not used, kept for API compatibility)
        config: Optional processor config (not used, kept for API compatibility)
        
    Returns:
        Extracted text content
    """
    processor = get_docling_processor()
    return processor.extract(content, filename)


async def extract_text_async(content: bytes, filename: str) -> str:
    """
    Extract text from document content using Docling (async version).
    
    Args:
        content: The raw file content as bytes
        filename: The filename (used to determine file type)
        
    Returns:
        Extracted text content
    """
    processor = get_docling_processor()
    return await processor.extract_async(content, filename)


# Initialize registry with Docling as default
ProcessorRegistry.set_default(DoclingProcessor)

__all__ = [
    # Base classes
    "DocumentProcessor",
    "ProcessorConfig",
    # Docling processor
    "DoclingProcessor",
    "get_docling_processor",
    # Registry
    "ProcessorRegistry",
    # Convenience functions
    "extract_text",
    "extract_text_async",
    # Chunking
    "Chunk",
    "ChunkMetadata",
    "chunk_document",
    "chunk_text_simple",
    "generate_doc_id",
    "generate_chunk_id",
]
