"""Factory functions for document processing.

Provides convenient entry points for extracting text from documents.
"""

from typing import Optional

from app.processors.base import ProcessorConfig
from app.processors.registry import ProcessorRegistry

# Import all processors to register them
from app.processors.text import PlainTextProcessor
from app.processors.pdf import PDFProcessor
from app.processors.docx import DOCXProcessor
from app.processors.image import ImageProcessor


def get_processor(
    filename: str,
    mimetype: Optional[str] = None,
    config: Optional[ProcessorConfig] = None,
):
    """Get the appropriate processor for a file.
    
    Args:
        filename: Name of the file to process
        mimetype: Optional MIME type of the file
        config: Optional processor configuration
        
    Returns:
        An instance of the appropriate processor
        
    Raises:
        ValueError: If no processor is found for the file type
    """
    return ProcessorRegistry.get_processor(filename, mimetype, config)


def extract_text(
    content: bytes,
    filename: str,
    mimetype: Optional[str] = None,
    config: Optional[ProcessorConfig] = None,
) -> str:
    """Extract text from a document.
    
    Convenience function that gets the appropriate processor
    and extracts text in one call.
    
    Args:
        content: Raw bytes of the document
        filename: Name of the file (used for format detection)
        mimetype: Optional MIME type of the file
        config: Optional processor configuration
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If no processor is found or extraction fails
    """
    processor = get_processor(filename, mimetype, config)
    return processor.extract(content, filename)


def can_process(filename: str, mimetype: Optional[str] = None) -> bool:
    """Check if a file type can be processed.
    
    Args:
        filename: Name of the file to check
        mimetype: Optional MIME type of the file
        
    Returns:
        True if a processor is available for this file type
    """
    return ProcessorRegistry.can_process(filename, mimetype)


def list_supported_extensions() -> list:
    """List all supported file extensions.
    
    Returns:
        List of supported file extensions
    """
    return ProcessorRegistry.list_processors()


def list_supported_mimetypes() -> list:
    """List all supported MIME types.
    
    Returns:
        List of supported MIME types
    """
    return ProcessorRegistry.list_mimetypes()
