"""
Unified Document Processor using Docling.

This module provides a single processor for all supported document types
using the Docling library for extraction.

Supported formats:
- PDF (with OCR support)
- DOCX/DOC
- Images (PNG, JPG, JPEG, TIFF, BMP, GIF, WEBP)
- Plain text (TXT, MD)
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import Docling
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling not available. Install with: pip install docling")


class DoclingProcessor:
    """
    Unified document processor using Docling for all document types.
    
    Falls back to simple text decoding for plain text files.
    """
    
    SUPPORTED_EXTENSIONS = [
        '.pdf', '.docx', '.doc',
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp',
        '.txt', '.md', '.markdown', '.rst', '.text'
    ]
    
    PLAIN_TEXT_EXTENSIONS = ['.txt', '.md', '.markdown', '.rst', '.text']
    
    def __init__(self) -> None:
        self._converter: Optional[DocumentConverter] = None
    
    @property
    def converter(self) -> Optional[DocumentConverter]:
        """Lazy-load the Docling converter."""
        if self._converter is None and DOCLING_AVAILABLE:
            try:
                self._converter = DocumentConverter()
                logger.info("Docling converter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Docling converter: {e}")
        return self._converter
    
    def can_process(self, filename: str) -> bool:
        """
        Check if a file can be processed.
        
        Args:
            filename: Name of the file
            
        Returns:
            True if the file type is supported
        """
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def extract(self, content: bytes, filename: str) -> str:
        """
        Extract text from a document.
        
        Args:
            content: File content as bytes
            filename: Original filename (for extension detection)
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If the file type is not supported
        """
        ext = Path(filename).suffix.lower()
        
        if not self.can_process(filename):
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Handle plain text files directly
        if ext in self.PLAIN_TEXT_EXTENSIONS:
            return self._extract_text(content)
        
        # Use Docling for other formats
        if not DOCLING_AVAILABLE:
            raise RuntimeError(
                "Docling is not available. Install with: pip install docling"
            )
        
        return self._extract_with_docling(content, filename)
    
    async def extract_async(self, content: bytes, filename: str) -> str:
        """
        Async version of extract (runs synchronously internally).
        
        Args:
            content: File content as bytes
            filename: Original filename
            
        Returns:
            Extracted text content
        """
        # Docling operations are synchronous, so we just wrap
        return self.extract(content, filename)
    
    def _extract_text(self, content: bytes) -> str:
        """
        Extract text from plain text files.
        
        Tries multiple encodings to handle various file sources.
        
        Args:
            content: File content as bytes
            
        Returns:
            Decoded text content
        """
        # Try common encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except (UnicodeDecodeError, AttributeError):
                continue
        
        # Fallback: decode with replacement
        return content.decode('utf-8', errors='replace')
    
    def _extract_with_docling(self, content: bytes, filename: str) -> str:
        """
        Extract text using Docling.
        
        Args:
            content: File content as bytes
            filename: Original filename
            
        Returns:
            Extracted text in Markdown format
        """
        converter = self.converter
        if converter is None:
            raise RuntimeError("Docling converter not available")
        
        # Write content to temporary file
        ext = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            # Convert using Docling
            result = converter.convert(tmp_path)
            
            # Export to markdown
            markdown_text = result.document.export_to_markdown()
            
            return markdown_text
            
        except Exception as e:
            logger.error(f"Docling extraction failed for {filename}: {e}")
            raise ValueError(f"Failed to extract text with Docling: {e}")
            
        finally:
            # Clean up temp file
            try:
                tmp_path.unlink()
            except Exception:
                pass
    
    def get_metadata_from_docling(self, content: bytes, filename: str) -> dict:
        """
        Extract metadata using Docling.
        
        Returns metadata useful for future knowledge graph integration.
        
        Args:
            content: File content as bytes
            filename: Original filename
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "source_filename": filename,
            "processor": "docling",
        }
        
        if not DOCLING_AVAILABLE or self.converter is None:
            return metadata
        
        ext = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            result = self.converter.convert(tmp_path)
            
            # Extract additional metadata from Docling result
            if hasattr(result, 'document'):
                doc = result.document
                
                # Get page count if available
                if hasattr(doc, 'pages') and doc.pages:
                    metadata["page_count"] = len(doc.pages)
                
                # Get title if available
                if hasattr(doc, 'title') and doc.title:
                    metadata["title"] = doc.title
                
        except Exception as e:
            logger.debug(f"Could not extract metadata: {e}")
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass
        
        return metadata


# Singleton instance
_docling_processor: Optional[DoclingProcessor] = None


def get_docling_processor() -> DoclingProcessor:
    """Get or create the DoclingProcessor singleton."""
    global _docling_processor
    if _docling_processor is None:
        _docling_processor = DoclingProcessor()
    return _docling_processor
