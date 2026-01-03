"""PDF document processor.

Uses PyPDF2 as the primary extractor with optional Docling support
for complex documents.
"""

from typing import Optional
from app.processors.base import DocumentProcessor, ProcessorConfig
from app.processors.registry import ProcessorRegistry


@ProcessorRegistry.register
class PDFProcessor(DocumentProcessor):
    """Processor for PDF files.
    
    Uses PyPDF2 for basic text extraction. Falls back to OCR-based
    solutions if available for scanned documents.
    """
    
    name = "PDF Processor"
    supported_extensions = [".pdf"]
    supported_mimetypes = ["application/pdf"]
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize PDF processor and check for optional dependencies."""
        super().__init__(config)
        self._pypdf2_available = self._check_pypdf2()
        self._docling_available = self._check_docling()
    
    def _check_pypdf2(self) -> bool:
        """Check if PyPDF2 is available."""
        try:
            import pypdf
            return True
        except ImportError:
            try:
                import PyPDF2
                return True
            except ImportError:
                return False
    
    def _check_docling(self) -> bool:
        """Check if Docling is available."""
        try:
            from docling.document_converter import DocumentConverter
            return True
        except ImportError:
            return False
    
    def extract(self, content: bytes, filename: str) -> str:
        """Extract text from PDF files.
        
        Args:
            content: Raw bytes of the PDF file
            filename: Original filename
            
        Returns:
            Extracted text content
            
        Raises:
            ProcessorError: If PDF extraction fails
        """
        # Try PyPDF2 first (lighter weight)
        if self._pypdf2_available:
            text = self._extract_with_pypdf(content, filename)
            if text and text.strip():
                return text
        
        # Fall back to Docling for complex PDFs
        if self._docling_available:
            text = self._extract_with_docling(content, filename)
            if text and text.strip():
                return text
        
        # If we got here but have PyPDF2, return whatever we got
        if self._pypdf2_available:
            return text or ""
        
        raise ValueError(
            f"Cannot extract text from PDF '{filename}'. "
            "Please install pypdf or docling: pip install pypdf"
        )
    
    def _extract_with_pypdf(self, content: bytes, filename: str) -> Optional[str]:
        """Extract text using PyPDF2/pypdf library."""
        import io
        
        try:
            # Try modern pypdf first
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader
            
            reader = PdfReader(io.BytesIO(content))
            text_parts = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            # Log but don't fail - we might have fallback
            return None
    
    def _extract_with_docling(self, content: bytes, filename: str) -> Optional[str]:
        """Extract text using Docling for complex documents."""
        import tempfile
        import os
        
        try:
            from docling.document_converter import DocumentConverter
            
            # Docling needs a file path, so create temp file
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", 
                delete=False
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                converter = DocumentConverter()
                result = converter.convert(tmp_path)
                return result.document.export_to_markdown()
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            return None
