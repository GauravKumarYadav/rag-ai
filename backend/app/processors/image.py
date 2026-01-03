"""Image document processor.

Uses OCR to extract text from images (PNG, JPG, etc.).
"""

from typing import Optional
from app.processors.base import DocumentProcessor, ProcessorConfig
from app.processors.registry import ProcessorRegistry


@ProcessorRegistry.register
class ImageProcessor(DocumentProcessor):
    """Processor for image files with OCR.
    
    Uses pytesseract for OCR text extraction from images.
    Requires Tesseract OCR to be installed on the system.
    """
    
    name = "Image OCR Processor"
    supported_extensions = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"]
    supported_mimetypes = [
        "image/png",
        "image/jpeg",
        "image/tiff",
        "image/bmp",
        "image/gif",
        "image/webp",
    ]
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize image processor and check dependencies."""
        super().__init__(config)
        self._tesseract_available = self._check_tesseract()
        self._docling_available = self._check_docling()
    
    def _check_tesseract(self) -> bool:
        """Check if pytesseract and PIL are available."""
        try:
            import pytesseract
            from PIL import Image
            return True
        except ImportError:
            return False
    
    def _check_docling(self) -> bool:
        """Check if Docling is available for OCR."""
        try:
            from docling.document_converter import DocumentConverter
            return True
        except ImportError:
            return False
    
    def extract(self, content: bytes, filename: str) -> str:
        """Extract text from images using OCR.
        
        Args:
            content: Raw bytes of the image
            filename: Original filename
            
        Returns:
            Extracted text content
            
        Raises:
            ProcessorError: If OCR fails or libraries not installed
        """
        # Try pytesseract first (most common)
        if self._tesseract_available:
            try:
                return self._extract_with_tesseract(content, filename)
            except Exception as e:
                # Fall through to try docling
                pass
        
        # Try Docling as fallback
        if self._docling_available:
            try:
                return self._extract_with_docling(content, filename)
            except Exception:
                pass
        
        if not self._tesseract_available:
            raise ValueError(
                f"Cannot process image '{filename}'. "
                "Please install pytesseract and Tesseract OCR: "
                "pip install pytesseract pillow"
            )
        
        raise ValueError(f"OCR extraction failed for '{filename}'")
    
    def _extract_with_tesseract(self, content: bytes, filename: str) -> str:
        """Extract text using Tesseract OCR."""
        import io
        import pytesseract
        from PIL import Image
        
        image = Image.open(io.BytesIO(content))
        
        # Convert to RGB if necessary (for PNG with alpha channel)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        text = pytesseract.image_to_string(image)
        return text.strip()
    
    def _extract_with_docling(self, content: bytes, filename: str) -> str:
        """Extract text using Docling OCR."""
        import tempfile
        import os
        
        from docling.document_converter import DocumentConverter
        
        # Get the extension from filename
        ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".png"
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            converter = DocumentConverter()
            result = converter.convert(tmp_path)
            return result.document.export_to_markdown()
        finally:
            os.unlink(tmp_path)
