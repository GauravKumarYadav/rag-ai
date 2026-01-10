"""PDF document processor.

Uses Docling as the primary extractor for OCR-capable processing,
with PyPDF2 as a fallback for text-based PDFs when OCR is disabled.
"""

from typing import Optional
from app.processors.base import DocumentProcessor, ProcessorConfig
from app.processors.registry import ProcessorRegistry


@ProcessorRegistry.register
class PDFProcessor(DocumentProcessor):
    """Processor for PDF files.
    
    When use_ocr=True (default): Uses Docling for full OCR + text extraction.
    When use_ocr=False: Uses PyPDF2 for fast text-only extraction (no OCR).
    
    This ensures scanned/image-based PDFs are handled correctly with OCR,
    while text-based PDFs can be processed quickly without OCR overhead.
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
        
        Strategy based on config.use_ocr:
        - use_ocr=True (default): Use Docling first (OCR-capable), fallback to PyPDF2
        - use_ocr=False: Use PyPDF2 only (fast, text-based PDFs only)
        
        Args:
            content: Raw bytes of the PDF file
            filename: Original filename
            
        Returns:
            Extracted text content
            
        Raises:
            ProcessorError: If PDF extraction fails
        """
        use_ocr = self.config.use_ocr if self.config else True
        
        if use_ocr:
            # OCR mode: Docling first (handles scanned docs), fallback to PyPDF2
            if self._docling_available:
                text = self._extract_with_docling(content, filename)
                if text and text.strip():
                    return text
            
            # Fallback to PyPDF2 if Docling fails or isn't available
            if self._pypdf2_available:
                text = self._extract_with_pypdf(content, filename)
                if text and text.strip():
                    return text
        else:
            # Non-OCR mode: PyPDF2 only (fast path for text-based PDFs)
            if self._pypdf2_available:
                text = self._extract_with_pypdf(content, filename)
                if text and text.strip():
                    return text
            
            # If PyPDF2 not available, try Docling without OCR
            if self._docling_available:
                text = self._extract_with_docling(content, filename, enable_ocr=False)
                if text and text.strip():
                    return text
        
        raise ValueError(
            f"Cannot extract text from PDF '{filename}'. "
            "Please install pypdf or docling: pip install pypdf docling"
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
    
    def _extract_with_docling(self, content: bytes, filename: str, enable_ocr: bool = True) -> Optional[str]:
        """Extract text using Docling for complex/scanned documents.
        
        Args:
            content: Raw PDF bytes
            filename: Original filename
            enable_ocr: Whether to enable OCR (True for scanned docs, False for text-based)
        """
        import tempfile
        import os
        
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            
            # Use config settings
            config = self.config
            
            # Optimize accelerator settings for CPU
            accelerator = AcceleratorOptions(
                num_threads=config.num_threads if config else 4,
                device="cpu",  # Explicit CPU to avoid GPU detection delays
            )
            
            # Configure pipeline - disable heavy ML models for speed
            pipeline_options = PdfPipelineOptions(
                accelerator_options=accelerator,
                do_ocr=enable_ocr,  # Enable/disable OCR based on parameter
                # Disable all heavy ML features for fast CPU processing
                do_table_structure=False,
                do_picture_classification=False,
                do_picture_description=False,
                do_code_enrichment=False,
                do_formula_enrichment=False,
                generate_page_images=False,
                generate_picture_images=False,
                # Smaller batch sizes for memory efficiency
                ocr_batch_size=4,
                layout_batch_size=4,
                table_batch_size=4,
                # Try to use existing PDF text first (much faster for text-based PDFs)
                force_backend_text=True,
            )
            
            # Docling needs a file path, so create temp file
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", 
                delete=False
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                # Configure converter with optimized options
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    }
                )
                result = converter.convert(tmp_path)
                return result.document.export_to_markdown()
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Docling extraction failed: {e}, trying direct OCR")
            # Fallback to direct OCR if Docling fails (e.g., can't download models)
            return self._extract_with_direct_ocr(content, filename)

    def _extract_with_direct_ocr(self, content: bytes, filename: str) -> Optional[str]:
        """Extract text using direct RapidOCR on PDF page images.
        
        This is a fallback when Docling can't download required models.
        Converts PDF pages to images and runs OCR directly.
        """
        import tempfile
        import os
        
        try:
            from pypdfium2 import PdfDocument
            from rapidocr_onnxruntime import RapidOCR
            from PIL import Image
            import io
            
            ocr = RapidOCR()
            all_text = []
            
            # Open PDF with pypdfium2
            pdf = PdfDocument(content)
            
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                # Render page to image (150 DPI for good OCR quality)
                bitmap = page.render(scale=2.0)  # 2x scale ≈ 144 DPI
                pil_image = bitmap.to_pil()
                
                # Convert to RGB if needed
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Save to bytes for OCR
                img_bytes = io.BytesIO()
                pil_image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                # Run OCR on the image
                result, _ = ocr(img_bytes.read())
                
                if result:
                    page_text = []
                    for line in result:
                        if len(line) >= 2:
                            text = line[1] if isinstance(line[1], str) else str(line[1])
                            page_text.append(text)
                    
                    if page_text:
                        all_text.append(f"## Page {page_num + 1}\n\n" + "\n".join(page_text))
            
            pdf.close()
            
            if all_text:
                return "\n\n".join(all_text)
            return None
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Direct OCR failed: {e}")
            return None
