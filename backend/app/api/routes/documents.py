import base64
import io
import tempfile
import os
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.rag.vector_store import get_vector_store, get_client_vector_store
from app.models.client import get_client_store


router = APIRouter()


class DocumentUploadResponse(BaseModel):
    message: str
    document_count: int
    chunk_count: int


class DocumentSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    metadata_filters: Optional[dict] = None
    client_id: Optional[str] = None


class DocumentSearchResponse(BaseModel):
    results: List[dict]


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if max_chars <= 0:
        return []
    stride = max(max_chars - overlap, 1)
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == length:
            break
        start += stride
    return [c for c in chunks if c]


def extract_text_with_docling(file_path: str, fast_mode: bool = False) -> str:
    """
    Extract text from document using Docling (supports PDF, DOCX, images with OCR).
    
    Args:
        file_path: Path to the document
        fast_mode: If True, optimize for speed over accuracy
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        
        # Optimize accelerator settings
        accelerator = AcceleratorOptions(
            num_threads=8,  # Increase parallelism
            device="auto",  # Will use MPS on Mac, CUDA on NVIDIA, CPU otherwise
        )
        
        # Configure pipeline for speed
        pipeline_options = PdfPipelineOptions(
            accelerator_options=accelerator,
            do_ocr=True,
            do_table_structure=not fast_mode,  # Skip table detection in fast mode
            do_picture_classification=False,   # Skip picture classification
            do_picture_description=False,      # Skip picture description
            do_code_enrichment=False,          # Skip code detection
            do_formula_enrichment=False,       # Skip formula detection
            generate_page_images=False,        # Don't generate images
            generate_picture_images=False,
            # Use larger batch sizes for speed
            ocr_batch_size=8 if not fast_mode else 16,
            layout_batch_size=8 if not fast_mode else 16,
            table_batch_size=8 if not fast_mode else 16,
            # Try to use existing PDF text first (much faster for text-based PDFs)
            force_backend_text=True,
        )
        
        # Configure converter with optimized options
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    except Exception as e:
        raise ValueError(f"Failed to extract text with Docling: {e}")


def extract_text_from_pdf_pypdf(content: bytes) -> str:
    """Fallback: Extract text from PDF using pypdf (text-based PDFs only)."""
    try:
        from pypdf import PdfReader
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")


@router.post("/upload", response_model=DocumentUploadResponse, summary="Upload documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(1200),
    chunk_overlap: int = Form(200),
    use_ocr: bool = Form(True),
    fast_mode: bool = Form(False),
    client_id: Optional[str] = Form(None),
    client_name: Optional[str] = Form(None),
):
    """
    Upload documents and add them to the vector store.
    
    Supports: PDF (with OCR for scanned docs), DOCX, TXT, MD, images.
    
    Options:
    - use_ocr=True: Use Docling with OCR (slower but handles scanned docs)
    - use_ocr=False: Use pypdf for text-based PDFs only (fastest)
    - fast_mode=True: Use Docling but skip table detection (good balance)
    
    If client_id is provided, documents are stored in a client-specific collection.
    If client_name is provided without client_id, a new client is created.
    Otherwise, documents go to the global document store.
    """
    # Handle client creation/lookup
    actual_client_id = client_id
    if client_name and not client_id:
        # Create new client with this name
        client_store = get_client_store()
        # Check if client with this name exists
        existing = client_store.get_by_name(client_name)
        if existing:
            actual_client_id = existing.id
        else:
            from app.models.client import ClientCreate
            new_client = client_store.create(ClientCreate(name=client_name))
            actual_client_id = new_client.id
    
    # Get appropriate vector store
    if actual_client_id:
        store = get_client_vector_store(actual_client_id)
    else:
        store = get_vector_store()
    
    all_chunks: List[str] = []
    all_ids: List[str] = []
    all_metadatas: List[dict] = []
    doc_count = 0

    for file in files:
        content = await file.read()
        filename = file.filename or "unknown"
        
        # Determine extraction method
        is_pdf = filename.lower().endswith(".pdf")
        is_docx = filename.lower().endswith(".docx")
        is_image = filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
        
        if (is_pdf or is_docx or is_image) and use_ocr:
            # Use Docling for PDF/DOCX/images with OCR support
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                text = extract_text_with_docling(tmp_path, fast_mode=fast_mode)
                os.unlink(tmp_path)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Docling extraction failed: {e}")
        elif is_pdf and not use_ocr:
            # Use pypdf for fast text-based PDF extraction
            try:
                text = extract_text_from_pdf_pypdf(content)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            # Plain text files
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {filename} is not valid UTF-8 text or supported format",
                )

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"No text content could be extracted from {filename}",
            )

        chunks = chunk_text(text, max_chars=chunk_size, overlap=chunk_overlap)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}#chunk-{idx}")
            all_metadatas.append({"source": filename, "chunk": idx})
        doc_count += 1

    if all_chunks:
        store.add_documents(contents=all_chunks, ids=all_ids, metadatas=all_metadatas)

    return DocumentUploadResponse(
        message="Documents uploaded successfully",
        document_count=doc_count,
        chunk_count=len(all_chunks),
    )


@router.post("/search", response_model=DocumentSearchResponse, summary="Search documents")
async def search_documents(request: DocumentSearchRequest):
    """Search the vector store for relevant documents.
    
    If client_id is provided, searches in client-specific collection.
    Otherwise, searches the global document store.
    """
    if request.client_id:
        store = get_client_vector_store(request.client_id)
        hits = store.query(
            query=request.query,
            top_k=request.top_k,
            where=request.metadata_filters,
            collection="documents",
        )
    else:
        store = get_vector_store()
        hits = store.query(
            query=request.query,
            top_k=request.top_k,
            where=request.metadata_filters,
            collection="documents",
        )
    return DocumentSearchResponse(
        results=[hit.model_dump() for hit in hits]
    )


@router.get("/stats", summary="Get document statistics")
async def get_document_stats():
    """Get statistics about indexed documents."""
    store = get_vector_store()
    try:
        doc_count = store.docs.count()
        memory_count = store.memories.count()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

    return {
        "documents_indexed": doc_count,
        "memories_indexed": memory_count,
    }


@router.delete("/clear", summary="Clear all documents")
async def clear_documents():
    """Clear all documents from the vector store (use with caution)."""
    store = get_vector_store()
    try:
        # Get all document IDs and delete them
        all_docs = store.docs.get()
        doc_ids = all_docs.get("ids", [])
        if doc_ids:
            store.docs.delete(ids=doc_ids)
        return {"message": f"Cleared {len(doc_ids)} document chunks"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")
