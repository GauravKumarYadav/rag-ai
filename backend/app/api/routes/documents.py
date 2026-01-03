import base64
import io
import tempfile
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.rag.vector_store import get_vector_store, get_client_vector_store
from app.models.client import get_client_store
from app.dependencies import get_current_user
from app.processors import ProcessorRegistry, extract_text


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
    current_user: Dict[str, Any] = Depends(get_current_user),
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
        mimetype = file.content_type
        
        # Use the processor registry to extract text
        try:
            if ProcessorRegistry.can_process(filename, mimetype):
                # Get processor with optional config
                from app.processors.base import ProcessorConfig
                config = ProcessorConfig(
                    use_ocr=use_ocr,
                    fast_mode=fast_mode,
                )
                text = extract_text(content, filename, mimetype, config)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {filename}",
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to extract text from {filename}: {str(e)}"
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
async def search_documents(
    request: DocumentSearchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
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


@router.get("", summary="List all documents")
async def list_documents(
    client_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List all uploaded documents with metadata."""
    try:
        if client_id:
            store = get_client_vector_store(client_id)
        else:
            store = get_vector_store()
        
        # Get all documents from the store
        all_docs = store.docs.get(include=["metadatas"])
        
        # Group by source file to get unique documents
        documents = {}
        for i, doc_id in enumerate(all_docs.get("ids", [])):
            metadata = all_docs.get("metadatas", [])[i] if all_docs.get("metadatas") else {}
            source = metadata.get("source", "Unknown")
            
            if source not in documents:
                documents[source] = {
                    "id": doc_id,
                    "filename": os.path.basename(source) if source != "Unknown" else "Unknown",
                    "source": source,
                    "chunk_count": 1,
                    "client_id": metadata.get("client_id"),
                    "client_name": metadata.get("client_name"),
                    "uploaded_at": metadata.get("uploaded_at", ""),
                }
            else:
                documents[source]["chunk_count"] += 1
        
        return list(documents.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.delete("/{document_id:path}", summary="Delete a document")
async def delete_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a document and all its chunks from the vector store.
    
    The document_id can be either:
    - The actual chunk ID from the vector store
    - The filename of the document
    - The full source path of the document
    """
    from urllib.parse import unquote
    document_id = unquote(document_id)  # Handle URL-encoded filenames
    
    store = get_vector_store()
    try:
        # Get all chunks
        all_docs = store.docs.get(include=["metadatas"])
        
        ids_to_delete = []
        source_to_delete = None
        
        # First, try to find by exact ID match
        for i, doc_id in enumerate(all_docs.get("ids", [])):
            if doc_id == document_id:
                metadata = all_docs.get("metadatas", [])[i] if all_docs.get("metadatas") else {}
                source_to_delete = metadata.get("source", "")
                break
        
        # If not found by ID, try to find by filename or source path
        if not source_to_delete:
            for i, doc_id in enumerate(all_docs.get("ids", [])):
                metadata = all_docs.get("metadatas", [])[i] if all_docs.get("metadatas") else {}
                source = metadata.get("source", "")
                filename = os.path.basename(source) if source else ""
                
                # Match by filename or full source path
                if filename == document_id or source == document_id or source.endswith(document_id):
                    source_to_delete = source
                    break
        
        # Now delete all chunks with the matching source
        if source_to_delete:
            for i, doc_id in enumerate(all_docs.get("ids", [])):
                metadata = all_docs.get("metadatas", [])[i] if all_docs.get("metadatas") else {}
                if metadata.get("source") == source_to_delete:
                    ids_to_delete.append(doc_id)
        
        if ids_to_delete:
            store.docs.delete(ids=ids_to_delete)
            return {"message": f"Deleted {len(ids_to_delete)} chunks", "deleted_count": len(ids_to_delete)}
        else:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.delete("/clear", summary="Clear all documents")
async def clear_documents(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
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


@router.get("/formats", summary="List supported file formats")
async def list_supported_formats():
    """List all supported file formats for document upload.
    
    Returns information about each registered document processor,
    including the file extensions and MIME types it supports.
    """
    extensions = ProcessorRegistry.list_processors()
    mimetypes = ProcessorRegistry.list_mimetypes()
    
    return {
        "supported_extensions": extensions,
        "supported_mimetypes": mimetypes,
        "processors": [
            {
                "name": processor_class.name,
                "extensions": processor_class.supported_extensions,
                "mimetypes": processor_class.supported_mimetypes,
            }
            for processor_class in set(ProcessorRegistry._processors.values())
        ],
    }
