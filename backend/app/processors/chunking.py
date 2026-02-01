"""
Document Chunking Module.

Provides configurable, deterministic text chunking with content-hash IDs.

Features:
- Character-based or token-based chunking
- Respects sentence and heading boundaries when possible
- Deterministic chunk IDs via content hashing
- Rich metadata for each chunk
"""

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from app.config import settings


@dataclass
class ChunkMetadata:
    """
    Rich metadata for a document chunk.
    
    Includes fields for future knowledge graph integration:
    - detected_entities: Named entities found in chunk
    - entity_ids: Links to KG nodes
    - semantic_type: Classification of chunk content
    - parent_chunk_id: For hierarchical chunking
    - references_chunks: Chunk IDs this references
    """
    
    # Document identifiers
    doc_id: str
    client_id: str
    source_filename: str
    
    # Position information
    chunk_index: int
    start_offset: int
    end_offset: int
    
    # Content structure (optional)
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    paragraph_index: Optional[int] = None
    
    # Embedding tracking
    embedding_fingerprint: Optional[str] = None
    
    # KG-ready fields (for future knowledge graph integration)
    detected_entities: List[Dict[str, str]] = field(default_factory=list)  # [{type: "PERSON", value: "John"}]
    entity_ids: List[str] = field(default_factory=list)  # Links to KG nodes
    semantic_type: Optional[str] = None  # fact, definition, procedure, example
    
    # Relationships (for hierarchical chunks)
    parent_chunk_id: Optional[str] = None
    references_chunks: List[str] = field(default_factory=list)
    
    # Additional metadata
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "doc_id": self.doc_id,
            "client_id": self.client_id,
            "source_filename": self.source_filename,
            "source": self.source_filename,  # Backward compatibility
            "chunk_index": self.chunk_index,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }
        
        if self.page_number is not None:
            result["page_number"] = self.page_number
        if self.section_heading:
            result["section_heading"] = self.section_heading
        if self.paragraph_index is not None:
            result["paragraph_index"] = self.paragraph_index
        if self.embedding_fingerprint:
            result["embedding_fingerprint"] = self.embedding_fingerprint
        
        # KG-ready fields (store as JSON-serializable)
        if self.detected_entities:
            result["detected_entities"] = self.detected_entities
        if self.entity_ids:
            result["entity_ids"] = self.entity_ids
        if self.semantic_type:
            result["semantic_type"] = self.semantic_type
        if self.parent_chunk_id:
            result["parent_chunk_id"] = self.parent_chunk_id
        if self.references_chunks:
            result["references_chunks"] = self.references_chunks
        
        # Add extra metadata
        result.update(self.extra)
        
        return result


@dataclass
class Chunk:
    """A document chunk with content, ID, and metadata."""
    
    id: str
    content: str
    metadata: ChunkMetadata
    
    @property
    def text(self) -> str:
        """Alias for content."""
        return self.content


def generate_chunk_id(doc_id: str, content: str, chunk_index: int) -> str:
    """
    Generate a deterministic chunk ID from content hash.
    
    Format: {doc_id}_{chunk_index}_{content_hash}
    
    This ensures:
    - Same content always produces the same ID
    - Re-processing the same document produces identical IDs
    - Changes to content produce new IDs (triggering re-embedding)
    
    Args:
        doc_id: The parent document ID
        content: The chunk text content
        chunk_index: Position of this chunk in the document
        
    Returns:
        Deterministic chunk ID string
    """
    # Hash the content for determinism
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
    
    # Create ID that includes doc reference and position
    chunk_id = f"{doc_id}_{chunk_index}_{content_hash}"
    
    return chunk_id


def generate_doc_id(source_filename: str, client_id: str) -> str:
    """
    Generate a deterministic document ID.
    
    Args:
        source_filename: The original filename
        client_id: The client this document belongs to
        
    Returns:
        Deterministic document ID
    """
    # Combine source and client for uniqueness
    combined = f"{client_id}:{source_filename}"
    doc_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    return f"doc_{doc_hash}"


def find_sentence_boundary(text: str, position: int, direction: int = 1) -> int:
    """
    Find the nearest sentence boundary from a position.
    
    Args:
        text: The text to search
        position: Starting position
        direction: 1 for forward, -1 for backward
        
    Returns:
        Position of the nearest sentence boundary
    """
    sentence_endings = re.compile(r'[.!?]\s+')
    
    if direction > 0:
        # Look forward
        match = sentence_endings.search(text, position)
        if match:
            return match.end()
    else:
        # Look backward
        text_before = text[:position]
        matches = list(sentence_endings.finditer(text_before))
        if matches:
            return matches[-1].end()
    
    return position


def find_heading_position(text: str, start: int, end: int) -> Optional[int]:
    """
    Find a markdown heading between start and end positions.
    
    Args:
        text: The text to search
        start: Start position
        end: End position
        
    Returns:
        Position of heading if found, None otherwise
    """
    # Match markdown headings
    heading_pattern = re.compile(r'\n#{1,6}\s+')
    
    search_text = text[start:end]
    match = heading_pattern.search(search_text)
    
    if match:
        return start + match.start()
    
    return None


def extract_section_heading(text: str, position: int) -> Optional[str]:
    """
    Extract the section heading that precedes a position in the text.
    
    Args:
        text: The full text
        position: Position to look before
        
    Returns:
        The heading text if found
    """
    # Look for the most recent heading before this position
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    text_before = text[:position]
    matches = list(heading_pattern.finditer(text_before))
    
    if matches:
        last_match = matches[-1]
        return last_match.group(2).strip()
    
    return None


def chunk_text_char(
    text: str,
    max_chars: int = 1200,
    overlap: int = 200,
    respect_sentences: bool = True,
    respect_headings: bool = True,
) -> List[tuple]:
    """
    Split text into overlapping chunks by character count.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Character overlap between chunks
        respect_sentences: Try to break at sentence boundaries
        respect_headings: Try to break at markdown headings
        
    Returns:
        List of (chunk_text, start_offset, end_offset) tuples
    """
    if max_chars <= 0:
        return []
    
    chunks = []
    stride = max(max_chars - overlap, 1)
    start = 0
    length = len(text)
    
    while start < length:
        # Calculate initial end position
        end = min(start + max_chars, length)
        
        # Try to find a better break point
        if end < length:
            # First, try to break at a heading
            if respect_headings:
                # Look for heading in the last quarter of the chunk
                look_start = start + int((end - start) * 0.75)
                heading_pos = find_heading_position(text, look_start, end)
                if heading_pos:
                    end = heading_pos
            
            # If no heading found, try sentence boundary
            if respect_sentences and end == min(start + max_chars, length):
                # Look for sentence boundary in overlap region
                boundary = find_sentence_boundary(
                    text, 
                    max(end - overlap, start), 
                    direction=1
                )
                if start < boundary < end + overlap // 2:
                    end = boundary
        
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))
        
        # Move to next position
        if end >= length:
            break
        
        # Adjust start for overlap
        start = end - overlap if end - overlap > start else end
    
    return chunks


def chunk_document(
    text: str,
    doc_id: str,
    client_id: str,
    source_filename: str,
    embedding_fingerprint: Optional[str] = None,
    page_numbers: Optional[Dict[int, int]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """
    Chunk a document with full metadata and content-hash IDs.
    
    Args:
        text: The document text to chunk
        doc_id: Unique document identifier
        client_id: Client this document belongs to
        source_filename: Original filename
        embedding_fingerprint: Current embedding model fingerprint
        page_numbers: Optional mapping of character offsets to page numbers
        extra_metadata: Additional metadata to include in all chunks
        
    Returns:
        List of Chunk objects with deterministic IDs and rich metadata
    """
    # Get chunking config from settings
    if settings.rag.chunk_method == "token":
        # Token-based chunking (requires tokenizer)
        # Fall back to character-based with estimated token ratio
        estimated_chars = settings.rag.chunk_token_size * 4  # ~4 chars per token
        estimated_overlap = settings.rag.chunk_token_overlap * 4
        max_chars = estimated_chars
        overlap = estimated_overlap
    else:
        max_chars = settings.rag.chunk_size
        overlap = settings.rag.chunk_overlap
    
    # Perform chunking
    raw_chunks = chunk_text_char(
        text=text,
        max_chars=max_chars,
        overlap=overlap,
        respect_sentences=settings.rag.respect_sentences,
        respect_headings=settings.rag.respect_headings,
    )
    
    chunks = []
    for idx, (chunk_text, start_offset, end_offset) in enumerate(raw_chunks):
        # Generate deterministic chunk ID
        if settings.rag.use_content_hash_ids:
            chunk_id = generate_chunk_id(doc_id, chunk_text, idx)
        else:
            chunk_id = f"{source_filename}#chunk-{idx}"
        
        # Determine page number if available
        page_number = None
        if page_numbers:
            # Find the page that contains the start of this chunk
            for offset, page in sorted(page_numbers.items()):
                if offset <= start_offset:
                    page_number = page
                else:
                    break
        
        # Extract section heading
        section_heading = extract_section_heading(text, start_offset)
        
        # Build metadata
        metadata = ChunkMetadata(
            doc_id=doc_id,
            client_id=client_id,
            source_filename=source_filename,
            chunk_index=idx,
            start_offset=start_offset,
            end_offset=end_offset,
            page_number=page_number,
            section_heading=section_heading,
            embedding_fingerprint=embedding_fingerprint,
            extra=extra_metadata or {},
        )
        
        chunks.append(Chunk(
            id=chunk_id,
            content=chunk_text,
            metadata=metadata,
        ))
    
    return chunks


def chunk_text_simple(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple text chunking for backward compatibility.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Character overlap between chunks
        
    Returns:
        List of chunk strings
    """
    raw_chunks = chunk_text_char(
        text=text,
        max_chars=max_chars,
        overlap=overlap,
        respect_sentences=True,
        respect_headings=True,
    )
    
    return [chunk_text for chunk_text, _, _ in raw_chunks]
