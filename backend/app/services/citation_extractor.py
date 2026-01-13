"""
Citation Extractor - Extract and validate citations from LLM responses.

This module ensures the LLM cites its sources when making factual claims,
which is critical for:
- Catching hallucinations
- Providing verifiable answers
- Building user trust

Citation format: [Source: filename#chunk-N] or [Source: doc_id]

Enhanced for chunk_id validation:
- Citations are validated against actual chunk_ids (not just source filenames)
- Coverage metrics use chunk_id mapping from CompressedFacts
- Dangling citations (referencing non-existent chunks) are detected
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, Any

from app.models.schemas import RetrievalHit

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A citation extracted from a response."""
    source_id: str            # The cited source (e.g., "contract.pdf#chunk-2")
    text_before: str          # Text immediately before the citation
    start_pos: int            # Start position in response
    end_pos: int              # End position in response
    sentence: str             # Full sentence containing the citation
    
    # Enhanced chunk_id validation
    matched_chunk_ids: List[str] = field(default_factory=list)  # Chunk IDs this citation matches
    is_valid: bool = False    # Whether citation matches known chunks


@dataclass
class CitationAnalysis:
    """Analysis of citations in a response."""
    citations: List[Citation]
    citation_count: int
    unique_sources: Set[str]
    uncited_factual_claims: List[str]
    coverage_ratio: float     # Ratio of cited claims to total claims
    valid_citations: List[Citation]      # Citations that match retrieved docs
    invalid_citations: List[Citation]    # Citations to non-existent sources
    
    # Enhanced chunk_id metrics
    matched_chunk_ids: Set[str] = field(default_factory=set)  # All chunks successfully cited
    unmatched_chunk_ids: Set[str] = field(default_factory=set)  # Provided chunks not cited
    dangling_citations: List[Citation] = field(default_factory=list)  # Citations to non-existent chunks
    chunk_coverage_ratio: float = 0.0  # Ratio of cited chunks to provided chunks


# Patterns that indicate factual claims
FACTUAL_PATTERNS = [
    r'\$[\d,]+',                    # Dollar amounts
    r'\d+%',                        # Percentages
    r'\d{1,2}/\d{1,2}/\d{2,4}',    # Dates (MM/DD/YYYY)
    r'\d{4}-\d{2}-\d{2}',          # Dates (YYYY-MM-DD)
    r'(?:is|are|was|were|has|have|had)\s+\w+',  # Is/are statements
    r'according to',                # Attribution phrases
    r'states? that',
    r'mentions? that',
    r'shows? that',
    r'indicates? that',
    r'reports? that',
    r'\d+\s*(?:years?|months?|days?|hours?|minutes?)',  # Time durations
    r'(?:total|sum|amount|value|cost|price)\s+(?:of|is)\s+',
]

# Patterns for non-factual/meta statements (shouldn't require citation)
NON_FACTUAL_PATTERNS = [
    r'^(I |Based on|According to the documents?|The documents? (?:show|indicate|mention))',
    r'^(Let me|I\'ll|I can|I don\'t|Unfortunately)',
    r'^(Yes|No|Sure|Of course|Certainly)',
    r'^(However|Therefore|Thus|Hence|So)',
    r'(?:would you like|do you want|shall I|can I)',
    r'I don\'t have (?:information|enough|any)',
]


class CitationExtractor:
    """
    Extract and analyze citations from LLM responses.
    
    Supports multiple citation formats:
    - [Source: filename#chunk-N]
    - [Source: doc_id]
    - [filename#chunk-N]
    - [doc_id]
    """
    
    # Citation patterns (ordered by specificity)
    CITATION_PATTERNS = [
        r'\[Source:\s*([^\]]+)\]',           # [Source: id]
        r'\[Ref:\s*([^\]]+)\]',              # [Ref: id]
        r'\[Citation:\s*([^\]]+)\]',         # [Citation: id]
        r'\[([^\]]+\.(?:pdf|docx?|txt|md|csv)(?:#chunk-\d+)?)\]',  # [filename.ext#chunk-N]
        r'\[([a-f0-9-]{36}(?:#chunk-\d+)?)\]',  # [uuid#chunk-N]
    ]
    
    def __init__(self, min_coverage: float = 0.7):
        """
        Initialize extractor.
        
        Args:
            min_coverage: Minimum ratio of cited claims for a response to pass
        """
        self.min_coverage = min_coverage
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.CITATION_PATTERNS]
        self._factual_patterns = [re.compile(p, re.IGNORECASE) for p in FACTUAL_PATTERNS]
        self._non_factual_patterns = [re.compile(p, re.IGNORECASE) for p in NON_FACTUAL_PATTERNS]
    
    def extract_citations(self, response: str) -> List[Citation]:
        """
        Extract all citations from a response.
        
        Args:
            response: The LLM response text
            
        Returns:
            List of Citation objects
        """
        citations = []
        found_positions: Set[Tuple[int, int]] = set()
        
        for pattern in self._compiled_patterns:
            for match in pattern.finditer(response):
                # Avoid duplicates from overlapping patterns
                pos = (match.start(), match.end())
                if pos in found_positions:
                    continue
                found_positions.add(pos)
                
                # Extract context
                text_before = response[max(0, match.start() - 100):match.start()].strip()
                sentence = self._get_containing_sentence(response, match.start())
                
                citations.append(Citation(
                    source_id=match.group(1).strip(),
                    text_before=text_before,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    sentence=sentence,
                ))
        
        return citations
    
    def _get_containing_sentence(self, text: str, position: int) -> str:
        """Get the sentence containing a given position."""
        # Find sentence boundaries
        sentence_endings = re.compile(r'[.!?]\s+')
        
        # Find start of sentence
        start = 0
        for match in sentence_endings.finditer(text[:position]):
            start = match.end()
        
        # Find end of sentence
        end = len(text)
        match = sentence_endings.search(text, position)
        if match:
            end = match.start() + 1
        
        return text[start:end].strip()
    
    def _is_factual_sentence(self, sentence: str) -> bool:
        """
        Determine if a sentence contains factual claims.
        
        Args:
            sentence: The sentence to analyze
            
        Returns:
            True if sentence appears to contain factual claims
        """
        sentence_clean = sentence.strip()
        
        # Skip meta/conversational sentences
        for pattern in self._non_factual_patterns:
            if pattern.search(sentence_clean):
                return False
        
        # Check for factual indicators
        for pattern in self._factual_patterns:
            if pattern.search(sentence_clean):
                return True
        
        # Short sentences without factual patterns are likely conversational
        if len(sentence_clean.split()) < 5:
            return False
        
        return False  # Default to not requiring citation
    
    def _has_citation(self, sentence: str) -> bool:
        """Check if a sentence contains a citation."""
        for pattern in self._compiled_patterns:
            if pattern.search(sentence):
                return True
        return False
    
    def find_uncited_claims(self, response: str) -> List[str]:
        """
        Find sentences that appear to make factual claims without citations.
        
        Args:
            response: The LLM response text
            
        Returns:
            List of sentences that may need citations
        """
        uncited = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        
        for sentence in sentences:
            if self._is_factual_sentence(sentence) and not self._has_citation(sentence):
                uncited.append(sentence.strip())
        
        return uncited
    
    def analyze(
        self,
        response: str,
        retrieved_docs: Optional[List[RetrievalHit]] = None,
        known_chunk_ids: Optional[Set[str]] = None,
    ) -> CitationAnalysis:
        """
        Perform full citation analysis on a response.
        
        Args:
            response: The LLM response text
            retrieved_docs: Optional list of retrieved documents to validate against
            known_chunk_ids: Optional set of valid chunk IDs for validation
            
        Returns:
            CitationAnalysis with detailed results
        """
        from app.core.metrics import (
            record_citations,
            record_citation_coverage,
            record_ungrounded_claim,
        )
        
        citations = self.extract_citations(response)
        uncited = self.find_uncited_claims(response)
        
        # Count unique sources
        unique_sources = {c.source_id for c in citations}
        
        # Calculate coverage
        total_claims = len(citations) + len(uncited)
        coverage = len(citations) / total_claims if total_claims > 0 else 1.0
        
        # Validate citations against retrieved docs and chunk_ids
        valid_citations = []
        invalid_citations = []
        dangling_citations = []
        matched_chunk_ids: Set[str] = set()
        
        # Build validation sets
        retrieved_ids: Set[str] = set()
        retrieved_sources: Set[str] = set()
        source_to_chunk: Dict[str, List[str]] = {}  # Map source names to chunk IDs
        
        if retrieved_docs:
            for doc in retrieved_docs:
                retrieved_ids.add(doc.id)
                source = doc.metadata.get("source_filename") or doc.metadata.get("source", "")
                if source:
                    retrieved_sources.add(source)
                    if source not in source_to_chunk:
                        source_to_chunk[source] = []
                    source_to_chunk[source].append(doc.id)
        
        if known_chunk_ids:
            retrieved_ids.update(known_chunk_ids)
        
        for citation in citations:
            citation_matched = False
            matched_ids = []
            
            # Check exact chunk ID match
            if citation.source_id in retrieved_ids:
                citation.is_valid = True
                citation.matched_chunk_ids = [citation.source_id]
                valid_citations.append(citation)
                matched_chunk_ids.add(citation.source_id)
                citation_matched = True
            else:
                # Check source file match (and record which chunks it maps to)
                for source in retrieved_sources:
                    if citation.source_id in source or source in citation.source_id:
                        citation.is_valid = True
                        citation.matched_chunk_ids = source_to_chunk.get(source, [])
                        valid_citations.append(citation)
                        matched_chunk_ids.update(citation.matched_chunk_ids)
                        citation_matched = True
                        break
            
            if not citation_matched:
                citation.is_valid = False
                invalid_citations.append(citation)
                dangling_citations.append(citation)
        
        # Calculate chunk coverage (what % of provided chunks are cited)
        all_chunk_ids = retrieved_ids
        unmatched_chunk_ids = all_chunk_ids - matched_chunk_ids
        chunk_coverage = len(matched_chunk_ids) / len(all_chunk_ids) if all_chunk_ids else 1.0
        
        # Record metrics
        record_citations(len(citations))
        record_citation_coverage(coverage)
        for _ in uncited:
            record_ungrounded_claim()
        
        return CitationAnalysis(
            citations=citations,
            citation_count=len(citations),
            unique_sources=unique_sources,
            uncited_factual_claims=uncited,
            coverage_ratio=coverage,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            matched_chunk_ids=matched_chunk_ids,
            unmatched_chunk_ids=unmatched_chunk_ids,
            dangling_citations=dangling_citations,
            chunk_coverage_ratio=chunk_coverage,
        )
    
    def passes_coverage_threshold(
        self,
        response: str,
        retrieved_docs: Optional[List[RetrievalHit]] = None,
        known_chunk_ids: Optional[Set[str]] = None,
    ) -> Tuple[bool, CitationAnalysis]:
        """
        Check if a response meets the minimum citation coverage.
        
        Args:
            response: The LLM response text
            retrieved_docs: Optional retrieved documents for validation
            known_chunk_ids: Optional set of valid chunk IDs
            
        Returns:
            Tuple of (passes_threshold, analysis)
        """
        analysis = self.analyze(response, retrieved_docs, known_chunk_ids)
        passes = analysis.coverage_ratio >= self.min_coverage
        
        if not passes:
            logger.warning(
                f"Citation coverage {analysis.coverage_ratio:.2%} below threshold "
                f"{self.min_coverage:.2%}. Uncited claims: {len(analysis.uncited_factual_claims)}"
            )
        
        # Also check for dangling citations
        if analysis.dangling_citations:
            logger.warning(
                f"Found {len(analysis.dangling_citations)} dangling citations "
                f"(references to non-existent chunks)"
            )
        
        return passes, analysis
    
    def validate_with_compressed_facts(
        self,
        response: str,
        compressed_facts: List[Any],  # List[CompressedFact]
    ) -> CitationAnalysis:
        """
        Validate citations against CompressedFacts with chunk_id tracking.
        
        This is the preferred validation method when using the context compressor,
        as it ensures citations map to actual chunk_ids.
        
        Args:
            response: The LLM response text
            compressed_facts: List of CompressedFact objects
            
        Returns:
            CitationAnalysis with chunk_id-based validation
        """
        # Extract all valid chunk_ids from compressed facts
        known_chunk_ids: Set[str] = set()
        for fact in compressed_facts:
            if hasattr(fact, 'chunk_ids'):
                known_chunk_ids.update(fact.chunk_ids)
            elif hasattr(fact, 'source_id'):
                known_chunk_ids.add(fact.source_id)
        
        # Also build source name mapping
        source_to_chunks: Dict[str, List[str]] = {}
        for fact in compressed_facts:
            source_name = getattr(fact, 'source_name', '')
            chunk_ids = getattr(fact, 'chunk_ids', [getattr(fact, 'source_id', '')])
            if source_name:
                if source_name not in source_to_chunks:
                    source_to_chunks[source_name] = []
                source_to_chunks[source_name].extend(chunk_ids)
        
        return self.analyze(response, None, known_chunk_ids)
    
    def format_coverage_warning(self, analysis: CitationAnalysis) -> str:
        """
        Format a warning message for insufficient citation coverage.
        
        Args:
            analysis: CitationAnalysis result
            
        Returns:
            Warning message string
        """
        warnings = []
        
        if analysis.dangling_citations:
            dangling_sources = [c.source_id for c in analysis.dangling_citations]
            warnings.append(
                f"⚠️ Some citations reference chunks not in the provided context: "
                f"{', '.join(dangling_sources[:3])}"
            )
        elif analysis.invalid_citations:
            invalid_sources = [c.source_id for c in analysis.invalid_citations]
            warnings.append(
                f"⚠️ Some citations reference documents not in the provided context: "
                f"{', '.join(invalid_sources[:3])}"
            )
        
        if analysis.uncited_factual_claims and analysis.coverage_ratio < self.min_coverage:
            warnings.append(
                f"⚠️ Some statements could not be verified against source documents."
            )
        
        return "\n".join(warnings) if warnings else ""


# Singleton instance
_citation_extractor: Optional[CitationExtractor] = None


def get_citation_extractor() -> CitationExtractor:
    """Get the singleton citation extractor."""
    global _citation_extractor
    
    if _citation_extractor is None:
        from app.config import settings
        _citation_extractor = CitationExtractor(
            min_coverage=settings.rag.min_citation_coverage
        )
    
    return _citation_extractor
