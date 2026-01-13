"""
Context Compression for Small Model RAG Optimization.

This module implements a hybrid compression strategy:
1. Extractive: Extract key sentences using TF-IDF and keyword overlap
2. LLM Refinement: Convert to atomic bullet facts with citations

The goal is to transform verbose retrieved chunks into dense, actionable
bullet points that small models can process effectively.

Output format:
- Policy X requires Y within 30 days. [doc3#c12]
- Exception: Z allowed if A is true. [doc3#c15]
"""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Dict, Any
from collections import Counter

from app.models.schemas import RetrievalHit

logger = logging.getLogger(__name__)


@dataclass
class CompressedFact:
    """A single compressed fact with citation."""
    text: str           # The fact as a bullet point
    source_id: str      # Document/chunk ID for citation
    source_name: str    # Human-readable source name
    score: float        # Relevance/confidence score
    metadata: Dict[str, Any]
    
    def to_citation_string(self) -> str:
        """Format as bullet with citation."""
        return f"- {self.text} [{self.source_name}]"


class ExtractiveCompressor:
    """
    Extractive compression using keyword overlap and sentence scoring.
    
    This is fast and doesn't require LLM calls.
    """
    
    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max_sentences_per_chunk
    
    def extract_key_sentences(
        self,
        hits: List[RetrievalHit],
        query: str,
    ) -> List[tuple[str, RetrievalHit]]:
        """
        Extract key sentences from retrieved chunks based on query relevance.
        
        Returns list of (sentence, source_hit) tuples.
        """
        query_words = self._tokenize(query)
        query_word_set = set(query_words)
        
        # Weight query terms by frequency (simple TF)
        query_tf = Counter(query_words)
        
        all_sentences = []
        
        for hit in hits:
            sentences = self._split_sentences(hit.content)
            
            for sentence in sentences:
                # Skip very short sentences
                if len(sentence.split()) < 5:
                    continue
                
                # Score sentence by query word overlap
                sent_words = set(self._tokenize(sentence))
                overlap = sent_words & query_word_set
                
                if not overlap:
                    continue
                
                # Weighted score by query term frequency
                score = sum(query_tf[w] for w in overlap) / len(query_tf) if query_tf else 0
                
                # Boost sentences with numbers (often important facts)
                if re.search(r'\d+', sentence):
                    score *= 1.2
                
                # Boost sentences with key phrases
                if any(phrase in sentence.lower() for phrase in 
                       ['must', 'required', 'important', 'exception', 'note that', 'however']):
                    score *= 1.1
                
                all_sentences.append((sentence, hit, score))
        
        # Sort by score and take top sentences
        all_sentences.sort(key=lambda x: x[2], reverse=True)
        
        # Limit sentences per source to avoid redundancy
        source_counts: Dict[str, int] = {}
        result = []
        
        for sentence, hit, score in all_sentences:
            source_id = hit.id
            if source_counts.get(source_id, 0) >= self.max_sentences_per_chunk:
                continue
            
            result.append((sentence, hit))
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
        
        return result
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split, remove punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'as', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'between', 'under', 'again', 'further',
                      'then', 'once', 'here', 'there', 'when', 'where', 'why',
                      'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                      'than', 'too', 'very', 'just', 'and', 'but', 'or', 'if',
                      'because', 'until', 'while', 'this', 'that', 'these', 'those',
                      'it', 'its', 'i', 'you', 'he', 'she', 'they', 'we', 'what',
                      'which', 'who', 'whom'}
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class LLMCompressor:
    """
    LLM-based compression to convert sentences to atomic bullet facts.
    
    Uses the same small model to refine extracted sentences into
    concise, standalone facts.
    """
    
    COMPRESSION_PROMPT = """Convert these sentences into concise bullet-point facts.
Each bullet should:
- Be a single, atomic fact (one piece of information)
- Include specific numbers, names, or values verbatim
- Be understandable without additional context

Sentences to compress:
{sentences}

Output format (one fact per line):
- [fact 1]
- [fact 2]
..."""
    
    def __init__(self, lm_client: Optional[object] = None) -> None:
        self.lm_client = lm_client
    
    async def refine_to_bullets(
        self,
        sentences_with_sources: List[tuple[str, RetrievalHit]],
        max_bullets: int = 15,
    ) -> List[CompressedFact]:
        """
        Refine extracted sentences to atomic bullet facts.
        
        If LLM client is not available, falls back to simple formatting.
        """
        if not sentences_with_sources:
            return []
        
        # Group sentences for LLM processing
        sentences_text = "\n".join(f"- {sent}" for sent, _ in sentences_with_sources[:max_bullets])
        
        if self.lm_client is None:
            # Fallback: use sentences as-is
            return self._simple_format(sentences_with_sources[:max_bullets])
        
        try:
            # Use LLM to compress
            prompt = self.COMPRESSION_PROMPT.format(sentences=sentences_text)
            messages = [{"role": "user", "content": prompt}]
            
            response = await self.lm_client.chat(messages, stream=False)
            
            # Parse LLM output
            bullets = self._parse_bullets(response)
            
            # Match bullets back to sources (best effort)
            return self._match_to_sources(bullets, sentences_with_sources)
            
        except Exception as e:
            logger.warning(f"LLM compression failed, using fallback: {e}")
            return self._simple_format(sentences_with_sources[:max_bullets])
    
    def _simple_format(
        self,
        sentences_with_sources: List[tuple[str, RetrievalHit]],
    ) -> List[CompressedFact]:
        """Simple formatting without LLM refinement."""
        facts = []
        for sentence, hit in sentences_with_sources:
            # Clean up sentence
            sentence = sentence.strip()
            if not sentence.endswith('.'):
                sentence += '.'
            
            source_name = hit.metadata.get('source', hit.id)
            if '/' in source_name:
                source_name = source_name.split('/')[-1]
            
            facts.append(CompressedFact(
                text=sentence,
                source_id=hit.id,
                source_name=source_name,
                score=hit.score,
                metadata=hit.metadata,
            ))
        
        return facts
    
    def _parse_bullets(self, response: str) -> List[str]:
        """Parse bullet points from LLM response."""
        bullets = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                bullets.append(line[2:].strip())
            elif line.startswith('• '):
                bullets.append(line[2:].strip())
            elif line.startswith('* '):
                bullets.append(line[2:].strip())
        return bullets
    
    def _match_to_sources(
        self,
        bullets: List[str],
        sentences_with_sources: List[tuple[str, RetrievalHit]],
    ) -> List[CompressedFact]:
        """Match compressed bullets back to their source documents."""
        facts = []
        
        # Simple matching: use index correspondence
        for i, bullet in enumerate(bullets):
            if i < len(sentences_with_sources):
                _, hit = sentences_with_sources[i]
            else:
                # Use last source as fallback
                _, hit = sentences_with_sources[-1]
            
            source_name = hit.metadata.get('source', hit.id)
            if '/' in source_name:
                source_name = source_name.split('/')[-1]
            
            facts.append(CompressedFact(
                text=bullet,
                source_id=hit.id,
                source_name=source_name,
                score=hit.score,
                metadata=hit.metadata,
            ))
        
        return facts


class ContextCompressor:
    """
    Main context compressor combining extractive and LLM compression.
    
    Usage:
        compressor = ContextCompressor()
        facts = await compressor.compress(hits, query)
        for fact in facts:
            print(fact.to_citation_string())
    """
    
    def __init__(
        self,
        lm_client: Optional[object] = None,
        max_sentences_per_chunk: int = 3,
    ) -> None:
        self.extractive = ExtractiveCompressor(max_sentences_per_chunk)
        self.llm_compressor = LLMCompressor(lm_client)
        self.lm_client = lm_client
    
    async def compress(
        self,
        hits: List[RetrievalHit],
        query: str,
        use_llm_refinement: bool = True,
        max_facts: int = 15,
    ) -> List[CompressedFact]:
        """
        Compress retrieved chunks into dense bullet facts.
        
        Pipeline:
        1. Extract key sentences (extractive)
        2. Optionally refine with LLM (if enabled and available)
        3. Return facts with citations
        
        Args:
            hits: Retrieved chunks to compress
            query: The search query for relevance scoring
            use_llm_refinement: Whether to use LLM for final refinement
            max_facts: Maximum number of facts to return
            
        Returns:
            List of CompressedFact with citations
        """
        if not hits:
            return []
        
        logger.debug(f"Compressing {len(hits)} hits for query: {query[:50]}...")
        
        # Stage 1: Extract key sentences
        sentences = self.extractive.extract_key_sentences(hits, query)
        logger.debug(f"Extracted {len(sentences)} key sentences")
        
        if not sentences:
            # Fallback: use chunk content directly
            return self._fallback_compress(hits, max_facts)
        
        # Stage 2: LLM refinement (optional)
        if use_llm_refinement and self.lm_client:
            facts = await self.llm_compressor.refine_to_bullets(
                sentences[:max_facts],
                max_bullets=max_facts
            )
        else:
            facts = self.llm_compressor._simple_format(sentences[:max_facts])
        
        logger.debug(f"Compressed to {len(facts)} facts")
        return facts
    
    def _fallback_compress(
        self,
        hits: List[RetrievalHit],
        max_facts: int,
    ) -> List[CompressedFact]:
        """Fallback when extractive compression fails."""
        facts = []
        for hit in hits[:max_facts]:
            # Take first sentence of each chunk
            content = hit.content
            first_sentence = content.split('.')[0] + '.' if '.' in content else content[:200]
            
            source_name = hit.metadata.get('source', hit.id)
            if '/' in source_name:
                source_name = source_name.split('/')[-1]
            
            facts.append(CompressedFact(
                text=first_sentence.strip(),
                source_id=hit.id,
                source_name=source_name,
                score=hit.score,
                metadata=hit.metadata,
            ))
        
        return facts
    
    def format_facts_for_prompt(self, facts: List[CompressedFact]) -> str:
        """
        Format compressed facts for inclusion in prompt.
        
        Returns formatted string like:
        Retrieved context:
        - Policy X requires Y. [policy.pdf]
        - Exception Z applies. [guidelines.pdf]
        """
        if not facts:
            return ""
        
        lines = ["Retrieved context:"]
        for fact in facts:
            lines.append(fact.to_citation_string())
        
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_context_compressor() -> ContextCompressor:
    """Get singleton context compressor instance (without LLM client)."""
    return ContextCompressor()


def get_context_compressor_with_llm(lm_client: object) -> ContextCompressor:
    """Get context compressor with LLM client for refinement."""
    return ContextCompressor(lm_client=lm_client)
