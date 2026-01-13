"""
Token Budgeting for Small Model RAG Optimization.

Enforces a hard token cap on retrieved context to ensure
small models receive concise, manageable input.

Default budget: ~1000 tokens for retrieval context.

Features:
- Token counting (approximate, char-based)
- Priority-based fact selection
- Sentence truncation for long facts
- Budget tracking
"""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

from app.config import settings
from app.services.context_compressor import CompressedFact

logger = logging.getLogger(__name__)


@dataclass
class BudgetAllocation:
    """Result of budget allocation."""
    facts: List[CompressedFact]
    total_tokens: int
    budget: int
    utilization: float  # 0.0 to 1.0
    truncated_count: int  # Number of facts that were truncated
    dropped_count: int  # Number of facts that couldn't fit


class TokenCounter:
    """
    Approximate token counter.
    
    Uses simple heuristics since exact token counting requires
    the specific tokenizer used by the model.
    
    Rule of thumb: ~4 characters per token for English text.
    """
    
    CHARS_PER_TOKEN = 4.0
    
    def count(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses character-based approximation.
        """
        if not text:
            return 0
        
        # Basic approximation
        char_count = len(text)
        token_estimate = char_count / self.CHARS_PER_TOKEN
        
        # Adjust for whitespace and punctuation
        word_count = len(text.split())
        
        # Average of char-based and word-based estimates
        # (words ≈ tokens for English, but compound words exist)
        combined_estimate = (token_estimate + word_count * 1.3) / 2
        
        return int(combined_estimate)
    
    def count_messages(self, messages: List[dict]) -> int:
        """Count total tokens in a list of messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                # Multimodal content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += self.count(item.get("text", ""))
            # Add overhead for role and formatting
            total += 4  # Approximate overhead per message
        return total


class TokenBudgeter:
    """
    Enforces token budget on retrieved context.
    
    Ensures the final context fits within the specified token limit
    while preserving the most relevant information.
    
    Usage:
        budgeter = TokenBudgeter(max_tokens=1000)
        allocation = budgeter.fit_to_budget(facts)
        print(f"Using {allocation.total_tokens} of {allocation.budget} tokens")
    """
    
    def __init__(
        self,
        max_tokens: Optional[int] = None,
        reserve_tokens: int = 100,  # Reserve for formatting overhead
    ) -> None:
        """
        Initialize token budgeter.
        
        Args:
            max_tokens: Maximum tokens for context. If None, uses config.
            reserve_tokens: Tokens to reserve for formatting overhead.
        """
        self.max_tokens = max_tokens or settings.rag.context_token_budget
        self.reserve_tokens = reserve_tokens
        self.counter = TokenCounter()
        
        # Effective budget after reserve
        self.effective_budget = self.max_tokens - self.reserve_tokens
    
    def fit_to_budget(
        self,
        facts: List[CompressedFact],
        max_fact_tokens: int = 150,  # Max tokens per individual fact
    ) -> BudgetAllocation:
        """
        Fit facts to token budget.
        
        Strategy:
        1. Sort by score (relevance)
        2. Add facts until budget is reached
        3. Truncate oversized facts
        4. Track what was dropped
        
        Args:
            facts: List of CompressedFact to fit
            max_fact_tokens: Maximum tokens for any single fact
            
        Returns:
            BudgetAllocation with selected facts and stats
        """
        if not facts:
            return BudgetAllocation(
                facts=[],
                total_tokens=0,
                budget=self.effective_budget,
                utilization=0.0,
                truncated_count=0,
                dropped_count=0,
            )
        
        # Sort by score (higher = more relevant)
        sorted_facts = sorted(facts, key=lambda f: f.score, reverse=True)
        
        selected: List[CompressedFact] = []
        total_tokens = 0
        truncated_count = 0
        dropped_count = 0
        
        for fact in sorted_facts:
            fact_tokens = self.counter.count(fact.text)
            
            # Check if fact fits
            if total_tokens + fact_tokens <= self.effective_budget:
                # Fact fits as-is
                selected.append(fact)
                total_tokens += fact_tokens
            
            elif total_tokens < self.effective_budget:
                # Partial budget remaining - try to fit truncated version
                remaining = self.effective_budget - total_tokens
                
                if remaining >= 30:  # Minimum useful content
                    truncated = self._truncate_fact(fact, remaining, max_fact_tokens)
                    if truncated:
                        selected.append(truncated)
                        total_tokens += self.counter.count(truncated.text)
                        truncated_count += 1
                    else:
                        dropped_count += 1
                else:
                    dropped_count += 1
            
            else:
                # Budget exhausted
                dropped_count += 1
        
        utilization = total_tokens / self.effective_budget if self.effective_budget > 0 else 0.0
        
        logger.debug(
            f"Budget allocation: {total_tokens}/{self.effective_budget} tokens, "
            f"{len(selected)} facts selected, {truncated_count} truncated, {dropped_count} dropped"
        )
        
        return BudgetAllocation(
            facts=selected,
            total_tokens=total_tokens,
            budget=self.effective_budget,
            utilization=min(1.0, utilization),
            truncated_count=truncated_count,
            dropped_count=dropped_count,
        )
    
    def _truncate_fact(
        self,
        fact: CompressedFact,
        target_tokens: int,
        max_tokens: int,
    ) -> Optional[CompressedFact]:
        """
        Truncate a fact to fit within token limit.
        
        Tries to preserve complete sentences where possible.
        """
        text = fact.text
        
        # Don't truncate if already small enough
        current_tokens = self.counter.count(text)
        if current_tokens <= target_tokens:
            return fact
        
        # Calculate target character length
        target_chars = int(target_tokens * TokenCounter.CHARS_PER_TOKEN)
        
        if target_chars < 20:
            # Too short to be useful
            return None
        
        # Try to break at sentence boundary
        truncated = self._truncate_to_sentence(text, target_chars)
        
        if not truncated or len(truncated) < 20:
            # Fall back to word boundary
            truncated = self._truncate_to_word(text, target_chars)
        
        if not truncated:
            return None
        
        # Add ellipsis to indicate truncation
        if not truncated.endswith('...'):
            truncated = truncated.rstrip('.') + '...'
        
        return CompressedFact(
            text=truncated,
            source_id=fact.source_id,
            source_name=fact.source_name,
            score=fact.score,
            metadata={**fact.metadata, "truncated": True},
        )
    
    def _truncate_to_sentence(self, text: str, max_chars: int) -> Optional[str]:
        """Truncate text at sentence boundary."""
        if len(text) <= max_chars:
            return text
        
        # Find sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        result = ""
        for sentence in sentences:
            if len(result) + len(sentence) + 1 <= max_chars:
                result = result + " " + sentence if result else sentence
            else:
                break
        
        return result.strip() if result else None
    
    def _truncate_to_word(self, text: str, max_chars: int) -> Optional[str]:
        """Truncate text at word boundary."""
        if len(text) <= max_chars:
            return text
        
        # Find last space before max_chars
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        
        if last_space > max_chars // 2:  # At least half the target
            return truncated[:last_space]
        
        return truncated
    
    def estimate_remaining_budget(
        self,
        system_tokens: int,
        history_tokens: int,
        query_tokens: int,
        context_window: Optional[int] = None,
    ) -> int:
        """
        Estimate remaining budget for retrieval context.
        
        Useful for dynamic budget allocation based on other content.
        
        Args:
            system_tokens: Tokens used by system prompt
            history_tokens: Tokens used by conversation history
            query_tokens: Tokens used by user query
            context_window: Model's context window (default: config)
            
        Returns:
            Available tokens for retrieval context
        """
        context_window = context_window or settings.llm.context_window
        
        # Reserve space for response (typically 20-30% of window)
        response_reserve = context_window // 4
        
        used = system_tokens + history_tokens + query_tokens
        available = context_window - response_reserve - used
        
        # Cap at our configured budget
        return min(available, self.effective_budget)
    
    def format_budget_report(self, allocation: BudgetAllocation) -> str:
        """Format a human-readable budget report."""
        return (
            f"Token Budget: {allocation.total_tokens}/{allocation.budget} "
            f"({allocation.utilization:.0%} utilization)\n"
            f"Facts: {len(allocation.facts)} included, "
            f"{allocation.truncated_count} truncated, "
            f"{allocation.dropped_count} dropped"
        )


@lru_cache(maxsize=1)
def get_token_budgeter() -> TokenBudgeter:
    """Get singleton token budgeter instance."""
    return TokenBudgeter()


def get_token_counter() -> TokenCounter:
    """Get a token counter instance."""
    return TokenCounter()
