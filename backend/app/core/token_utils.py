"""
Token Estimation Utilities for Production RAG.

Provides consistent, cached token counting across the application
with both accurate (tiktoken) and fast approximate modes.
"""

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Default encoding for tiktoken
DEFAULT_ENCODING = "cl100k_base"
APPROX_CHARS_PER_TOKEN = 4.0  # ~4 chars per token for English text


class TokenEstimator:
    """
    Production token estimator with caching and minimal overhead.
    
    Features:
    - LRU-cached accurate token counts (tiktoken)
    - Fast approximation mode for high-volume scenarios
    - Consistent interface across all application components
    - Graceful degradation if tiktoken unavailable
    """
    
    def __init__(self, encoding: str = DEFAULT_ENCODING, cache_size: int = 1024):
        self.encoding_name = encoding
        self._encoder: Optional[Any] = None
        self._cache_size = cache_size
        self._approx_ratio = APPROX_CHARS_PER_TOKEN
        self._encoder_available = True
    
    @property
    def encoder(self) -> Optional[Any]:
        """Lazy-load tiktoken encoder with error handling."""
        if self._encoder is None and self._encoder_available:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding(self.encoding_name)
                logger.info(f"Loaded tiktoken encoding: {self.encoding_name}")
            except ImportError:
                logger.warning("tiktoken not available, using approximation only")
                self._encoder_available = False
            except Exception as e:
                logger.warning(f"Failed to load tiktoken: {e}, using approximation")
                self._encoder_available = False
        return self._encoder
    
    @lru_cache(maxsize=1024)
    def estimate_precise(self, text: str) -> int:
        """
        Accurate token count using tiktoken (cached for repeated texts).
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Token count (>= 0)
        """
        if not text:
            return 0
        
        encoder = self.encoder
        if encoder:
            try:
                return len(encoder.encode(text))
            except Exception as e:
                logger.debug(f"tiktoken encoding failed: {e}, using fallback")
        
        return self.estimate_fast(text)
    
    def estimate_fast(self, text: str) -> int:
        """
        Fast approximation using heuristics (O(1) allocation).
        
        Uses character-based heuristics optimized for English text:
        - ~4 characters per token for standard English
        - Adjustments for code (many spaces = fewer tokens)
        - Unicode detection for CJK content
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count (>= 1 if text non-empty)
        """
        if not text:
            return 0
        
        length = len(text)
        
        # Detect Unicode/CJK content (>20% non-ASCII = assume mixed content)
        # CJK characters are typically 1 token per character
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        if non_ascii_count > length * 0.2:
            # Conservative estimate for mixed/Unicode content
            return max(1, int(length * 0.8))
        
        # Adjust for code/spaces (consecutive spaces are cheap in tokenization)
        space_count = text.count(' ')
        newline_count = text.count('\n')
        code_adjustment = (space_count + newline_count) * 0.3
        
        adjusted_length = length - code_adjustment
        return max(1, int(adjusted_length / self._approx_ratio))
    
    def estimate(self, text: str, fast: bool = False) -> int:
        """
        Default estimation with speed/accuracy tradeoff.
        
        Args:
            text: Text to count tokens in
            fast: If True, always use fast approximation.
                  If False, use precise for short texts, fast for long.
                  
        Returns:
            Token count
        """
        if not text:
            return 0
        
        if fast or len(text) > 10000:  # Use fast mode for long texts
            return self.estimate_fast(text)
        
        return self.estimate_precise(text)
    
    def estimate_batch(self, texts: list, fast: bool = False) -> list:
        """
        Efficiently estimate tokens for multiple texts.
        
        Args:
            texts: List of strings to estimate
            fast: Whether to use fast approximation
            
        Returns:
            List of token counts
        """
        return [self.estimate(t, fast=fast) for t in texts]


# Singleton instance for application-wide use
_token_estimator: Optional[TokenEstimator] = None


def get_token_estimator() -> TokenEstimator:
    """Get or create singleton token estimator."""
    global _token_estimator
    if _token_estimator is None:
        _token_estimator = TokenEstimator()
    return _token_estimator


def estimate_tokens(text: str, fast: bool = False) -> int:
    """
    Convenience function for direct token estimation.
    
    Args:
        text: Text to estimate
        fast: Use fast approximation
        
    Returns:
        Estimated token count
    """
    return get_token_estimator().estimate(text, fast=fast)


def estimate_tokens_batch(texts: list, fast: bool = False) -> list:
    """
    Convenience function for batch token estimation.
    
    Args:
        texts: List of strings
        fast: Use fast approximation
        
    Returns:
        List of token counts
    """
    return get_token_estimator().estimate_batch(texts, fast=fast)


def estimate_messages_tokens(messages: list, fast: bool = False) -> int:
    """
    Estimate tokens for a list of chat messages.
    
    Args:
        messages: List of message objects with 'content' attribute or dict with 'content' key
        fast: Use fast approximation
        
    Returns:
        Total estimated tokens
    """
    total = 0
    for msg in messages:
        if hasattr(msg, 'content'):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get('content', '')
        else:
            content = str(msg)
        total += estimate_tokens(content, fast=fast)
    return total
