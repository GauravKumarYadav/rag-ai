"""RAG evaluation metrics.

Implements standard IR metrics:
- Precision@K: Fraction of retrieved docs that are relevant
- Recall@K: Fraction of relevant docs that are retrieved
- MRR: Mean Reciprocal Rank of first relevant result
- Faithfulness: Answer grounded in retrieved context
"""

from typing import List, Dict, Any, Optional, Set
from app.core.logging import get_logger

logger = get_logger(__name__)


def compute_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 5,
) -> float:
    """Compute Precision@K.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: Set of ground truth relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if not retrieved_ids or k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    
    return relevant_in_top_k / k


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 5,
) -> float:
    """Compute Recall@K.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: Set of ground truth relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    if not retrieved_ids or k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    
    return relevant_in_top_k / len(relevant_ids)


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """Compute Mean Reciprocal Rank.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: Set of ground truth relevant document IDs
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    
    return 0.0


async def compute_faithfulness(
    answer: str,
    retrieved_contexts: List[str],
    llm_client: Any,
) -> float:
    """Compute faithfulness score using LLM.
    
    Measures whether the answer is grounded in the retrieved context.
    
    Args:
        answer: Generated answer text
        retrieved_contexts: List of retrieved document texts
        llm_client: LLM client for evaluation
        
    Returns:
        Faithfulness score (0.0 to 1.0)
    """
    if not answer or not retrieved_contexts:
        return 0.0
    
    context = "\n\n".join(retrieved_contexts[:5])  # Limit context size
    
    prompt = f"""Evaluate if the following answer is faithful to (fully supported by) the given context.

Context:
---
{context[:3000]}
---

Answer:
---
{answer}
---

Score the faithfulness from 0 to 1:
- 1.0: Answer is fully supported by the context
- 0.5: Answer is partially supported
- 0.0: Answer contains claims not in context

Respond with ONLY a number between 0 and 1."""

    try:
        response = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        
        content = response.get("content", "0").strip()
        # Extract first number from response
        import re
        match = re.search(r"(\d+\.?\d*)", content)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        
        return 0.0
        
    except Exception as e:
        logger.error(f"Error computing faithfulness: {e}")
        return 0.0


def aggregate_metrics(
    results: List[Dict[str, float]],
) -> Dict[str, float]:
    """Aggregate metrics across multiple evaluation samples.
    
    Args:
        results: List of per-sample metric dicts
        
    Returns:
        Aggregated metrics with mean values
    """
    if not results:
        return {}
    
    # Collect all metric keys
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    
    # Compute means
    aggregated = {}
    for key in all_keys:
        values = [r.get(key, 0.0) for r in results if key in r]
        if values:
            aggregated[f"mean_{key}"] = sum(values) / len(values)
            aggregated[f"min_{key}"] = min(values)
            aggregated[f"max_{key}"] = max(values)
    
    aggregated["sample_count"] = len(results)
    
    return aggregated
