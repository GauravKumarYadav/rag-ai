"""Evaluation runner - executes RAG evaluation against a dataset.

Runs retrieval for each Q&A pair, computes metrics, and stores results.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List

from app.clients.lmstudio import get_lmstudio_client
from app.rag.retriever import get_retriever
from app.rag.vector_store import get_client_vector_store
from app.evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_faithfulness,
    aggregate_metrics,
)
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


async def run_evaluation(
    dataset_id: int,
    k: int = 5,
) -> Dict[str, Any]:
    """Run evaluation on a dataset.
    
    Args:
        dataset_id: ID of the evaluation dataset
        k: Number of top results to consider for metrics
        
    Returns:
        Evaluation results with aggregated metrics
    """
    from app.db.mysql import get_db_pool
    
    # Load dataset
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT name, client_id, qa_pairs FROM evaluation_datasets WHERE id = %s",
                (dataset_id,)
            )
            row = await cursor.fetchone()
            
            if not row:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            dataset_name, client_id, qa_pairs_json = row
            qa_pairs = json.loads(qa_pairs_json)
    
    logger.info(
        f"Starting evaluation run",
        extra={"dataset_id": dataset_id, "dataset_name": dataset_name, "sample_count": len(qa_pairs)}
    )
    
    # Get retriever and LLM client
    retriever = get_retriever()
    llm_client = get_lmstudio_client()
    
    # Run evaluation for each Q&A pair
    per_sample_results = []
    
    for qa in qa_pairs:
        question = qa.get("question", "")
        expected_chunk_id = qa.get("chunk_id", "")
        
        if not question:
            continue
        
        try:
            # Retrieve documents
            hits = retriever.search(question, top_k=k * 2)  # Get more for recall
            retrieved_ids = [hit.id for hit in hits]
            retrieved_texts = [hit.content for hit in hits]
            
            # Ground truth: the chunk the question was generated from
            relevant_ids = {expected_chunk_id}
            
            # Compute retrieval metrics
            precision = compute_precision_at_k(retrieved_ids, relevant_ids, k)
            recall = compute_recall_at_k(retrieved_ids, relevant_ids, k)
            mrr = compute_mrr(retrieved_ids, relevant_ids)
            
            # Compute faithfulness if we have retrieved context
            faithfulness = 0.0
            if settings.evaluation.compute_faithfulness and retrieved_texts:
                # Generate an answer using the retrieved context
                context = "\n\n".join(retrieved_texts[:k])
                answer_prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                
                response = await llm_client.chat(
                    messages=[{"role": "user", "content": answer_prompt}],
                    temperature=0.1,
                )
                generated_answer = response.get("content", "")
                
                faithfulness = await compute_faithfulness(
                    generated_answer,
                    retrieved_texts[:k],
                    llm_client,
                )
            
            sample_result = {
                "question": question,
                "chunk_id": expected_chunk_id,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "mrr": mrr,
                "faithfulness": faithfulness,
                "retrieved_count": len(retrieved_ids),
                "hit_in_top_k": expected_chunk_id in retrieved_ids[:k],
            }
            
            per_sample_results.append(sample_result)
            
        except Exception as e:
            logger.error(f"Error evaluating question: {e}", extra={"question": question[:100]})
            continue
    
    # Aggregate metrics
    aggregated = aggregate_metrics([
        {
            "precision_at_k": r["precision_at_k"],
            "recall_at_k": r["recall_at_k"],
            "mrr": r["mrr"],
            "faithfulness": r["faithfulness"],
        }
        for r in per_sample_results
    ])
    
    # Calculate hit rate
    hit_count = sum(1 for r in per_sample_results if r.get("hit_in_top_k"))
    aggregated["hit_rate"] = hit_count / len(per_sample_results) if per_sample_results else 0.0
    
    # Save evaluation run
    run_id = await save_evaluation_run(
        dataset_id=dataset_id,
        metrics=aggregated,
        per_sample_results=per_sample_results,
        k=k,
    )
    
    logger.info(
        f"Evaluation run completed",
        extra={
            "run_id": run_id,
            "dataset_id": dataset_id,
            "mean_precision": aggregated.get("mean_precision_at_k", 0),
            "mean_recall": aggregated.get("mean_recall_at_k", 0),
            "mean_mrr": aggregated.get("mean_mrr", 0),
            "hit_rate": aggregated.get("hit_rate", 0),
        }
    )
    
    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "metrics": aggregated,
        "sample_count": len(per_sample_results),
        "k": k,
    }


async def save_evaluation_run(
    dataset_id: int,
    metrics: Dict[str, float],
    per_sample_results: List[Dict[str, Any]],
    k: int,
) -> int:
    """Save evaluation run results to database.
    
    Args:
        dataset_id: Dataset ID
        metrics: Aggregated metrics
        per_sample_results: Per-sample detailed results
        k: K value used for metrics
        
    Returns:
        Run ID
    """
    from app.db.mysql import get_db_pool
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO evaluation_runs 
                (dataset_id, k_value, metrics, detailed_results, status)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    dataset_id,
                    k,
                    json.dumps(metrics),
                    json.dumps(per_sample_results),
                    "completed",
                )
            )
            await conn.commit()
            return cursor.lastrowid


async def get_evaluation_run(run_id: int) -> Optional[Dict[str, Any]]:
    """Get evaluation run details.
    
    Args:
        run_id: Run ID
        
    Returns:
        Run details or None if not found
    """
    from app.db.mysql import get_db_pool
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT r.id, r.dataset_id, r.k_value, r.metrics, r.detailed_results,
                       r.status, r.created_at, d.name as dataset_name
                FROM evaluation_runs r
                JOIN evaluation_datasets d ON r.dataset_id = d.id
                WHERE r.id = %s
                """,
                (run_id,)
            )
            row = await cursor.fetchone()
            
            if not row:
                return None
            
            return {
                "id": row[0],
                "dataset_id": row[1],
                "k_value": row[2],
                "metrics": json.loads(row[3]) if row[3] else {},
                "detailed_results": json.loads(row[4]) if row[4] else [],
                "status": row[5],
                "created_at": row[6].isoformat() if row[6] else None,
                "dataset_name": row[7],
            }


async def list_evaluation_runs(
    dataset_id: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List evaluation runs.
    
    Args:
        dataset_id: Optional filter by dataset
        limit: Max results
        offset: Pagination offset
        
    Returns:
        List of run summaries
    """
    from app.db.mysql import get_db_pool
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            if dataset_id:
                await cursor.execute(
                    """
                    SELECT r.id, r.dataset_id, r.k_value, r.metrics, r.status, 
                           r.created_at, d.name as dataset_name
                    FROM evaluation_runs r
                    JOIN evaluation_datasets d ON r.dataset_id = d.id
                    WHERE r.dataset_id = %s
                    ORDER BY r.created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (dataset_id, limit, offset)
                )
            else:
                await cursor.execute(
                    """
                    SELECT r.id, r.dataset_id, r.k_value, r.metrics, r.status,
                           r.created_at, d.name as dataset_name
                    FROM evaluation_runs r
                    JOIN evaluation_datasets d ON r.dataset_id = d.id
                    ORDER BY r.created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset)
                )
            
            rows = await cursor.fetchall()
            
            return [
                {
                    "id": row[0],
                    "dataset_id": row[1],
                    "k_value": row[2],
                    "metrics": json.loads(row[3]) if row[3] else {},
                    "status": row[4],
                    "created_at": row[5].isoformat() if row[5] else None,
                    "dataset_name": row[6],
                }
                for row in rows
            ]
