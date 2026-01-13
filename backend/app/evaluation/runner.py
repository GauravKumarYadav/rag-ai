"""Evaluation runner - executes RAG evaluation against a dataset.

Runs retrieval for each Q&A pair, computes metrics, and stores results.

Supports two types of evaluation:
1. Synthetic: Auto-generated Q&A pairs from indexed documents
2. Curated: Hand-crafted test cases with expected answers and sources
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from pathlib import Path

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

# Path to curated datasets
CURATED_DATASETS_PATH = Path("data/evaluation")


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


# =============================================================================
# CURATED GOLD DATASET EVALUATION
# =============================================================================

def load_curated_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Load a curated evaluation dataset from JSON file.
    
    Args:
        dataset_path: Path to JSON file (relative to CURATED_DATASETS_PATH or absolute)
        
    Returns:
        Parsed dataset dictionary
    """
    # Try relative path first
    full_path = CURATED_DATASETS_PATH / dataset_path
    if not full_path.exists():
        # Try absolute path
        full_path = Path(dataset_path)
    
    if not full_path.exists():
        raise ValueError(f"Curated dataset not found: {dataset_path}")
    
    with open(full_path, "r") as f:
        return json.load(f)


def list_curated_datasets() -> List[Dict[str, Any]]:
    """
    List all available curated datasets.
    
    Returns:
        List of dataset summaries
    """
    datasets = []
    
    if not CURATED_DATASETS_PATH.exists():
        return datasets
    
    for file_path in CURATED_DATASETS_PATH.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            datasets.append({
                "filename": file_path.name,
                "name": data.get("name", file_path.stem),
                "description": data.get("description", ""),
                "version": data.get("version", "1.0"),
                "case_count": len(data.get("cases", [])),
                "categories": data.get("categories", []),
            })
        except Exception as e:
            logger.warning(f"Could not load dataset {file_path}: {e}")
    
    return datasets


async def run_curated_evaluation(
    dataset_path: str = "curated_gold.json",
    client_id: Optional[str] = None,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Run evaluation on a curated gold dataset.
    
    This evaluates:
    - Retrieval quality (precision, recall, MRR)
    - Answer quality (keyword coverage, source accuracy)
    - Negative case handling (correctly rejecting unanswerable queries)
    
    Args:
        dataset_path: Path to curated dataset JSON
        client_id: Optional client ID to scope retrieval
        k: Number of top results to consider
        
    Returns:
        Detailed evaluation results
    """
    # Load dataset
    dataset = load_curated_dataset(dataset_path)
    cases = dataset.get("cases", [])
    criteria = dataset.get("evaluation_criteria", {})
    
    logger.info(
        f"Starting curated evaluation",
        extra={
            "dataset": dataset.get("name"),
            "case_count": len(cases),
            "client_id": client_id,
        }
    )
    
    # Get retriever
    retriever = get_retriever()
    
    # Results by category
    category_results: Dict[str, List[Dict]] = {}
    all_results: List[Dict[str, Any]] = []
    
    for case in cases:
        case_id = case.get("id", "unknown")
        category = case.get("category", "general")
        query = case.get("query", "")
        expected_sources = set(case.get("expected_sources", []))
        expected_chunk_ids = set(case.get("expected_chunk_ids", []))
        keywords = case.get("keywords", [])
        is_negative = case.get("negative", False)
        
        if not query:
            continue
        
        try:
            # Retrieve documents
            if client_id:
                hits = retriever.search_with_client_filter(
                    query=query,
                    client_id=client_id,
                    top_k=k * 2,
                )
            else:
                hits = retriever.search(query, top_k=k * 2)
            
            retrieved_ids = [hit.id for hit in hits]
            retrieved_sources = [
                hit.metadata.get("source_filename", hit.metadata.get("source", ""))
                for hit in hits
            ]
            retrieved_texts = [hit.content for hit in hits]
            
            # Compute retrieval metrics
            if expected_chunk_ids:
                precision = compute_precision_at_k(retrieved_ids, expected_chunk_ids, k)
                recall = compute_recall_at_k(retrieved_ids, expected_chunk_ids, k)
                mrr = compute_mrr(retrieved_ids, expected_chunk_ids)
            elif expected_sources:
                # Match by source filename
                retrieved_source_set = set(retrieved_sources[:k])
                source_hits = len(expected_sources & retrieved_source_set)
                precision = source_hits / k if k > 0 else 0
                recall = source_hits / len(expected_sources) if expected_sources else 0
                mrr = 0.0
                for i, src in enumerate(retrieved_sources[:k]):
                    if src in expected_sources:
                        mrr = 1.0 / (i + 1)
                        break
            else:
                precision = 0.0
                recall = 0.0
                mrr = 0.0
            
            # Compute keyword coverage (how many expected keywords appear in retrieved text)
            all_retrieved_text = " ".join(retrieved_texts[:k]).lower()
            keyword_hits = sum(1 for kw in keywords if kw.lower() in all_retrieved_text)
            keyword_coverage = keyword_hits / len(keywords) if keywords else 1.0
            
            # For negative cases, check if retrieval correctly returns low-quality results
            negative_handled = False
            if is_negative:
                # Negative case: should NOT find relevant content
                # Consider it handled if no high-confidence results or keywords not found
                if not hits or (hits[0].score > 0.5 and keyword_coverage < 0.3):
                    negative_handled = True
            
            result = {
                "case_id": case_id,
                "category": category,
                "query": query,
                "is_negative": is_negative,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "mrr": mrr,
                "keyword_coverage": keyword_coverage,
                "retrieved_count": len(hits),
                "negative_handled": negative_handled if is_negative else None,
                "sources_found": list(set(retrieved_sources[:k]) & expected_sources) if expected_sources else [],
            }
            
            all_results.append(result)
            
            # Group by category
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
            
        except Exception as e:
            logger.error(f"Error evaluating case {case_id}: {e}")
            all_results.append({
                "case_id": case_id,
                "category": category,
                "query": query,
                "error": str(e),
            })
    
    # Aggregate metrics by category
    category_metrics = {}
    for cat, results in category_results.items():
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            continue
        
        category_metrics[cat] = {
            "count": len(valid_results),
            "mean_precision": sum(r["precision_at_k"] for r in valid_results) / len(valid_results),
            "mean_recall": sum(r["recall_at_k"] for r in valid_results) / len(valid_results),
            "mean_mrr": sum(r["mrr"] for r in valid_results) / len(valid_results),
            "mean_keyword_coverage": sum(r["keyword_coverage"] for r in valid_results) / len(valid_results),
        }
        
        # For negative cases, compute rejection rate
        negative_cases = [r for r in valid_results if r.get("is_negative")]
        if negative_cases:
            category_metrics[cat]["negative_cases"] = len(negative_cases)
            category_metrics[cat]["rejection_rate"] = (
                sum(1 for r in negative_cases if r.get("negative_handled")) / len(negative_cases)
            )
    
    # Compute overall metrics
    valid_results = [r for r in all_results if "error" not in r]
    positive_results = [r for r in valid_results if not r.get("is_negative")]
    negative_results = [r for r in valid_results if r.get("is_negative")]
    
    overall_metrics = {
        "total_cases": len(cases),
        "evaluated_cases": len(valid_results),
        "positive_cases": len(positive_results),
        "negative_cases": len(negative_results),
    }
    
    if positive_results:
        overall_metrics["mean_precision"] = sum(r["precision_at_k"] for r in positive_results) / len(positive_results)
        overall_metrics["mean_recall"] = sum(r["recall_at_k"] for r in positive_results) / len(positive_results)
        overall_metrics["mean_mrr"] = sum(r["mrr"] for r in positive_results) / len(positive_results)
        overall_metrics["mean_keyword_coverage"] = sum(r["keyword_coverage"] for r in positive_results) / len(positive_results)
    
    if negative_results:
        overall_metrics["rejection_rate"] = (
            sum(1 for r in negative_results if r.get("negative_handled")) / len(negative_results)
        )
    
    # Compare against criteria
    passes_criteria = True
    criteria_results = {}
    
    if "retrieval" in criteria:
        ret_criteria = criteria["retrieval"]
        if "min_precision" in ret_criteria:
            criteria_results["precision"] = overall_metrics.get("mean_precision", 0) >= ret_criteria["min_precision"]
            passes_criteria = passes_criteria and criteria_results["precision"]
        if "min_recall" in ret_criteria:
            criteria_results["recall"] = overall_metrics.get("mean_recall", 0) >= ret_criteria["min_recall"]
            passes_criteria = passes_criteria and criteria_results["recall"]
        if "min_mrr" in ret_criteria:
            criteria_results["mrr"] = overall_metrics.get("mean_mrr", 0) >= ret_criteria["min_mrr"]
            passes_criteria = passes_criteria and criteria_results["mrr"]
    
    if "negative_cases" in criteria and negative_results:
        neg_criteria = criteria["negative_cases"]
        if "rejection_rate" in neg_criteria:
            criteria_results["rejection_rate"] = overall_metrics.get("rejection_rate", 0) >= neg_criteria["rejection_rate"]
            passes_criteria = passes_criteria and criteria_results["rejection_rate"]
    
    logger.info(
        f"Curated evaluation completed",
        extra={
            "dataset": dataset.get("name"),
            "passes_criteria": passes_criteria,
            "mean_precision": overall_metrics.get("mean_precision", 0),
            "mean_recall": overall_metrics.get("mean_recall", 0),
        }
    )
    
    return {
        "dataset_name": dataset.get("name"),
        "dataset_version": dataset.get("version"),
        "evaluation_type": "curated_gold",
        "k": k,
        "client_id": client_id,
        "overall_metrics": overall_metrics,
        "category_metrics": category_metrics,
        "criteria_results": criteria_results,
        "passes_all_criteria": passes_criteria,
        "detailed_results": all_results,
        "evaluated_at": datetime.utcnow().isoformat(),
    }


async def run_evaluation_combined(
    dataset_id: Optional[int] = None,
    curated_path: Optional[str] = None,
    dataset_type: str = "synthetic",
    client_id: Optional[str] = None,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Run evaluation with support for both synthetic and curated datasets.
    
    Args:
        dataset_id: Database ID for synthetic datasets
        curated_path: Path for curated dataset file
        dataset_type: "synthetic" or "curated_gold"
        client_id: Optional client ID for scoping
        k: Top-k for metrics
        
    Returns:
        Evaluation results
    """
    if dataset_type == "curated_gold":
        if not curated_path:
            curated_path = "curated_gold.json"
        return await run_curated_evaluation(
            dataset_path=curated_path,
            client_id=client_id,
            k=k,
        )
    else:
        if not dataset_id:
            raise ValueError("dataset_id required for synthetic evaluation")
        return await run_evaluation(
            dataset_id=dataset_id,
            k=k,
        )
