"""
Evaluation Worker - Scheduled RAG evaluation service.

Runs independently from the main application:
- Daily scheduled evaluations
- Dedicated resources for evaluation runs
- Stores results in MySQL
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

import httpx
import aiomysql
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Configuration
MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "Sarita1!@2024_4")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "audit_logs")
CHROMADB_URL = os.getenv("CHROMADB_URL", "http://chromadb:8000")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:1b")

# Cron schedule (default: 2 AM UTC daily)
CRON_SCHEDULE = os.getenv("CRON_SCHEDULE", "0 2 * * *")
CRON_TIMEZONE = os.getenv("CRON_TIMEZONE", "UTC")

# Global clients
db_pool: Optional[aiomysql.Pool] = None
http_client: Optional[httpx.AsyncClient] = None
scheduler: Optional[AsyncIOScheduler] = None


async def init_database():
    """Initialize database connection and create tables."""
    global db_pool
    
    db_pool = await aiomysql.create_pool(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DATABASE,
        autocommit=True,
        charset="utf8mb4",
    )
    
    # Create tables if not exist
    async with db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_datasets (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    client_id VARCHAR(255),
                    qa_pairs JSON NOT NULL,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)
            
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    dataset_id INT NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP NULL,
                    status VARCHAR(50) DEFAULT 'running',
                    metrics JSON,
                    per_sample_results JSON,
                    error_message TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES evaluation_datasets(id)
                )
            """)
    
    print(f"Connected to MySQL: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from embedding service."""
    response = await http_client.post(
        f"{EMBEDDING_SERVICE_URL}/embed",
        json={"texts": texts}
    )
    response.raise_for_status()
    return response.json()["embeddings"]


async def query_chromadb(
    collection_name: str,
    query_embedding: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Query ChromaDB for similar documents."""
    try:
        # Get collection
        response = await http_client.get(
            f"{CHROMADB_URL}/api/v2/collections/{collection_name}"
        )
        if response.status_code == 404:
            return []
        
        collection = response.json()
        collection_id = collection.get("id", collection_name)
        
        # Query
        response = await http_client.post(
            f"{CHROMADB_URL}/api/v2/collections/{collection_id}/query",
            json={
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        ids = data.get("ids", [[]])[0]
        documents = data.get("documents", [[]])[0]
        metadatas = data.get("metadatas", [[]])[0]
        distances = data.get("distances", [[]])[0]
        
        for i, doc_id in enumerate(ids):
            results.append({
                "id": doc_id,
                "content": documents[i] if i < len(documents) else "",
                "distance": distances[i] if i < len(distances) else 0.0,
                "metadata": metadatas[i] if i < len(metadatas) else {},
            })
        
        return results
        
    except Exception as e:
        print(f"ChromaDB query failed: {e}")
        return []


async def compute_faithfulness(
    question: str,
    answer: str,
    context: str,
) -> float:
    """Use LLM to compute faithfulness score."""
    prompt = f"""Rate how well the answer is supported by the context on a scale of 0 to 1.
0 = answer contradicts or has no support from context
1 = answer is fully supported by context

Context: {context[:2000]}

Question: {question}

Answer: {answer}

Respond with only a number between 0 and 1."""

    try:
        response = await http_client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
            }
        )
        response.raise_for_status()
        text = response.json().get("response", "0.5").strip()
        
        # Extract number from response
        import re
        match = re.search(r"[0-9.]+", text)
        if match:
            score = float(match.group())
            return min(1.0, max(0.0, score))
        return 0.5
        
    except Exception as e:
        print(f"Faithfulness computation failed: {e}")
        return 0.5


async def run_evaluation(dataset_id: int, k: int = 5) -> Dict[str, Any]:
    """Run evaluation on a dataset."""
    # Load dataset
    async with db_pool.acquire() as conn:
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
    
    print(f"Starting evaluation: dataset={dataset_name}, samples={len(qa_pairs)}")
    
    # Create evaluation run record
    async with db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO evaluation_runs (dataset_id, status) VALUES (%s, 'running')",
                (dataset_id,)
            )
            run_id = cursor.lastrowid
    
    # Determine collection
    collection_name = f"{client_id}_documents" if client_id else "documents"
    
    # Run evaluation
    per_sample_results = []
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    faithfulness_scores = []
    hit_count = 0
    
    for qa in qa_pairs:
        question = qa.get("question", "")
        expected_chunk_id = qa.get("chunk_id", "")
        expected_answer = qa.get("answer", "")
        
        if not question:
            continue
        
        try:
            # Get query embedding
            embeddings = await get_embeddings([question])
            query_embedding = embeddings[0]
            
            # Retrieve documents
            hits = await query_chromadb(collection_name, query_embedding, top_k=k * 2)
            retrieved_ids = [h["id"] for h in hits]
            retrieved_texts = [h["content"] for h in hits]
            
            # Compute metrics
            # Precision@k
            relevant_in_top_k = 1 if expected_chunk_id in retrieved_ids[:k] else 0
            precision = relevant_in_top_k / k if k > 0 else 0
            precision_scores.append(precision)
            
            # Recall@k (binary for single relevant doc)
            recall = 1.0 if expected_chunk_id in retrieved_ids[:k] else 0.0
            recall_scores.append(recall)
            
            # MRR
            try:
                rank = retrieved_ids.index(expected_chunk_id) + 1
                mrr = 1.0 / rank
            except ValueError:
                mrr = 0.0
            mrr_scores.append(mrr)
            
            # Hit rate
            if expected_chunk_id in retrieved_ids[:k]:
                hit_count += 1
            
            # Faithfulness (if we have expected answer)
            if expected_answer and retrieved_texts:
                context = "\n".join(retrieved_texts[:3])
                faithfulness = await compute_faithfulness(question, expected_answer, context)
                faithfulness_scores.append(faithfulness)
            
            per_sample_results.append({
                "question": question,
                "expected_chunk_id": expected_chunk_id,
                "retrieved_ids": retrieved_ids[:k],
                "precision": precision,
                "recall": recall,
                "mrr": mrr,
            })
            
        except Exception as e:
            print(f"Error evaluating question: {e}")
            per_sample_results.append({
                "question": question,
                "error": str(e),
            })
    
    # Aggregate metrics
    total = len(qa_pairs)
    metrics = {
        "precision_at_k": sum(precision_scores) / len(precision_scores) if precision_scores else 0,
        "recall_at_k": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
        "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
        "hit_rate": hit_count / total if total > 0 else 0,
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
        "total_samples": total,
        "k": k,
    }
    
    # Update run record
    async with db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """UPDATE evaluation_runs 
                   SET status = 'completed', 
                       completed_at = NOW(),
                       metrics = %s,
                       per_sample_results = %s
                   WHERE id = %s""",
                (json.dumps(metrics), json.dumps(per_sample_results), run_id)
            )
    
    print(f"Evaluation completed: run_id={run_id}, hit_rate={metrics['hit_rate']:.2%}")
    
    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "metrics": metrics,
    }


async def scheduled_evaluation():
    """Run scheduled evaluation on all active datasets."""
    print(f"[{datetime.utcnow().isoformat()}] Starting scheduled evaluation")
    
    try:
        async with db_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT id, name FROM evaluation_datasets WHERE active = TRUE"
                )
                datasets = await cursor.fetchall()
        
        if not datasets:
            print("No active datasets for evaluation")
            return
        
        for dataset_id, dataset_name in datasets:
            try:
                print(f"Evaluating dataset: {dataset_name}")
                result = await run_evaluation(dataset_id)
                print(f"Dataset {dataset_name}: hit_rate={result['metrics']['hit_rate']:.2%}")
            except Exception as e:
                print(f"Evaluation failed for {dataset_name}: {e}")
        
        print(f"Scheduled evaluation completed for {len(datasets)} datasets")
        
    except Exception as e:
        print(f"Scheduled evaluation error: {e}")


async def main():
    """Main entry point."""
    global http_client, scheduler
    
    print("Evaluation Worker starting...")
    print(f"Schedule: {CRON_SCHEDULE} ({CRON_TIMEZONE})")
    
    # Initialize connections
    http_client = httpx.AsyncClient(timeout=120.0)
    await init_database()
    
    # Setup scheduler
    scheduler = AsyncIOScheduler()
    
    # Parse cron expression
    parts = CRON_SCHEDULE.split()
    if len(parts) == 5:
        trigger = CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4],
            timezone=CRON_TIMEZONE,
        )
    else:
        # Default: 2 AM daily
        trigger = CronTrigger(hour=2, minute=0, timezone=CRON_TIMEZONE)
    
    scheduler.add_job(scheduled_evaluation, trigger, id="daily_evaluation")
    scheduler.start()
    
    print("Evaluation worker ready. Press Ctrl+C to stop.")
    
    # Run initial evaluation if requested
    if os.getenv("RUN_ON_STARTUP", "false").lower() == "true":
        await scheduled_evaluation()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour
    except KeyboardInterrupt:
        pass
    finally:
        scheduler.shutdown()
        await http_client.aclose()
        db_pool.close()
        await db_pool.wait_closed()
        print("Evaluation worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
