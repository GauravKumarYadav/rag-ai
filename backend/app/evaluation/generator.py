"""Auto-generate Q&A pairs from document chunks using LLM.

Uses the configured LLM to create question-answer pairs that can be used
as ground truth for evaluating RAG retrieval quality.
"""

import json
import random
from typing import List, Optional, Dict, Any

from app.clients.lmstudio import get_lmstudio_client
from app.rag.vector_store import get_vector_store, get_client_vector_store
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Prompt template for Q&A generation
QA_GENERATION_PROMPT = """You are generating question-answer pairs for evaluating a RAG system.

Given the following document chunk, generate a question that can ONLY be answered using information from this chunk, and provide the correct answer.

Document chunk:
---
{chunk}
---

Generate a JSON object with:
- "question": A specific question answerable from this chunk
- "answer": The correct answer based on the chunk
- "chunk_id": "{chunk_id}"

Respond ONLY with valid JSON, no other text."""


async def generate_qa_pairs(
    client_id: Optional[str] = None,
    sample_size: Optional[int] = None,
    min_chunk_length: int = 100,
) -> List[Dict[str, Any]]:
    """Generate Q&A pairs from document chunks.
    
    Args:
        client_id: Optional client ID to generate from client-specific docs
        sample_size: Number of Q&A pairs to generate (default from settings)
        min_chunk_length: Minimum chunk length to consider
        
    Returns:
        List of dicts with question, answer, chunk_id, source
    """
    sample_size = sample_size or settings.evaluation.default_sample_size
    
    # Get vector store
    if client_id:
        store = get_client_vector_store(client_id)
    else:
        store = get_vector_store()
    
    # Fetch all document chunks
    try:
        all_docs = store.docs.get(include=["documents", "metadatas"])
    except Exception as e:
        logger.error(f"Failed to fetch documents: {e}")
        return []
    
    doc_ids = all_docs.get("ids", [])
    documents = all_docs.get("documents", [])
    metadatas = all_docs.get("metadatas", [])
    
    if not doc_ids:
        logger.warning("No documents found for Q&A generation")
        return []
    
    # Filter chunks by minimum length
    valid_indices = [
        i for i, doc in enumerate(documents)
        if doc and len(doc) >= min_chunk_length
    ]
    
    if not valid_indices:
        logger.warning("No chunks meet minimum length requirement")
        return []
    
    # Sample chunks
    sample_indices = random.sample(
        valid_indices,
        min(sample_size, len(valid_indices))
    )
    
    # Get LLM client
    llm_client = get_lmstudio_client()
    
    qa_pairs = []
    
    for idx in sample_indices:
        chunk_id = doc_ids[idx]
        chunk_text = documents[idx]
        metadata = metadatas[idx] if metadatas else {}
        source = metadata.get("source", "unknown")
        
        prompt = QA_GENERATION_PROMPT.format(
            chunk=chunk_text[:2000],  # Limit chunk size for prompt
            chunk_id=chunk_id,
        )
        
        try:
            response = await llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            
            # Parse JSON response
            content = response.get("content", "")
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            qa_data = json.loads(content.strip())
            qa_data["source"] = source
            qa_data["chunk_id"] = chunk_id
            qa_pairs.append(qa_data)
            
            logger.info(
                f"Generated Q&A pair for chunk {chunk_id}",
                extra={"source": source, "question_length": len(qa_data.get("question", ""))}
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            continue
        except Exception as e:
            logger.error(f"Error generating Q&A for chunk {chunk_id}: {e}")
            continue
    
    logger.info(
        f"Generated {len(qa_pairs)} Q&A pairs",
        extra={"client_id": client_id, "sample_size": sample_size}
    )
    
    return qa_pairs


async def save_dataset(
    qa_pairs: List[Dict[str, Any]],
    name: str,
    client_id: Optional[str] = None,
) -> int:
    """Save generated Q&A pairs as an evaluation dataset.
    
    Args:
        qa_pairs: List of Q&A pair dicts
        name: Dataset name
        client_id: Optional client ID
        
    Returns:
        Dataset ID
    """
    from app.db.mysql import get_db_pool
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO evaluation_datasets (name, client_id, qa_pairs, sample_size)
                VALUES (%s, %s, %s, %s)
                """,
                (name, client_id, json.dumps(qa_pairs), len(qa_pairs))
            )
            await conn.commit()
            dataset_id = cursor.lastrowid
    
    logger.info(
        f"Saved evaluation dataset",
        extra={"dataset_id": dataset_id, "name": name, "size": len(qa_pairs)}
    )
    
    return dataset_id
