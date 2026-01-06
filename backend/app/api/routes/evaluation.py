"""
Evaluation API routes for RAG quality assessment.

Endpoints cover dataset generation, running evaluations, and viewing results.
These routes lean on the existing evaluation generator and runner modules.
"""

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user, require_superuser
from app.config import settings
from app.core.logging import get_logger
from app.db.mysql import get_db_pool
from app.evaluation.generator import generate_qa_pairs, save_dataset
from app.evaluation.runner import (
    get_evaluation_run,
    list_evaluation_runs,
    run_evaluation,
)


router = APIRouter(prefix="/evaluation", tags=["evaluation"])
logger = get_logger(__name__)


class GenerateDatasetRequest(BaseModel):
    """Request body for creating a new evaluation dataset."""

    name: str = Field(..., min_length=1, max_length=255)
    client_id: Optional[str] = Field(None, description="Filter docs by client")
    sample_size: Optional[int] = Field(None, ge=1, le=500)


class GenerateDatasetResponse(BaseModel):
    id: int
    name: str
    client_id: Optional[str]
    sample_size: int
    qa_pairs_count: int
    created_at: str


class RunEvaluationRequest(BaseModel):
    dataset_id: int = Field(..., description="Dataset ID to evaluate")
    k: int = Field(5, ge=1, le=20, description="Top-K documents for metrics")


class RunEvaluationResponse(BaseModel):
    run_id: int
    dataset_id: int
    status: str
    message: str


@router.post("/datasets", response_model=GenerateDatasetResponse)
async def create_dataset(
    request: GenerateDatasetRequest,
    current_user: dict = Depends(require_superuser),
):
    """Generate Q&A pairs from documents and persist as a dataset."""

    sample_size = request.sample_size or settings.evaluation.default_sample_size

    qa_pairs = await generate_qa_pairs(
        client_id=request.client_id,
        sample_size=sample_size,
    )

    if not qa_pairs:
        raise HTTPException(status_code=400, detail="No content to generate dataset")

    dataset_id = await save_dataset(
        qa_pairs=qa_pairs,
        name=request.name,
        client_id=request.client_id,
    )

    logger.info(
        "Evaluation dataset created",
        extra={"dataset_id": dataset_id, "qa_pairs": len(qa_pairs), "user": current_user.get("username")},
    )

    return GenerateDatasetResponse(
        id=dataset_id,
        name=request.name,
        client_id=request.client_id,
        sample_size=len(qa_pairs),
        qa_pairs_count=len(qa_pairs),
        created_at="now",
    )


@router.get("/datasets")
async def list_datasets(
    active_only: bool = Query(True, description="Only active datasets"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """List datasets with run counts."""

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            where_clause = "WHERE active = TRUE" if active_only else ""
            await cursor.execute(
                f"""
                SELECT d.id, d.name, d.client_id, d.sample_size, d.qa_pairs, d.active, d.created_at,
                       (SELECT COUNT(*) FROM evaluation_runs r WHERE r.dataset_id = d.id) as run_count
                FROM evaluation_datasets d
                {where_clause}
                ORDER BY d.created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            rows = await cursor.fetchall()
    
    datasets = []
    for row in rows:
        qa_pairs = json.loads(row[4]) if row[4] else []
        datasets.append(
            {
                "id": row[0],
                "name": row[1],
                "client_id": row[2],
                "sample_size": row[3],
                "qa_pairs_count": len(qa_pairs),
                "active": bool(row[5]),
                "created_at": row[6].isoformat() if row[6] else None,
                "run_count": row[7],
            }
        )

    return datasets


@router.delete("/datasets/{dataset_id}")
async def deactivate_dataset(
    dataset_id: int,
    current_user: dict = Depends(require_superuser),
):
    """Soft-delete a dataset."""

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "UPDATE evaluation_datasets SET active = FALSE WHERE id = %s",
                (dataset_id,),
            )
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Dataset not found")

    logger.info("Dataset deactivated", extra={"dataset_id": dataset_id, "user": current_user.get("username")})
    return {"message": "Dataset deactivated"}


@router.post("/runs", response_model=RunEvaluationResponse)
async def start_evaluation(
    request: RunEvaluationRequest,
    current_user: dict = Depends(require_superuser),
):
    """Run evaluation synchronously and return run identifier."""

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT id FROM evaluation_datasets WHERE id = %s AND active = TRUE",
                (request.dataset_id,),
            )
            if not await cursor.fetchone():
                raise HTTPException(status_code=404, detail="Dataset not found or inactive")

    result = await run_evaluation(dataset_id=request.dataset_id, k=request.k)

    return RunEvaluationResponse(
        run_id=result.get("run_id", -1),
        dataset_id=request.dataset_id,
        status="completed",
        message="Evaluation finished",
    )


@router.get("/runs")
async def list_runs(
    dataset_id: Optional[int] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """List evaluation runs."""

    runs = await list_evaluation_runs(dataset_id=dataset_id, limit=limit, offset=offset)
    return runs


@router.get("/runs/{run_id}")
async def get_run(run_id: int, current_user: dict = Depends(get_current_user)):
    """Fetch a single evaluation run."""

    run = await get_evaluation_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
