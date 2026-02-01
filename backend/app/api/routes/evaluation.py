"""
Evaluation API routes for RAG evaluation.

Provides endpoints for creating and managing evaluation datasets and runs.
"""

import json
import uuid
from datetime import datetime
from typing import List, Optional

import redis
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.auth.dependencies import require_superuser
from app.config import settings
from app.core.logging import get_logger


router = APIRouter()
logger = get_logger(__name__)

# Redis keys
DATASETS_KEY = "evaluation:datasets"
RUNS_KEY = "evaluation:runs"


def _get_redis() -> Optional[redis.Redis]:
    """Get Redis client for evaluation storage."""
    try:
        r = redis.from_url(settings.redis.url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


# ============================================================
# Models
# ============================================================

class DatasetCreate(BaseModel):
    """Create dataset request model."""
    name: str
    client_id: Optional[str] = None
    sample_size: int = 100


class DatasetResponse(BaseModel):
    """Dataset response model."""
    id: int
    name: str
    client_id: Optional[str] = None
    sample_size: int
    created_at: str
    status: str = "ready"


class DatasetListResponse(BaseModel):
    """Dataset list response model."""
    datasets: List[DatasetResponse]
    total: int


class RunCreate(BaseModel):
    """Create evaluation run request model."""
    dataset_id: int
    k: int = 5


class RunResponse(BaseModel):
    """Run response model."""
    id: int
    dataset_id: int
    dataset_name: str
    k: int
    created_at: str
    status: str
    metrics: Optional[dict] = None


class RunListResponse(BaseModel):
    """Run list response model."""
    runs: List[RunResponse]
    total: int


# ============================================================
# Datasets
# ============================================================

@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    current_user: dict = Depends(require_superuser),
):
    """
    List all evaluation datasets.
    
    Requires admin privileges.
    """
    r = _get_redis()
    
    if r is None:
        return DatasetListResponse(datasets=[], total=0)
    
    # Get all datasets
    raw_datasets = r.hgetall(DATASETS_KEY)
    
    datasets = []
    for dataset_id, raw in raw_datasets.items():
        try:
            data = json.loads(raw)
            datasets.append(DatasetResponse(**data))
        except Exception:
            continue
    
    # Sort by created_at descending
    datasets.sort(key=lambda x: x.created_at, reverse=True)
    
    return DatasetListResponse(
        datasets=datasets,
        total=len(datasets),
    )


@router.post("/datasets", response_model=DatasetResponse)
async def create_dataset(
    request: DatasetCreate,
    current_user: dict = Depends(require_superuser),
):
    """
    Create a new evaluation dataset.
    
    Requires admin privileges.
    """
    r = _get_redis()
    
    if r is None:
        raise HTTPException(
            status_code=503,
            detail="Redis not available for evaluation storage",
        )
    
    # Generate incremental ID
    dataset_id = r.incr("evaluation:dataset_counter")
    
    dataset = {
        "id": dataset_id,
        "name": request.name,
        "client_id": request.client_id,
        "sample_size": request.sample_size,
        "created_at": datetime.utcnow().isoformat(),
        "status": "ready",
    }
    
    r.hset(DATASETS_KEY, str(dataset_id), json.dumps(dataset))
    
    logger.info(f"Created evaluation dataset: {request.name} (ID: {dataset_id})")
    
    return DatasetResponse(**dataset)


# ============================================================
# Runs
# ============================================================

@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    current_user: dict = Depends(require_superuser),
):
    """
    List all evaluation runs.
    
    Requires admin privileges.
    """
    r = _get_redis()
    
    if r is None:
        return RunListResponse(runs=[], total=0)
    
    # Get all runs
    raw_runs = r.hgetall(RUNS_KEY)
    
    runs = []
    for run_id, raw in raw_runs.items():
        try:
            data = json.loads(raw)
            runs.append(RunResponse(**data))
        except Exception:
            continue
    
    # Sort by created_at descending
    runs.sort(key=lambda x: x.created_at, reverse=True)
    
    return RunListResponse(
        runs=runs,
        total=len(runs),
    )


@router.post("/runs", response_model=RunResponse)
async def create_run(
    request: RunCreate,
    current_user: dict = Depends(require_superuser),
):
    """
    Create and execute a new evaluation run.
    
    Requires admin privileges.
    """
    r = _get_redis()
    
    if r is None:
        raise HTTPException(
            status_code=503,
            detail="Redis not available for evaluation storage",
        )
    
    # Get dataset
    raw_dataset = r.hget(DATASETS_KEY, str(request.dataset_id))
    if not raw_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = json.loads(raw_dataset)
    
    # Generate incremental ID
    run_id = r.incr("evaluation:run_counter")
    
    run = {
        "id": run_id,
        "dataset_id": request.dataset_id,
        "dataset_name": dataset["name"],
        "k": request.k,
        "created_at": datetime.utcnow().isoformat(),
        "status": "completed",
        "metrics": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "note": "Evaluation metrics placeholder - implement actual RAG evaluation",
        },
    }
    
    r.hset(RUNS_KEY, str(run_id), json.dumps(run))
    
    logger.info(f"Created evaluation run for dataset: {dataset['name']} (Run ID: {run_id})")
    
    return RunResponse(**run)
