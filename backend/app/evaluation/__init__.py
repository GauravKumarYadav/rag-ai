"""RAG Evaluation Framework.

Provides:
- Auto-generation of Q&A pairs from documents
- Retrieval quality metrics (precision, recall, MRR)
- Scheduled evaluation runs with cron
"""

from app.evaluation.generator import generate_qa_pairs
from app.evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_faithfulness,
)
from app.evaluation.runner import run_evaluation
from app.evaluation.scheduler import start_evaluation_scheduler, stop_evaluation_scheduler

__all__ = [
    "generate_qa_pairs",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_mrr",
    "compute_faithfulness",
    "run_evaluation",
    "start_evaluation_scheduler",
    "stop_evaluation_scheduler",
]
