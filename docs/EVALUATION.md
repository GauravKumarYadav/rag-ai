# RAG Evaluation Guide

This guide covers the RAG evaluation framework for automatically testing retrieval quality and generating performance metrics.

---

## Overview

The evaluation system:

1. **Generates** test Q&A pairs from your document chunks
2. **Runs** retrieval queries and measures quality
3. **Computes** metrics (precision, recall, MRR, faithfulness)
4. **Schedules** daily evaluation runs via cron

---

## Quick Start

### 1. Generate a Test Dataset

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/evaluation/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "your-client-id",
    "name": "my-test-dataset",
    "sample_size": 50
  }'
```

### 2. Run an Evaluation

```bash
curl -X POST http://localhost:8000/api/v1/evaluation/runs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "k": 5
  }'
```

### 3. View Results

```bash
curl http://localhost:8000/api/v1/evaluation/runs \
  -H "Authorization: Bearer $TOKEN"
```

Or use the Admin Dashboard at http://localhost:8000/admin.html

---

## Metrics

### Precision@K

**What it measures:** Of the K documents retrieved, how many are relevant?

$$\text{Precision@K} = \frac{\text{Relevant Documents Retrieved}}{\text{K}}$$

**Good values:** > 0.7 means most retrieved docs are relevant

### Recall@K

**What it measures:** Of all relevant documents, how many were retrieved?

$$\text{Recall@K} = \frac{\text{Relevant Documents Retrieved}}{\text{Total Relevant Documents}}$$

**Good values:** > 0.8 means you're finding most relevant docs

### Mean Reciprocal Rank (MRR)

**What it measures:** How high does the first relevant document rank?

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

**Good values:** > 0.8 means relevant docs appear near the top

### Faithfulness

**What it measures:** Is the generated answer supported by retrieved context?

Uses LLM to check if the answer is grounded in the source documents.

**Good values:** > 0.9 means answers are well-grounded

---

## API Endpoints

All endpoints require authentication and superuser role.

### Generate Dataset

```http
POST /api/v1/evaluation/generate
```

**Request Body:**

```json
{
  "client_id": "string",
  "name": "string",
  "sample_size": 50,
  "chunk_sample_strategy": "random"
}
```

**Response:**

```json
{
  "dataset_id": 1,
  "name": "my-test-dataset",
  "client_id": "client-123",
  "qa_count": 50,
  "created_at": "2026-01-03T12:00:00Z"
}
```

### List Datasets

```http
GET /api/v1/evaluation/datasets
```

**Query Parameters:**
- `client_id` (optional): Filter by client

### Run Evaluation

```http
POST /api/v1/evaluation/runs
```

**Request Body:**

```json
{
  "dataset_id": 1,
  "k": 5
}
```

**Response:**

```json
{
  "run_id": 1,
  "dataset_id": 1,
  "precision_at_k": 0.75,
  "recall_at_k": 0.82,
  "mrr": 0.88,
  "faithfulness": 0.94,
  "k": 5,
  "created_at": "2026-01-03T12:00:00Z"
}
```

### List Evaluation Runs

```http
GET /api/v1/evaluation/runs
```

**Query Parameters:**
- `dataset_id` (optional): Filter by dataset
- `limit` (optional): Max results (default: 50)

---

## Automated Scheduling

Evaluations run daily via APScheduler.

### Configuration

```python
# backend/app/config.py
class EvaluationSettings(BaseModel):
    enabled: bool = True
    schedule_cron: str = "0 2 * * *"  # 2 AM UTC daily
    default_sample_size: int = 50
    default_k: int = 5
    auto_generate_datasets: bool = True
```

### Environment Variables

```bash
EVALUATION_ENABLED=true
EVALUATION_SCHEDULE_CRON="0 2 * * *"
EVALUATION_DEFAULT_SAMPLE_SIZE=50
EVALUATION_DEFAULT_K=5
```

### How Scheduled Runs Work

1. At the configured time, the scheduler triggers
2. For each client with documents in ChromaDB:
   - Check if a recent dataset exists (< 7 days)
   - If not, generate a new dataset
   - Run evaluation with default K
3. Results are stored in MySQL `evaluation_runs` table

---

## Dataset Generation

### How It Works

1. **Sample chunks** from ChromaDB for the client
2. **Send to LLM** with prompt to generate Q&A pairs
3. **Parse response** and validate format
4. **Store** in MySQL `evaluation_datasets` table

### LLM Prompt

The generator uses a structured prompt:

```
Given the following document chunk, generate a question-answer pair 
that tests retrieval quality.

Chunk: {chunk_text}

Requirements:
- Question should be answerable from the chunk
- Answer should be specific and factual
- Avoid yes/no questions

Output format:
Q: [question]
A: [answer]
```

### Chunk Sampling Strategies

| Strategy | Description |
|----------|-------------|
| `random` | Random sample across all chunks |
| `diverse` | Maximize topic diversity |
| `recent` | Prefer recently added chunks |

---

## Database Schema

### evaluation_datasets

```sql
CREATE TABLE evaluation_datasets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    client_id VARCHAR(255) NOT NULL,
    qa_pairs JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255)
);
```

### evaluation_runs

```sql
CREATE TABLE evaluation_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_id INT NOT NULL,
    precision_at_k FLOAT,
    recall_at_k FLOAT,
    mrr FLOAT,
    faithfulness FLOAT,
    k INT DEFAULT 5,
    detailed_results JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES evaluation_datasets(id)
);
```

---

## Admin Dashboard

The admin dashboard at `/admin.html` provides:

### Evaluation Tab

- List all datasets with metadata
- View evaluation run history
- Trigger new evaluations
- Compare metrics across runs

### Charts

- Precision/Recall over time
- MRR trends
- Client comparison

---

## Best Practices

### Dataset Quality

1. **Sample size**: 50-100 Q&A pairs is usually sufficient
2. **Regenerate periodically**: Weekly or when docs change significantly
3. **Review samples**: Manually check generated Q&A quality

### Evaluation Frequency

1. **Daily**: For production systems with active changes
2. **Weekly**: For stable systems
3. **On-demand**: After major document updates

### Interpreting Results

| Scenario | Likely Cause | Action |
|----------|--------------|--------|
| Low precision | Irrelevant docs retrieved | Tune embedding model, improve chunking |
| Low recall | Missing relevant docs | Check indexing, increase K |
| Low MRR | Relevant docs ranked low | Tune reranking, improve embeddings |
| Low faithfulness | Hallucination | Improve prompts, add grounding |

---

## Troubleshooting

### Dataset Generation Fails

1. Check LLM connection: Verify LMStudio is running
2. Check ChromaDB: Ensure client has documents
3. Check logs: `grep "evaluation" ./logs/app.log`

### Metrics All Zero

1. Verify dataset has valid Q&A pairs
2. Check retrieval is working: Test via chat endpoint
3. Ensure correct client_id is used

### Scheduler Not Running

1. Check `EVALUATION_ENABLED=true`
2. Verify cron syntax in config
3. Check APScheduler logs for errors
