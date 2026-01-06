PYTHON ?= python3
VENV ?= ../.venv/bin/python
export PYTHONPATH := backend

.PHONY: dev frontend ingest test docker-up docker-down microservices-up microservices-down microservices-logs ollama-pull lint install clean help

help:
	@echo "Available commands:"
	@echo "  make install          - Install Python dependencies"
	@echo "  make dev              - Start backend development server"
	@echo "  make frontend         - Serve frontend (open in browser)"
	@echo "  make ingest           - Ingest documents from data/raw"
	@echo "  make test             - Run tests"
	@echo "  make docker-up        - Start monolithic Docker stack"
	@echo "  make docker-down      - Stop monolithic Docker services"
	@echo "  make microservices-up - Start full microservices stack with Ollama"
	@echo "  make microservices-down - Stop microservices stack"
	@echo "  make microservices-logs - View microservices logs"
	@echo "  make ollama-pull      - Pull required Ollama models"
	@echo "  make lint             - Check for syntax errors"
	@echo "  make clean            - Clean up generated files"

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -r backend/requirements.txt
	@echo "✓ Dependencies installed. Activate with: source .venv/bin/activate"

dev:
	cd backend && $(VENV) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	@echo "Opening frontend at http://localhost:3000"
	cd frontend && $(PYTHON) -m http.server 3000

ingest:
	$(VENV) scripts/ingest_documents.py --input data/raw --db data/chroma

test:
	$(VENV) -m pytest tests/ -v

docker-up:
	docker-compose up -d --build
	@echo "✓ Services started:"
	@echo "  Backend:  http://localhost:8000"
	@echo "  Frontend: http://localhost:3000"
	@echo "  API Docs: http://localhost:8000/docs"

docker-down:
	docker-compose down

# ============================================================
# MICROSERVICES STACK
# ============================================================

microservices-up:
	@echo "Starting microservices stack..."
	docker-compose -f docker-compose.microservices.yml up -d --build
	@echo ""
	@echo "⏳ Waiting for Ollama to be ready..."
	@sleep 10
	@echo "✓ Microservices started:"
	@echo ""
	@echo "  🌐 Frontend:        http://localhost (via nginx)"
	@echo "  🤖 Chat API:        http://localhost:8000"
	@echo "  📄 Document API:    http://localhost:8002"
	@echo "  🔢 Embedding Svc:   http://localhost:8010"
	@echo "  🦙 Ollama:          http://localhost:11434"
	@echo "  🗄️  ChromaDB:        http://localhost:8020"
	@echo "  📊 Grafana:         http://localhost:3001 (admin/admin)"
	@echo "  📈 Prometheus:      http://localhost:9090"
	@echo ""
	@echo "💡 Run 'make ollama-pull' to download required models"

microservices-down:
	docker-compose -f docker-compose.microservices.yml down

microservices-logs:
	docker-compose -f docker-compose.microservices.yml logs -f

microservices-ps:
	docker-compose -f docker-compose.microservices.yml ps

# Pull required Ollama models
ollama-pull:
	@echo "Pulling Ollama models..."
	docker exec rag-ollama ollama pull nomic-embed-text
	docker exec rag-ollama ollama pull llama3.2:1b
	@echo "✓ Models pulled:"
	@echo "  - nomic-embed-text (embeddings)"
	@echo "  - llama3.2:1b (chat, 1.3GB, fast)"
	@echo ""
	@echo "💡 For better quality, also run:"
	@echo "   docker exec rag-ollama ollama pull llama3.2:3b"

# Scale document workers
scale-workers:
	docker-compose -f docker-compose.microservices.yml up -d --scale document-worker=3
	@echo "✓ Scaled to 3 document workers"

lint:
	cd backend && $(PYTHON) -m compileall app

clean:
	rm -rf data/chroma/*
	rm -rf backend/__pycache__ backend/app/__pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

