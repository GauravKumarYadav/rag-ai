PYTHON ?= python3
VENV ?= ../.venv/bin/python
export PYTHONPATH := backend

.PHONY: dev frontend ingest test docker-up docker-down lint install clean help

help:
	@echo "Available commands:"
	@echo "  make install     - Install Python dependencies"
	@echo "  make dev         - Start backend development server"
	@echo "  make frontend    - Serve frontend (open in browser)"
	@echo "  make ingest      - Ingest documents from data/raw"
	@echo "  make test        - Run tests"
	@echo "  make docker-up   - Start all services with Docker"
	@echo "  make docker-down - Stop Docker services"
	@echo "  make lint        - Check for syntax errors"
	@echo "  make clean       - Clean up generated files"

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

lint:
	cd backend && $(PYTHON) -m compileall app

clean:
	rm -rf data/chroma/*
	rm -rf backend/__pycache__ backend/app/__pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

