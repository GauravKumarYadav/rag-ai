PYTHON ?= python3
VENV ?= ../.venv/bin/python
export PYTHONPATH := backend

.PHONY: dev up down logs ps test lint install clean help

help:
	@echo "Available commands:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make dev          - Start backend development server"
	@echo "  make up           - Start all services (Docker)"
	@echo "  make down         - Stop all services"
	@echo "  make logs         - View logs"
	@echo "  make ps           - Show running services"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Check for syntax errors"
	@echo "  make clean        - Clean up data"

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -r backend/requirements.txt
	@echo "Dependencies installed. Activate with: source .venv/bin/activate"

dev:
	cd backend && $(VENV) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

up:
	docker-compose up -d --build
	@echo ""
	@echo "Services started:"
	@echo "  Frontend: http://localhost"
	@echo "  API:      http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo "  ChromaDB: http://localhost:8020"

down:
	docker-compose down

logs:
	docker-compose logs -f

ps:
	docker-compose ps

test:
	cd backend && $(VENV) -m pytest tests/ -v

lint:
	cd backend && $(PYTHON) -m compileall app

clean:
	docker-compose down -v
	rm -rf data/bm25/*
	rm -rf backend/__pycache__ backend/app/__pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
