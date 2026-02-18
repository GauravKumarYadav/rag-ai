PYTHON ?= python3
VENV ?= ../.venv/bin/python
export PYTHONPATH := backend

.PHONY: dev up down logs ps test lint install clean help frontend-dev frontend-build frontend-install prod-up prod-down prod-logs

help:
	@echo "Available commands:"
	@echo "  make install           - Install Python dependencies"
	@echo "  make dev               - Start backend development server"
	@echo "  make frontend-install  - Install frontend dependencies"
	@echo "  make frontend-dev      - Start frontend Vite dev server"
	@echo "  make frontend-build    - Build frontend for production"
	@echo "  make up                - Start all services (Docker)"
	@echo "  make down              - Stop all services"
	@echo "  make logs              - View logs"
	@echo "  make ps                - Show running services"
	@echo "  make test              - Run tests"
	@echo "  make lint              - Check for syntax errors"
	@echo "  make clean             - Clean up data"
	@echo "  make prod-up           - Start production stack"
	@echo "  make prod-down         - Stop production stack"
	@echo "  make prod-logs         - View production logs"

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -r backend/requirements.txt
	@echo "Dependencies installed. Activate with: source .venv/bin/activate"

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

dev:
	cd backend && $(VENV) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

up:
	docker-compose up -d --build
	@echo ""
	@echo "Services started:"
	@echo "  Frontend: http://localhost:81"
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

prod-up:
	docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build
	@echo ""
	@echo "Production services started on port 80"

prod-down:
	docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml down

prod-logs:
	docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml logs -f
