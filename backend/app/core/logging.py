"""Structured JSON logging with correlation IDs and file rotation.

This module provides:
- JSON-formatted logs for Loki/ELK ingestion
- Correlation ID tracking across requests
- Configurable file rotation with retention
- Context-aware logging with request metadata

Usage:
    from app.core.logging import get_logger, setup_logging
    
    # Setup at app startup
    setup_logging()
    
    # Get logger in any module
    logger = get_logger(__name__)
    logger.info("Processing request", extra={"user_id": "123", "action": "chat"})
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID in context."""
    correlation_id_var.set(correlation_id)


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging.
    
    Produces JSON lines compatible with Loki, ELK, and other log aggregators.
    Includes correlation ID, timestamp, level, and structured metadata.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON line."""
        import json
        
        # Base log structure
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation ID if present
        correlation_id = get_correlation_id()
        if correlation_id:
            log_data["correlation_id"] = correlation_id
        
        # Add source location for errors
        if record.levelno >= logging.WARNING:
            log_data["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields (user_id, action, client_id, etc.)
        # Skip internal logging attributes
        skip_fields = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "message", "taskName",
        }
        
        for key, value in record.__dict__.items():
            if key not in skip_fields and not key.startswith("_"):
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to inject and propagate correlation IDs.
    
    - Reads X-Correlation-ID header from incoming requests
    - Generates new UUID if not present
    - Sets correlation ID in context for logging
    - Adds correlation ID to response headers
    """
    
    HEADER_NAME = "X-Correlation-ID"
    
    async def dispatch(self, request: Request, call_next):
        # Get or generate correlation ID
        correlation_id = request.headers.get(self.HEADER_NAME)
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Set in context for logging
        set_correlation_id(correlation_id)
        
        # Process request
        response = await call_next(request)
        
        # Add to response headers
        response.headers[self.HEADER_NAME] = correlation_id
        
        # Clear context
        correlation_id_var.set(None)
        
        return response


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that automatically includes correlation ID."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add correlation ID to extra fields."""
        extra = kwargs.get("extra", {})
        
        # Add correlation ID if not already present
        correlation_id = get_correlation_id()
        if correlation_id and "correlation_id" not in extra:
            extra["correlation_id"] = correlation_id
        
        kwargs["extra"] = extra
        return msg, kwargs


# Module-level logger cache
_loggers: Dict[str, ContextLogger] = {}


def get_logger(name: str) -> ContextLogger:
    """Get a context-aware logger for the given module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        ContextLogger that includes correlation ID automatically
    """
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = ContextLogger(logger, {})
    return _loggers[name]


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "./logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 30,  # 30 days
    json_format: bool = True,
) -> None:
    """Configure structured logging for the application.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log filename (default: app.log)
        log_dir: Directory for log files
        max_bytes: Max size before rotation (default 10MB)
        backup_count: Number of backup files to keep (default 30)
        json_format: Use JSON formatting (default True)
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    # Console handler (always add)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file is None:
        log_file = "app.log"
    
    file_path = log_path / log_file
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    # Log startup
    logger = get_logger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "log_level": log_level,
            "log_file": str(file_path),
            "json_format": json_format,
            "backup_count": backup_count,
        }
    )


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None,
    client_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Log an HTTP request with structured data.
    
    Convenience function for logging requests in a consistent format.
    """
    logger = get_logger("http.request")
    
    extra = {
        "http_method": method,
        "http_path": path,
        "http_status": status_code,
        "duration_ms": round(duration_ms, 2),
    }
    
    if user_id:
        extra["user_id"] = user_id
    if client_id:
        extra["client_id"] = client_id
    if error:
        extra["error"] = error
    
    if status_code >= 500:
        logger.error(f"{method} {path} {status_code}", extra=extra)
    elif status_code >= 400:
        logger.warning(f"{method} {path} {status_code}", extra=extra)
    else:
        logger.info(f"{method} {path} {status_code}", extra=extra)


def log_rag_retrieval(
    query: str,
    num_results: int,
    duration_ms: float,
    client_id: Optional[str] = None,
) -> None:
    """Log a RAG retrieval operation."""
    logger = get_logger("rag.retrieval")
    logger.info(
        "RAG retrieval completed",
        extra={
            "query_length": len(query),
            "num_results": num_results,
            "duration_ms": round(duration_ms, 2),
            "client_id": client_id,
        }
    )


def log_llm_request(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_ms: float,
    streaming: bool = False,
) -> None:
    """Log an LLM API request."""
    logger = get_logger("llm.request")
    logger.info(
        "LLM request completed",
        extra={
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "duration_ms": round(duration_ms, 2),
            "streaming": streaming,
        }
    )
