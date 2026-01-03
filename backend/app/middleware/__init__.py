"""Middleware module for request processing."""

from app.middleware.audit import AuditMiddleware

__all__ = ["AuditMiddleware"]
