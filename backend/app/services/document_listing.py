"""
Document listing helpers.

Provides a shared way to list available documents with client isolation.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from app.rag.vector_store import get_client_vector_store, get_vector_store

_DEBUG_LOG_PATH = "/Users/g0y01hx/Desktop/personal_work/chatbot/.cursor/debug.log"


def _debug_log(payload: dict) -> None:
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def list_documents(
    client_id: Optional[str] = None,
    include_global: bool = False,
) -> List[Dict[str, Any]]:
    """
    List available documents grouped by source.

    Args:
        client_id: Optional client identifier. If None or "global", uses global store.

    Returns:
        List of document metadata dicts.
    """
    # #region agent log
    _debug_log({
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "H3",
        "location": "document_listing.py:list_documents",
        "message": "Entry",
        "data": {
            "client_id": client_id,
            "include_global": include_global,
        },
        "timestamp": int(time.time() * 1000),
    })
    # #endregion agent log
    if not client_id or client_id == "global":
        scopes = ["global"]
    else:
        scopes = [client_id]
        if include_global:
            scopes.append("global")

    documents: Dict[str, Dict[str, Any]] = {}

    for scope in scopes:
        store = get_vector_store() if scope == "global" else get_client_vector_store(scope)
        all_docs = store.docs.get(include=["metadatas"])

        ids = all_docs.get("ids", [])
        metadatas = all_docs.get("metadatas", [])
        # #region agent log
        _debug_log({
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "H3",
            "location": "document_listing.py:list_documents",
            "message": "Scope fetch",
            "data": {
                "scope": scope,
                "ids_count": len(ids),
                "metadatas_count": len(metadatas),
            },
            "timestamp": int(time.time() * 1000),
        })
        # #endregion agent log
        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if metadatas else {}
            source = metadata.get("source", "Unknown")

            if source not in documents:
                documents[source] = {
                    "id": doc_id,
                    "filename": os.path.basename(source) if source != "Unknown" else "Unknown",
                    "source": source,
                    "chunk_count": 1,
                    "client_id": metadata.get("client_id"),
                    "client_name": metadata.get("client_name"),
                    "uploaded_at": metadata.get("uploaded_at", ""),
                    "scope": scope,
                }
            else:
                documents[source]["chunk_count"] += 1

    # #region agent log
    _debug_log({
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "H3",
        "location": "document_listing.py:list_documents",
        "message": "Exit",
        "data": {
            "total_documents": len(documents),
            "scopes": scopes,
        },
        "timestamp": int(time.time() * 1000),
    })
    # #endregion agent log
    return list(documents.values())
