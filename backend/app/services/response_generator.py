"""
LLM-backed response generator for natural conversational replies.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.services.natural_responses import (
    action_request_fallback_response,
    clarification_needs_context_response,
    followup_needs_reference_response,
    format_document_list_response,
    format_tool_results_response,
    get_conversational_response,
    llm_unavailable_response,
    no_relevant_documents_response,
    tool_needs_details_response,
)

RESPONSE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond concisely and naturally. "
    "Use only the provided data. Do not invent document names, numbers, or facts. "
    "If tool results are provided, preserve values exactly."
)


def _dedupe_filenames(documents: Optional[List[Dict[str, Any]]]) -> List[str]:
    if not documents:
        return []
    filenames: List[str] = []
    seen = set()
    for doc in documents:
        name = doc.get("filename") or doc.get("source") or "Unknown"
        if name not in seen:
            seen.add(name)
            filenames.append(name)
    return filenames


def _fallback_response(
    response_type: str,
    user_message: str,
    documents: Optional[List[Dict[str, Any]]] = None,
    tool_results: Optional[Dict[str, Any]] = None,
    client_label: Optional[str] = None,
    needs_details: bool = False,
) -> str:
    if response_type == "document_list":
        return format_document_list_response(documents or [], client_label=client_label)
    if response_type == "clarification":
        return clarification_needs_context_response()
    if response_type == "followup_needs_reference":
        return followup_needs_reference_response()
    if response_type == "action_request":
        return action_request_fallback_response()
    if response_type == "tool_results":
        if needs_details:
            return tool_needs_details_response()
        return format_tool_results_response(tool_results or {})
    if response_type == "no_documents":
        return no_relevant_documents_response()
    return get_conversational_response(user_message)


async def generate_response(
    *,
    lm_client: Any,
    response_type: str,
    user_message: str,
    documents: Optional[List[Dict[str, Any]]] = None,
    tool_results: Optional[Dict[str, Any]] = None,
    client_label: Optional[str] = None,
    errors: Optional[List[str]] = None,
    needs_details: bool = False,
    allow_fallback_templates: bool = False,
) -> str:
    """
    Generate a natural response using the LLM, with optional template fallback.
    """
    if lm_client is None:
        return llm_unavailable_response()

    filenames = _dedupe_filenames(documents)
    payload = {
        "response_type": response_type,
        "user_message": user_message,
        "client_label": client_label,
        "documents": filenames,
        "tool_results": tool_results or {},
        "errors": errors or [],
        "needs_details": needs_details,
    }

    instructions = {
        "chitchat": "Reply warmly to the user's message in 1-2 sentences.",
        "document_list": (
            "List the available documents clearly. If the list is empty, "
            "say you do not see any documents and suggest uploading one."
        ),
        "clarification": "Ask for the missing context in a friendly way.",
        "followup_needs_reference": "Ask which document or topic the user means.",
        "action_request": "Briefly explain what you can do and ask what they want next.",
        "tool_results": (
            "Summarize tool results directly. If errors exist, mention them. "
            "If details are missing, ask for the exact expression or dates."
        ),
        "no_documents": (
            "Explain that no relevant documents were found and suggest rephrasing "
            "or uploading the document."
        ),
        "rejection": "Acknowledge the rejection and ask what they want instead.",
    }

    instruction = instructions.get(response_type, instructions["chitchat"])
    prompt = (
        f"{instruction}\n\n"
        "Data (JSON):\n"
        f"{json.dumps(payload, ensure_ascii=True)}\n\n"
        "Response:"
    )

    try:
        messages = [
            {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        result = await lm_client.chat(messages, stream=False)
        if result and result.strip():
            return result.strip()
    except Exception:
        if allow_fallback_templates:
            return _fallback_response(
                response_type=response_type,
                user_message=user_message,
                documents=documents,
                tool_results=tool_results,
                client_label=client_label,
                needs_details=needs_details,
            )
        return llm_unavailable_response()

    if allow_fallback_templates:
        return _fallback_response(
            response_type=response_type,
            user_message=user_message,
            documents=documents,
            tool_results=tool_results,
            client_label=client_label,
            needs_details=needs_details,
        )

    return llm_unavailable_response()
