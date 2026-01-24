"""
Natural response templates shared across pipelines.
"""

import random
import re
from typing import Any, Dict, List, Optional


CONVERSATIONAL_RESPONSES = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! What would you like to know?",
        "Hey! I'm here to help. What can I assist you with?",
    ],
    "thanks": [
        "You're welcome! Let me know if you need anything else.",
        "Happy to help! Want to look at another document or question?",
    ],
    "farewell": [
        "Goodbye! Feel free to come back anytime.",
        "Take care! I'm here if you need more help.",
    ],
    "acknowledgment": [
        "Got it! What would you like to do next?",
        "Understood. How can I help you further?",
    ],
    "help": [
        "I can help you search and summarize your documents, answer questions, "
        "and handle calculations or date/time queries. What would you like to do?",
    ],
}


def _pick_response(responses: List[str]) -> str:
    return random.choice(responses)


def get_conversational_response(message: str) -> str:
    """
    Generate a natural conversational response for simple messages.
    """
    query_lower = message.lower().strip()

    if re.match(r"^(hi|hello|hey|howdy|greetings|yo)[\s!?.]*$", query_lower, re.I):
        return _pick_response(CONVERSATIONAL_RESPONSES["greeting"])
    if re.match(r"^(thanks|thank you|thx|ty|appreciate).*$", query_lower, re.I):
        return _pick_response(CONVERSATIONAL_RESPONSES["thanks"])
    if re.match(r"^(bye|goodbye|see you|later|farewell).*$", query_lower, re.I):
        return _pick_response(CONVERSATIONAL_RESPONSES["farewell"])
    if re.match(r"^(yes|no|ok|okay|sure|alright|got it)[\s!?.]*$", query_lower, re.I):
        return _pick_response(CONVERSATIONAL_RESPONSES["acknowledgment"])
    if re.match(r"^(help|what can you do|who are you).*$", query_lower, re.I):
        return _pick_response(CONVERSATIONAL_RESPONSES["help"])

    return "I'm here to help. What would you like to know?"


def format_document_list_response(
    documents: List[Dict[str, Any]],
    client_label: Optional[str] = None,
) -> str:
    """
    Format a friendly response listing available documents.
    """
    if not documents:
        return (
            "I don't see any documents yet. You can upload one, and I can summarize "
            "or search it for you."
        )

    filenames: List[str] = []
    seen = set()
    for doc in documents:
        name = doc.get("filename") or doc.get("source") or "Unknown"
        if name not in seen:
            seen.add(name)
            filenames.append(name)

    header = "Here are the documents I can see"
    if client_label:
        header += f" for {client_label}"
    header += f" ({len(filenames)}):"

    lines = [header]
    for idx, name in enumerate(filenames, start=1):
        lines.append(f"{idx}. {name}")

    lines.append("Want me to summarize one or answer a question about it?")
    return "\n".join(lines)


def followup_needs_reference_response() -> str:
    """
    Friendly response when a follow-up lacks references.
    """
    return (
        "I might be missing the exact reference. Which document or topic should I use?"
    )


def clarification_needs_context_response() -> str:
    """
    Friendly response when clarification lacks context.
    """
    return "I might be missing context. Can you share a bit more detail?"


def action_request_fallback_response() -> str:
    """
    Friendly response for action requests without retrieval or tools.
    """
    return (
        "I can help with your documents, calculations, and date/time questions. "
        "What would you like to do?"
    )


def no_relevant_documents_response() -> str:
    """
    Friendly response when no relevant documents are found.
    """
    return (
        "I couldn't find relevant information in the documents for that. "
        "You can try rephrasing, or upload the document you want me to use."
    )


def format_tool_results_response(tool_results: Dict[str, Any]) -> str:
    """
    Format tool results into a natural response.
    """
    if not tool_results:
        return "I couldn't complete that request."

    parts: List[str] = []
    for tool_name, result in tool_results.items():
        if tool_name == "calculator":
            parts.append(f"The result is: {result}")
        elif tool_name == "datetime":
            parts.append(str(result))
        else:
            parts.append(f"{tool_name}: {result}")

    if len(parts) == 1:
        return parts[0]
    return "Here is what I found:\n" + "\n".join(f"- {p}" for p in parts)


def tool_needs_details_response() -> str:
    """
    Friendly response when a tool query lacks parameters.
    """
    return (
        "I can help with that. Please share the exact expression or the dates you "
        "want me to use."
    )


def llm_unavailable_response() -> str:
    """
    Response when the LLM service is unavailable.
    """
    return (
        "I'm having trouble reaching the language model right now. "
        "Please try again once it's running."
    )
