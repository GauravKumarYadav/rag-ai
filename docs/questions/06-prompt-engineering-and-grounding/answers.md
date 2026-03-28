# 6 — Prompt Engineering & Grounding — Answers

> All answers below are grounded in the **rag-ai** codebase.

---

## Q34. How do you structure the system prompt for a RAG agent to minimize hallucination while keeping responses natural?

Our system prompt (from `synthesis_agent.py`) is carefully structured:

```python
SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base.
{client_context}

## Response Guidelines:
1. Answer based on the provided context when available
2. Cite sources using [Source: filename] format when using document information
3. If the context doesn't contain enough information, say so clearly
4. Be concise but complete
5. If tool results are provided, incorporate them naturally

## Formatting Guidelines (use Markdown):
- Use **bold** for emphasis and key terms
- Use bullet points or numbered lists for multiple items
- Use tables when comparing data
- Use code blocks with language tags
- Use > blockquotes for important notes
- Use headings (##, ###) to organize longer responses

When no relevant context is found, respond based on your general knowledge
but clarify that it's not from the documents."""
```

**Anti-hallucination techniques in this prompt:**
1. **Grounding first:** Rule 1 says "answer based on the provided context" — context comes before parametric knowledge.
2. **Mandatory citation:** Rule 2 forces the LLM to attribute claims, making unsupported claims visible.
3. **Explicit uncertainty:** Rule 3 instructs the LLM to admit gaps rather than fabricate.
4. **Knowledge source labeling:** The final line distinguishes document-based vs. general knowledge answers.
5. **Client context injection:** `{client_context}` tells the LLM exactly whose documents it’s working with, preventing cross-client confusion.

**Natural tone is preserved by:**
- Not over-constraining the response format.
- Allowing Markdown for rich, readable output.
- Having a separate `CHITCHAT_SYSTEM_PROMPT` for casual conversation: *"Respond naturally to greetings and casual conversation. Keep responses brief and warm."*

---

## Q35. What techniques do you use to enforce citation and attribution?

Our system enforces citation through multiple layers:

**1. System prompt instruction:**
> *"Cite sources using [Source: filename] format when using document information."*

**2. Source metadata in context:**
The `SynthesisAgent._build_context()` prepends source labels to each chunk:
```python
for i, hit in enumerate(retrieved[:6], 1):
    source = hit.metadata.get("source", ...)
    context_parts.append(f"[Source: {source}]\n{content}")
```
This makes source names visible to the LLM in the same format it’s asked to cite.

**3. Structured sources in response:**
The `RetrievalAgent._build_sources()` creates a structured sources list:
```python
source = {
    "id": hit.id,
    "content_preview": hit.content[:200],
    "score": hit.score,
    "source": hit.metadata.get("source"),
    "collection_type": hit.metadata.get("collection_type"),
    "page": hit.metadata.get("page_number"),
    "section": hit.metadata.get("section_heading"),
}
```
This is returned alongside the response for programmatic citation verification.

**4. Rich chunk metadata:**
`ChunkMetadata` includes `source_filename`, `page_number`, `section_heading` — enabling page-level and section-level citations.

---

## Q36. How do you handle the "lost in the middle" problem?

The "lost in the middle" problem: LLMs tend to focus on content at the beginning and end of the context window, ignoring information in the middle.

**Our mitigations:**

1. **Limiting context size:** We only pass 6 chunks (not 20+), keeping total context short enough that no chunk falls into a "dead zone":
   ```python
   for i, hit in enumerate(retrieved[:6], 1):  # Limit to 6 chunks
   ```

2. **Chunk truncation:** Each chunk is capped at 600 characters:
   ```python
   if len(content) > 600:
       content = content[:600] + "..."
   ```
   Shorter chunks = shorter total context = less middle-loss.

3. **Relevance-ordered context:** Chunks are sorted by rerank score (most relevant first). The most important information is at the beginning, where LLMs attend most strongly.

4. **Structured prompt layout:** The user prompt follows a specific order:
   ```
   1. Conversation context (summary)
   2. Document context (retrieved chunks)
   3. Tool results (if any)
   4. User question (at the end)
   ```
   The question at the end acts as a "recency anchor" that re-focuses the LLM’s attention.

5. **Source delimiters:** Chunks are separated by `---` dividers, making boundaries visually distinct for the LLM:
   ```python
   return "\n\n---\n\n".join(context_parts)
   ```

---

## Q37. Describe your approach to dynamic few-shot example selection.

Our system uses **static few-shot examples** in the intent classification prompt:

```python
INTENT_CLASSIFICATION_PROMPT = """...
Examples:
- "Hello!" → {"intent": "chitchat", "needs_retrieval": false}
- "What is the refund policy?" → {"intent": "question", "needs_retrieval": true}
- "What documents do I have?" → {"intent": "document_list", "needs_retrieval": false}
- "What is 15% of 200?" → {"intent": "tool", "needs_retrieval": false}
..."""
```

**Why static works here:** Intent classification has a small, bounded set of categories (5 intents). The examples cover boundary cases ("What documents contain pricing info?" → `question`, not `document_list`).

**For dynamic few-shot selection (if needed):**
1. Maintain a vector index of (query, correct_intent) pairs.
2. For each new query, embed it and retrieve the top-3 most similar historical examples.
3. Inject those as few-shot examples in the classification prompt.
4. This adapts to domain-specific query patterns without prompt rewriting.

The existing embedding infrastructure (`get_embedding_function()`) and ChromaDB could serve as the few-shot example store.

---

## Q38. How do you guard against prompt injection attacks?

Our system has several defense layers:

**1. Tool safety — `SafeCalculator` (`tool_agent.py`):**
```python
SUSPICIOUS_PATTERNS = ['import', 'eval', 'exec', 'compile', '__', 'lambda', ';']

@classmethod
def _contains_suspicious_patterns(cls, expression):
    # Fast rejection of code injection attempts
```
The calculator uses a **shunting-yard parser** instead of `eval()`, making code injection impossible.

**2. Multi-strategy JSON extraction (`query_agent.py`):**
The intent classifier doesn’t blindly `eval()` LLM output. It uses multi-strategy parsing with fallbacks:
```python
def _extract_intent_with_fallback(self, text):
    # Strategy 1: JSON code blocks
    # Strategy 2: Inline JSON objects 
    # Strategy 3: Pattern matching (regex)
```
Invalid intents are rejected by validation:
```python
valid_intents = {"chitchat", "question", "follow_up", "tool", "document_list"}
if intent not in valid_intents:
    return None
```

**3. Client isolation:** Even if a prompt injection manipulates the query, the retrieval agent is scoped to the authenticated user’s client collections. An attacker can’t access other clients’ documents.

**4. JWT authentication:** All API endpoints require valid JWT tokens (`backend/app/auth/`). Unauthenticated requests never reach the agent pipeline.

**5. Input length limits:** `context[:500]` and `conversation_summary[:300]` in various prompts prevent context stuffing attacks.

**6. Separation of concerns:** The user message is treated as data (in `HumanMessage`), not as system instructions. System prompts are hardcoded, not user-modifiable.
