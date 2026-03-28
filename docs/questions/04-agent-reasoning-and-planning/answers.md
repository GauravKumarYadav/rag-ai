# 4 — Agent Reasoning & Planning — Answers

> All answers below are grounded in the **rag-ai** codebase.

---

## Q21. How do you implement a "reflection" step where the agent evaluates whether its retrieved context is sufficient?

In our system, reflection is implemented **implicitly** in the `SynthesisAgent`:

1. **Context sufficiency check:** The `SynthesisAgent._build_context()` method limits context to 6 chunks, each truncated to 600 chars. If `retrieved_chunks` is empty, the user prompt explicitly states:
   ```python
   if not context:
       prompt_parts.append("No relevant documents found.")
   ```

2. **System prompt instruction:** The synthesis prompt includes:
   > *"If the context doesn’t contain enough information, say so clearly."*
   > *"When no relevant context is found, respond based on your general knowledge but clarify that it’s not from the documents."*

3. **Source attribution as implicit reflection:** By requiring `[Source: filename]` citations, the agent is forced to verify its claims against retrieved evidence. If it can’t cite a source, the prompt guides it to admit the gap.

**For more explicit reflection,** the architecture supports adding a "reflection" node in the LangGraph between retrieval and synthesis that evaluates relevance scores. The `RetrievalHit.score` and `rerank_score` metadata fields are already available for this.

---

## Q22. Describe how you would build an agent that can decide at runtime whether to retrieve, call an API, query a SQL database, or respond from parametric memory.

This is **exactly what our system does** (minus SQL). The `QueryAgent.process()` makes runtime decisions:

```python
async def process(self, state):
    # Step 1: Classify intent via LLM
    intent, needs_retrieval = await self._classify_intent(message)
    
    # Step 2: Detect tools (overrides LLM classification)
    tool_name, tool_params = self._detect_tool(message)
    if tool_name:
        intent = "tool"
        needs_retrieval = False
    
    # Step 3: Rewrite query if retrieval needed
    if needs_retrieval and intent in ("question", "follow_up"):
        rewritten_query = await self._rewrite_query(message)
```

The `Orchestrator._route_after_query()` then routes:

| Decision | Route | Example |
|---|---|---|
| `tool_name` set | `ToolAgent` | "What is 15% of 200?" |
| `intent == "chitchat"` | `SynthesisAgent` (parametric memory) | "Hello!" |
| `intent == "document_list"` | `_document_list_node` (metadata API) | "What documents do I have?" |
| `needs_retrieval == True` | `RetrievalAgent` → `SynthesisAgent` | "What is the refund policy?" |

Adding SQL: Create a `sql_query` tool in `_detect_tool()`, add a SQL executor to `ToolAgent`, and the existing routing handles the rest.

---

## Q23. What is chain-of-thought prompting in an agentic context, and how do you ensure the reasoning trace stays grounded?

Chain-of-thought (CoT) prompting asks the LLM to show its reasoning steps before answering.

**In our system, grounding is ensured through:**

1. **Structured agent pipeline:** Instead of one big CoT prompt, reasoning is decomposed across agents:
   - `QueryAgent` outputs structured JSON (`{"intent": "...", "needs_retrieval": ...}`) — explicit reasoning about query type.
   - `RetrievalAgent` adds source metadata to every chunk.
   - `SynthesisAgent` is instructed to cite sources: *"Cite sources using [Source: filename] format."*

2. **LangSmith tracing:** Every agent step is traced with `@traceable` decorators:
   ```python
   @traceable(name="agent.query")
   @traceable(name="agent.retrieval")
   @traceable(name="agent.synthesis")
   ```
   This provides a full reasoning trace without embedding it in the prompt.

3. **Low temperature for classification:** The `QueryAgent` uses `temperature=0.1` for deterministic intent classification, preventing hallucinated reasoning.

4. **Context window management:** The synthesis prompt is structured as `[conversation context] + [document context] + [tool results] + [question]`, keeping the reasoning chain anchored to evidence at every step.

---

## Q24. How do you prevent an agent from entering infinite retrieval loops?

Our system prevents this through several mechanisms:

1. **`max_steps` bound:** `settings.agent.max_steps = 10` limits the total number of graph traversals.

2. **DAG structure (no cycles):** The current LangGraph is a **directed acyclic graph** — there are no edges from synthesis back to query or retrieval. The flow is strictly:
   ```
   query → [retrieval|tool|document_list] → synthesis → END
   ```
   This makes infinite loops structurally impossible in the current design.

3. **Intent-based routing:** The `_route_after_query()` function always terminates at one of: `"tool"`, `"retrieval"`, `"synthesis"`, or `"document_list"`. There’s no "retry" option.

4. **Exception handling:** The `Orchestrator.run()` wraps the entire graph execution in a try/except, returning a fallback response on any error:
   ```python
   except Exception as e:
       return "I encountered an error processing your request.", []
   ```

**If we add iterative retrieval in the future,** the `max_steps` guard and a "diminishing returns" check (comparing new results to previous results) would prevent loops.

---

## Q25. Explain "tool selection" in agentic RAG.

Tool selection is how the agent decides which tool to invoke and with what parameters.

**In our system, tool selection happens in two layers:**

**Layer 1 — Pattern matching (`QueryAgent._detect_tool()`):**
```python
def _detect_tool(self, message):
    # Calculator: regex for math expressions
    math_pattern = r'[\d\.\s\+\-\*\/\(\)\^%]+'
    # Percentage: "15% of 200"
    percent_pattern = r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)'
    # DateTime: keyword detection
    datetime_keywords = ["date", "time", "today", "now"]
```

**Layer 2 — LLM classification (`_classify_intent()`):**
The intent classifier can output `"tool"` intent, but the explicit pattern matching overrides it for reliability:
```python
if tool_name:
    intent = "tool"
    needs_retrieval = False
```

**Tool execution in `ToolAgent`:**
- `SafeCalculator`: Uses shunting-yard algorithm for secure math evaluation (no `eval()`).
- `datetime`: Returns current date/time.

The two-layer approach ensures:
- **Speed:** Pattern matching is O(n) regex, no LLM call needed.
- **Safety:** `SafeCalculator` rejects suspicious input (`import`, `eval`, `exec`, `__`).
- **Fallback:** If pattern matching fails, the LLM can still classify as `tool` intent.

---

## Q26. How do you handle conflicting information from multiple sources?

Our system retrieves from **two collections** (client + global) and must handle conflicts:

**Current approach:**
1. **Client priority:** `RetrievalAgent._merge_results()` adds client results **first**, then global results. Client-specific documents take precedence.
2. **Deduplication by chunk ID:** Prevents the same chunk from appearing twice:
   ```python
   seen_ids = set()
   for hit in client_results:
       if hit.id not in seen_ids:
           seen_ids.add(hit.id)
           hit.metadata["collection_type"] = "client"
           merged.append(hit)
   ```
3. **Source attribution:** The synthesis prompt requires `[Source: filename]` citations, so the user can see which source each claim comes from.
4. **Multiple source context:** The synthesis agent receives up to 6 chunks with source labels, allowing it to present conflicting views: *"According to [Source: policy-2024.pdf], X. However, [Source: policy-2023.pdf] states Y."*

**What could be improved:**
- Timestamp-based recency filtering (prefer newer documents).
- Confidence-weighted synthesis based on rerank scores.
- Explicit conflict detection in the synthesis prompt.

---

## Q27. What techniques do you use to give the agent "memory" of prior interactions?

Our memory system is implemented in `backend/app/memory/` with multiple layers:

**1. Session Buffer (`session_buffer.py`):**
- Stores recent messages in Redis lists with 24h TTL.
- Uses a **circuit breaker pattern** (`RedisCircuitBreaker`) with automatic fallback to in-memory storage when Redis is down.
- LRU eviction for memory-bounded fallback.

**2. Auto-Summarization (`summarizer.py`):**
- When conversation tokens exceed `max_context_tokens = 4000`, older messages are summarized via LLM.
- The summary is stored in Redis and passed to agents as `conversation_summary`.
- Summarization preserves: key facts, decisions, current topic, specific details, user’s goal.

**3. Sliding Window:**
- `sliding_window_size = 10` recent messages are kept in full.
- Older messages are compressed into the summary.

**4. Memory flow through agents:**
```python
# From AgentState
conversation_summary: str       # Summarized history
recent_messages: List[Dict]     # Recent sliding window
```

**5. Conversation context in synthesis:**
```python
def _build_conversation_context(self, state):
    if summary:
        parts.append(f"Summary: {summary}")
    if recent:
        recent_text = "\n".join([...for msg in recent[-3:]])
```

**6. Query rewriting with context:** The `QueryAgent._rewrite_query()` receives conversation summary to resolve references like "tell me more about that" into standalone queries.

**Storage layout (from ARCHITECTURE.md):**

| Data | Storage | TTL |
|---|---|---|
| Session messages | Redis list | 24h |
| Conversation summary | Redis string | 24h |
| Session metadata | Redis hash | 24h |
