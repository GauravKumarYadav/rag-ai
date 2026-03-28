# 1 — Foundations & Architecture — Answers

> All answers below are grounded in the **rag-ai** codebase.

---

## Q1. How does an agentic RAG system differ from a traditional (single-shot) RAG pipeline, and when would you choose one over the other?

**Traditional (single-shot) RAG:** Query → Retrieve → Generate. One pass, no decision-making loop.

**Agentic RAG (our system):** The LLM is wrapped inside an *agent loop* that can observe, think, act, and reflect. In our project the `Orchestrator` (`backend/app/agents/orchestrator.py`) builds a **LangGraph `StateGraph`** with specialized nodes:

1. **Query Agent** — classifies intent, rewrites queries.
2. **Retrieval Agent** — searches client + global collections.
3. **Tool Agent** — executes calculator/datetime tools.
4. **Synthesis Agent** — generates grounded responses.

The graph uses **conditional edges** (`_route_after_query`) to decide at runtime whether to retrieve, call a tool, list documents, or go straight to synthesis. A traditional pipeline cannot make these runtime routing decisions.

**When to choose which:**

- Use single-shot RAG for simple, well-scoped Q&A over a single knowledge base with predictable query types.
- Use agentic RAG (like ours) when queries are heterogeneous (chitchat, tool use, multi-collection retrieval, follow-ups) and require dynamic routing, query rewriting, or multi-step reasoning.

---

## Q2. Walk me through the end-to-end architecture of an agentic RAG system you have built or designed.

Our system's end-to-end flow (from `README.md` and `docs/ARCHITECTURE.md`):

```
User → Frontend (React+Vite) → Nginx → FastAPI (Chat-API)
         │
         ▼
   ChatService.handle_chat()
         │
         ├── Memory: SessionBuffer (Redis) retrieves conversation_summary + recent_messages
         │
         ▼
   Orchestrator.run()  (LangGraph StateGraph)
         │
         ▼
   QueryAgent.process()  →  classify intent, detect tools, rewrite query
         │
         ├─ chitchat ───────────────────────────────┐
         ├─ tool ─── ToolAgent.process() ───────────┤
         ├─ document_list ── _document_list_node() ─┤
         └─ question/follow_up ─── RetrievalAgent  ─┤
              │                                     │
              │ (BM25 + Vector → RRF → Reranker)    │
              │                                     │
              └─────────────────────────────────────┤
                                                    ▼
                                          SynthesisAgent.process()
                                                    │
                                                    ▼
                                            _post_turn()
                                            (update memory, trigger
                                             auto-summarization)
                                                    │
                                                    ▼
                                              Response to user
```

**Key components and why:**


| Component      | Technology                                                     | Why                                                       |
| -------------- | -------------------------------------------------------------- | --------------------------------------------------------- |
| LLM            | LM Studio / Ollama / Groq / OpenAI (pluggable via `config.py`) | Local-first, privacy; multi-provider flexibility          |
| Vector Store   | ChromaDB                                                       | Lightweight, embedded, local — no infrastructure overhead |
| Keyword Search | BM25 (rank_bm25)                                               | Excels at exact keyword matches that vector search misses |
| Fusion         | Reciprocal Rank Fusion                                         | Combines BM25 + vector without score calibration          |
| Reranker       | cross-encoder/ms-marco-MiniLM-L-6-v2                           | More accurate than bi-encoders for final ranking          |
| Memory         | Redis                                                          | Fast session storage with TTL; auto-summarization         |
| Orchestration  | LangGraph (StateGraph)                                         | Stateful DAG with conditional edges for agent routing     |
| Observability  | LangSmith                                                      | Full tracing of every agent step                          |
| Doc Processing | Docling                                                        | Unified PDF/DOCX/Image → Markdown                         |


---

## Q3. What role does the "agent loop" (observe → think → act → reflect) play in improving retrieval quality?

In our system, the agent loop manifests as the **LangGraph state machine**:

1. **Observe:** The `QueryAgent` receives the user message along with `conversation_summary` and `recent_messages` from Redis memory.
2. **Think:** The `QueryAgent` classifies intent (`chitchat`, `question`, `follow_up`, `tool`, `document_list`) and rewrites the query for better retrieval using `_rewrite_query()`.
3. **Act:** Based on the classified intent, the orchestrator's `_route_after_query()` conditionally routes to the appropriate agent (retrieval, tool, synthesis, or document list).
4. **Reflect (implicit):** The synthesis agent evaluates retrieved context — if no relevant documents are found, it explicitly tells the user ("No relevant documents found") rather than hallucinating, guided by the system prompt: *"If the context doesn't contain enough information, say so clearly."*

This loop improves retrieval quality because:

- **Query rewriting** resolves conversational references using conversation context, producing standalone queries that hit the vector store more accurately.
- **Intent classification** prevents unnecessary retrieval calls for chitchat or tool requests, saving latency.
- **Dynamic routing** means the right agent processes the right type of query.

---

## Q4. How do you decide between a single-agent RAG architecture versus a multi-agent orchestration pattern?

Our project uses a **multi-agent pattern** (4 specialized agents + orchestrator). The decision was driven by:

1. **Separation of concerns:** Each agent has a single responsibility (SRP). `QueryAgent` only classifies/rewrites. `RetrievalAgent` only fetches. `SynthesisAgent` only generates. This makes each agent independently testable and maintainable.
2. **Heterogeneous query types:** Our system handles 5 different intents (`chitchat`, `question`, `follow_up`, `tool`, `document_list`). A single agent would need a monolithic prompt covering all cases.
3. **Multi-collection search:** The `RetrievalAgent` must search both client-specific and global collections, merge, deduplicate, and rerank — complex enough to warrant its own agent.
4. **Cost control:** The `QueryAgent` uses `temperature=0.1` and `max_tokens=256` (cheap classification), while the `SynthesisAgent` uses `temperature=0.35` and `max_tokens=4096` (expensive generation). Separating them allows independent tuning.

**When a single agent suffices:** Simple, single-collection Q&A with one query type and no tool use.

---

## Q5. Explain the concept of "tool-augmented retrieval."

In our system, retrieval is one of several *callable tools* the agent can choose from. The `QueryAgent._detect_tool()` method inspects the user message for patterns that indicate tool use (math expressions, datetime keywords). If a tool is detected:

```python
# From query_agent.py
tool_name, tool_params = self._detect_tool(message)
if tool_name:
    intent = "tool"
    needs_retrieval = False
```

The orchestrator then routes to `ToolAgent` instead of `RetrievalAgent`. The `ToolAgent` executes the tool (e.g., `SafeCalculator` using shunting-yard algorithm for safe math evaluation) and passes the result to `SynthesisAgent`.

**How this changes system design:**

- The LLM doesn't compute math itself (avoidance of hallucinated calculations).
- Retrieval is treated as a *conditional* step, not a mandatory one.
- The agent state carries `tool_name`, `tool_params`, and `tool_result` fields alongside `retrieved_chunks`, making tools first-class citizens.
- The synthesis prompt incorporates tool results: *"If tool results are provided, incorporate them naturally into your response."*

---

## Q6. How would you architect an agentic RAG system that serves both structured (SQL) and unstructured (document) sources?

Our current system serves **unstructured documents** (PDF, DOCX, images via Docling). The architecture is already designed for extensibility:

1. **Add a SQL tool to the ToolAgent:** The `ToolAgent` already supports calculator and datetime tools. Adding a `sql_query` tool would follow the same pattern — the `QueryAgent` detects SQL-like intent, sets `tool_name="sql_query"`, and the `ToolAgent` executes it against the database.
2. **Router in the QueryAgent:** The existing intent classifier already distinguishes between `question` (needs retrieval), `tool` (needs computation), and `chitchat`. Adding a `structured_query` intent would route to a SQL agent.
3. **Shared `AgentState`:** The `AgentState` TypedDict (in `state.py`) already has `tool_name`, `tool_params`, and `tool_result` — generic enough to carry SQL query/results.
4. **Synthesis merges both:** The `SynthesisAgent` already merges `context` (from documents) with `tool_context` (from tools). SQL results would flow through `tool_context`.

---

## Q7. What are the trade-offs between plan-then-execute and ReAct-style agent patterns?

Our system uses a **plan-then-execute** pattern (closer to a static DAG):

- The `QueryAgent` does all upfront planning (classify intent, detect tools, rewrite query) in a single step.
- The `Orchestrator._route_after_query()` then routes to exactly one execution path.
- There is no iterative loop where the agent re-plans after seeing retrieval results.

**Trade-offs:**


|                       | Plan-then-Execute (our approach)                      | ReAct (reactive loop)                                    |
| --------------------- | ----------------------------------------------------- | -------------------------------------------------------- |
| **Latency**           | Lower — single planning step, then straight execution | Higher — multiple LLM calls for observe/think/act cycles |
| **Cost**              | Lower — fewer LLM invocations per query               | Higher — each iteration costs tokens                     |
| **Retrieval quality** | May miss if initial retrieval is poor                 | Can self-correct with iterative retrieval                |
| **Complexity**        | Simpler DAG, easier to trace/debug                    | Complex loop, harder to observe                          |
| **Predictability**    | Deterministic routing, bounded steps (`max_steps=10`) | May enter infinite loops without guardrails              |


Our `max_steps = settings.agent.max_steps` (default 10) provides a safety bound, even though the current graph doesn't iterate. This is forward-looking for when we add iterative retrieval.