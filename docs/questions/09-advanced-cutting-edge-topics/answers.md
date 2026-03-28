# 9 — Advanced / Cutting-Edge Topics — Answers

> All answers below are grounded in the **rag-ai** codebase.

---

## Q51. How would you implement a "corrective RAG" (CRAG) pattern?

CRAG adds a fact-checking step where the agent verifies its own output against retrieved evidence before returning it.

**Implementation in our architecture:**

1. **Add a "verification" node** in the LangGraph after `synthesis`:
   ```
   query → retrieval → synthesis → verification → END
   ```

2. **The verification node would:**
   - Take the `response` and `retrieved_chunks` from state.
   - Prompt an LLM: "Does this response accurately reflect the retrieved context? Flag any unsupported claims."
   - If claims are unsupported, either:
     - Remove the unsupported claims and regenerate.
     - Add a disclaimer.
     - Route back to retrieval with a refined query.

3. **Why our architecture supports this easily:**
   - LangGraph’s `StateGraph` allows adding nodes and conditional edges trivially.
   - The `AgentState` already carries both `response` and `retrieved_chunks`, so the verification node has everything it needs.
   - The `@traceable` decorator pattern means the verification step would be automatically traced in LangSmith.

4. **Trade-off:** One additional LLM call per request, increasing latency by ~1-3 seconds. Worth it for high-stakes domains (legal, medical).

---

## Q52. Explain the Self-RAG paradigm.

Self-RAG fine-tunes a model with special retrieval-aware tokens like `[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]` that the model emits during generation to signal when retrieval is needed and whether retrieved passages are relevant/supportive.

**How it differs from our prompt-based agentic RAG:**

| Aspect | Self-RAG | Our System |
|---|---|---|
| Retrieval trigger | Model emits `[Retrieve]` token | `QueryAgent` classifies intent via separate LLM call |
| Relevance check | `[IsRel]` token during generation | Cross-encoder reranker post-retrieval |
| Support check | `[IsSup]` token | Synthesis prompt instruction: "cite sources" |
| Model | Fine-tuned LLM with special tokens | Off-the-shelf LLM (Qwen3-VL, Llama, GPT-4o-mini) |
| Flexibility | Needs model retraining for changes | Prompt-based, instantly changeable |
| Latency | Lower (single model, inline decisions) | Higher (multi-agent, multiple LLM calls) |

**Our prompt-based approach is more practical** because:
- We support multiple LLM providers (LM Studio, Ollama, Groq, OpenAI) without retraining.
- The multi-agent pipeline is inspectable via LangSmith traces.
- Changes require prompt edits, not model fine-tuning.

---

## Q53. How do you build a graph-augmented RAG system?

Our system has a **knowledge graph module** already built (`backend/app/knowledge/`):

**1. Graph Storage (`graph_store.py`):**
- Uses **SQLite** for per-client isolated knowledge graphs.
- Stores entities, relationships, and mentions:
  ```python
  class EntityType:
      PERSON, ORGANIZATION, DOCUMENT, DATE, AMOUNT, PRODUCT, LOCATION, OTHER
  
  class RelationType:
      AUTHORED, REFERENCES, WORKS_FOR, DATED, OWNS, MENTIONS, RELATED_TO, SIGNED, PAID, RECEIVED
  ```

**2. Entity Extraction (`entity_extractor.py` — 13.5 KB):**
- Uses the LLM for NER (Named Entity Recognition) to extract entities from chunks.
- Entities are stored in the graph with links to their source chunks.

**3. Graph Querying (`graph_query.py` — 11.3 KB):**
- Supports traversal queries across the knowledge graph.
- `kg_expansion_depth: int = 2` controls how many hops the graph traversal explores.

**4. Integration with vector search:**
- Chunk metadata includes KG-ready fields:
  ```python
  detected_entities: List[Dict[str, str]]  # [{type: "PERSON", value: "John"}]
  entity_ids: List[str]                     # Links to KG nodes
  semantic_type: Optional[str]              # fact, definition, procedure
  ```
- Enable with: `RAG__KNOWLEDGE_GRAPH_ENABLED=true`

**5. Designed for scale:** SQLite now, with a clear path to Neo4j later (entity/relationship model is graph-native).

---

## Q54. What is Agentic Retrieval with Planning (ARP)?

ARP is a pattern where the agent creates an explicit retrieval plan before executing any searches, specifying:
- What information is needed.
- Which sources to query.
- In what order.
- What the stopping criteria are.

**In our system:**
- The `QueryAgent` performs lightweight planning: intent classification + query rewriting.
- The `Orchestrator._route_after_query()` executes the plan via conditional edges.
- But there’s no explicit multi-step retrieval plan.

**Planning horizon vs. quality vs. latency:**

| Planning Horizon | Quality | Latency | Our System |
|---|---|---|---|
| None (direct retrieve) | Baseline | Low | — |
| 1-step (classify + route) | Good | Low-Medium | ✅ Current approach |
| Multi-step (decompose + sequence) | High | High | Future via LangGraph edges |
| Full plan (reasoning chain) | Highest | Very High | Not yet needed |

Our 1-step planning is appropriate for our query complexity. Multi-step planning would be warranted for complex analytical queries requiring data from multiple documents.

---

## Q55. How would you integrate RLHF to improve retrieval decisions?

RLHF (Reinforcement Learning from Human Feedback) could improve our system in two areas:

**1. Query rewriting quality:**
- Collect (original_query, rewritten_query, retrieval_quality_score) triples.
- Fine-tune the query rewriting model to maximize retrieval quality.
- Our LangSmith traces provide the data: query → rewrite → retrieved chunks → user satisfaction.

**2. Intent classification accuracy:**
- Log misclassifications (user said "question" was classified as "chitchat").
- Use this feedback to update few-shot examples or fine-tune the classifier.

**Practical implementation in our architecture:**
- LangSmith feedback API to collect user ratings on responses.
- Export rated traces as training data.
- Fine-tune the intent classifier on domain-specific examples.
- The multi-provider support means we could switch to a fine-tuned model without architectural changes.

---

## Q56. Describe a collaborative multi-agent RAG system (retriever, critic, synthesizer).

**Our system is already a collaborative multi-agent system!** The roles map directly:

| Role | Our Agent | Responsibility |
|---|---|---|
| **Router/Planner** | `QueryAgent` | Intent classification, query rewriting, tool detection |
| **Retriever** | `RetrievalAgent` | BM25 + vector search, RRF fusion, reranking |
| **Tool Executor** | `ToolAgent` | Calculator, datetime execution |
| **Synthesizer** | `SynthesisAgent` | Response generation with citations |
| **Coordinator** | `Orchestrator` | LangGraph workflow, conditional routing |

**Adding an explicit "critic" agent:**
1. Add a `CriticAgent` node in the LangGraph between retrieval and synthesis.
2. The critic evaluates retrieved chunks: Are they relevant? Sufficient? Conflicting?
3. If insufficient, route back to retrieval with a refined query.
4. The `AgentState` would carry a `critique` field for the synthesis agent to consider.

The LangGraph `StateGraph` makes this a ~20-line change:
```python
graph.add_node("critic", self._critic_node)
graph.add_conditional_edges("critic", self._route_after_critic, {...})
```

---

## Q57. How do you approach guardrails and safety layers in agentic RAG?

Our safety layers:

**1. Input safety:**
- `SafeCalculator` rejects suspicious patterns: `['import', 'eval', 'exec', 'compile', '__', 'lambda', ';']`
- No `eval()` anywhere in the codebase — shunting-yard algorithm for math.
- Intent validation: only 5 valid intents accepted.

**2. Data access safety:**
- JWT authentication on all endpoints.
- Per-client collection isolation (physical separation, not filter-based).
- Client ID validation on document operations.

**3. Output safety:**
- System prompt instructs: "Be concise but complete" — no encouragement to speculate.
- Citation requirement forces grounding.
- `max_tokens = 4096` prevents unbounded generation.

**4. Operational safety:**
- Circuit breaker prevents cascade failures.
- Timeouts on LLM calls (`timeout = 120.0`).
- Error boundaries in every agent.
- Reranker semaphore prevents CPU exhaustion.

**5. Logging and audit:**
- All actions traced in LangSmith.
- Structured JSON logging with correlation IDs.

---

## Q58. What is the role of embedding model fine-tuning in agentic RAG?

Fine-tuning an embedding model on domain-specific data improves retrieval by learning domain-specific semantic relationships.

**In our system:**
- Default model: `text-embedding-nomic-embed-text-v1.5` (768 dimensions).
- This is a general-purpose model. For domain-specific corpora (legal, medical, financial), fine-tuning with contrastive learning on (query, relevant_document) pairs would improve retrieval precision.

**Our infrastructure supports this:**
1. The `embedding_model` is configurable: `RAG__EMBEDDING_MODEL=my-fine-tuned-model`.
2. Embedding fingerprinting detects model changes and flags re-indexing needs.
3. The `VectorStoreBase` abstract interface decouples embedding from storage.
4. Multiple embedding providers (LMStudio, Ollama, Groq, dedicated microservice) are supported.

**When to fine-tune vs. use off-the-shelf:**
- Off-the-shelf: General-purpose queries, diverse document types (our current use case).
- Fine-tuned: Highly specialized vocabulary, domain-specific relationships, when retrieval precision is critical.

---

## Q59. How would you implement speculative retrieval (pre-fetching)?

Speculative retrieval pre-fetches documents the agent is likely to need based on conversation context.

**Implementation in our architecture:**

1. **Conversation summary analysis:** After each turn, the `ConversationSummarizer` produces a summary. A background task could analyze this summary to predict likely next queries.

2. **Pre-fetch into Redis cache:**
   ```python
   # After post_turn processing:
   likely_queries = predict_next_queries(conversation_summary)
   for query in likely_queries:
       results = await retrieval_agent.process(query)
       cache.set(f"prefetch:{query_hash}", results, ttl=300)
   ```

3. **Cache check before retrieval:** In `RetrievalAgent.process()`, check the prefetch cache before hitting the vector store.

4. **Our Redis infrastructure supports this:** `cache_ttl: int = 3600` is already configured for caching.

**Trade-off:** Speculative retrieval consumes resources on predictions that may be wrong. Best suited for high-traffic systems where the cache hit rate justifies the overhead.

---

## Q60. How do you balance parametric knowledge with retrieved knowledge?

**Our system’s approach:**

1. **Retrieved knowledge takes priority:** The synthesis prompt says:
   > *"Answer based on the provided context when available."*

2. **Parametric knowledge as fallback:**
   > *"When no relevant context is found, respond based on your general knowledge but clarify that it’s not from the documents."*

3. **Chitchat uses parametric only:** When `intent == "chitchat"`, the system uses `CHITCHAT_SYSTEM_PROMPT` with no retrieval, relying entirely on the LLM’s parametric memory.

4. **Explicit labeling:** The synthesis agent distinguishes between document-sourced and general-knowledge responses, so users know the provenance of each claim.

5. **When to prefer parametric:**
   - Greetings, small talk, general knowledge questions.
   - When retrieval returns no relevant results (`retrieved_chunks == []`).
   - When the query is about the system itself ("What can you do?").

6. **When to prefer retrieved:**
   - Domain-specific questions.
   - Questions about specific documents or policies.
   - Anything where accuracy matters more than fluency.

The `QueryAgent`’s intent classification is the decision boundary: `chitchat` → parametric, `question/follow_up` → retrieval-first.
