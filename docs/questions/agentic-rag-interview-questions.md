# Agentic RAG Application Interview Questions

> **Target audience:** Senior Data / AI Engineers with 5+ years of experience  
> **Last updated:** 2026-02-27

---

## 1 — Foundations & Architecture

1. How does an agentic RAG system differ from a traditional (single-shot) RAG pipeline, and when would you choose one over the other?
2. Walk me through the end-to-end architecture of an agentic RAG system you have built or designed. What were the key components and why?
3. What role does the "agent loop" (observe → think → act → reflect) play in improving retrieval quality compared to a static retrieve-then-generate approach?
4. How do you decide between a single-agent RAG architecture versus a multi-agent orchestration pattern?
5. Explain the concept of "tool-augmented retrieval." How does giving an LLM access to retrieval as a callable tool change system design?
6. How would you architect an agentic RAG system that must serve both structured (SQL databases) and unstructured (document) knowledge sources in a single query?
7. What are the trade-offs between a plan-then-execute agent pattern and a fully reactive (ReAct-style) agent pattern for RAG workloads?

---

## 2 — Retrieval Strategies

8. When an agent's first retrieval attempt returns low-relevance results, what self-correction strategies can it employ before responding to the user?
9. Compare dense vector retrieval, sparse keyword retrieval, and hybrid retrieval. In what scenarios would an agent dynamically switch between them?
10. How do you implement adaptive chunking strategies, and how does chunk size impact downstream agent reasoning?
11. Explain query decomposition in agentic RAG. How does an agent break a complex question into sub-queries, and how does it merge the results?
12. What is Hypothetical Document Embedding (HyDE), and when would an agent use it versus a direct embedding lookup?
13. How would you handle multi-hop retrieval — questions where the answer depends on chaining facts across multiple documents?
14. Describe a re-ranking strategy an agent can use after initial retrieval. How do cross-encoders fit into this?
15. What is the role of metadata filtering in retrieval, and how can an agent learn to apply filters dynamically?

---

## 3 — Vector Stores & Indexing

16. How do you choose between vector databases (Pinecone, Weaviate, Qdrant, Milvus) for a production agentic RAG system? What criteria matter most?
17. Explain the difference between HNSW, IVF-PQ, and brute-force ANN indexes. When does the choice materially impact agent performance?
18. How do you handle incremental index updates when documents are added, modified, or deleted without full re-indexing?
19. What strategies do you use for multi-tenancy in a shared vector store while keeping retrieval isolated per tenant?
20. How do you version and roll back an embedding index when your embedding model changes?

---

## 4 — Agent Reasoning & Planning

21. How do you implement a "reflection" step where the agent evaluates whether its retrieved context is sufficient before generating an answer?
22. Describe how you would build an agent that can decide at runtime whether to retrieve from a knowledge base, call an API, query a SQL database, or respond from parametric memory.
23. What is chain-of-thought prompting in an agentic context, and how do you ensure the reasoning trace stays grounded in retrieved evidence?
24. How do you prevent an agent from entering infinite retrieval loops when it repeatedly fails to find relevant information?
25. Explain the concept of "tool selection" in agentic RAG. How does an LLM decide which retrieval tool to invoke and with what parameters?
26. How do you handle conflicting information retrieved from multiple sources? What resolution strategies can the agent employ?
27. What techniques do you use to give the agent a "memory" of prior interactions within a multi-turn RAG conversation?

---

## 5 — Evaluation & Metrics

28. What metrics do you use to evaluate an agentic RAG system beyond simple answer accuracy (e.g., faithfulness, relevance, latency, cost)?
29. How do you measure and reduce hallucination rates in an agentic RAG pipeline?
30. Explain the RAGAS evaluation framework. Which of its metrics are most useful for agentic RAG, and why?
31. How do you build a regression test suite for an agentic RAG system so that model or prompt changes don't silently degrade quality?
32. What is "context precision" vs. "context recall," and how do you instrument your agent to track both?
33. How would you A/B test two different agent strategies (e.g., single retrieval vs. iterative retrieval) in production?

---

## 6 — Prompt Engineering & Grounding

34. How do you structure the system prompt for a RAG agent to minimize hallucination while keeping responses natural?
35. What techniques do you use to enforce citation and attribution in agent-generated answers?
36. How do you handle the "lost in the middle" problem — where LLMs ignore context placed in the middle of a long prompt?
37. Describe your approach to dynamic few-shot example selection for an agentic RAG system.
38. How do you guard against prompt injection attacks that could manipulate the agent's retrieval or generation behavior?

---

## 7 — Scalability & Production

39. How do you manage latency in an agentic RAG system where the agent may perform multiple retrieval rounds before answering?
40. What caching strategies (semantic cache, exact-match cache) do you use to reduce redundant retrievals and LLM calls?
41. How do you implement observability and tracing for a multi-step agentic RAG pipeline in production (e.g., LangSmith, Phoenix, OpenTelemetry)?
42. Describe how you would auto-scale an agentic RAG service to handle bursty traffic without blowing up your LLM API budget.
43. How do you handle rate limits and token budget management when the agent might issue many LLM calls per user request?
44. What is your strategy for graceful degradation — what does the agent do when the vector store is down or the LLM provider is throttling?
45. How do you deploy embedding model updates without downtime or inconsistent retrieval results?

---

## 8 — Data Pipeline & Ingestion

46. How do you design an ingestion pipeline that keeps the knowledge base fresh — handling new documents, updates, and deletions in near real-time?
47. What pre-processing steps (cleaning, deduplication, entity extraction) do you perform before embedding documents, and why?
48. How do you handle multi-modal data (tables, images, charts inside PDFs) in a RAG ingestion pipeline?
49. Describe your approach to document-level and passage-level deduplication to avoid noisy retrieval results.
50. How do you detect and handle data drift in the knowledge base that could degrade retrieval quality over time?

---

## 9 — Advanced / Cutting-Edge Topics

51. How would you implement a "corrective RAG" (CRAG) pattern where the agent fact-checks its own output against retrieved evidence before returning it?
52. Explain the Self-RAG paradigm. How does fine-tuning a model with retrieval-aware special tokens differ from prompt-based agentic RAG?
53. How do you build a graph-augmented RAG system where the agent can traverse a knowledge graph alongside a vector store?
54. What is Agentic Retrieval with Planning (ARP), and how does planning horizon affect answer quality vs. latency?
55. How would you integrate reinforcement learning from human feedback (RLHF) to improve an agent's retrieval decisions over time?
56. Describe how you would build a collaborative multi-agent RAG system — e.g., one agent retrieves, another critiques, and a third synthesizes.
57. How do you approach guardrails and safety layers in agentic RAG to prevent the agent from retrieving or generating harmful content?
58. What is the role of embedding model fine-tuning (e.g., with contrastive learning on domain data) in improving retrieval for agentic RAG?
59. How would you implement speculative retrieval — pre-fetching documents the agent is likely to need based on conversation context?
60. How do you balance parametric knowledge (what the LLM already knows) with retrieved knowledge, and when should the agent prefer one over the other?

---

## 10 — Scenario & System Design

61. Design an agentic RAG system for a legal firm that needs to answer questions across 10 million case documents with strict citation requirements. Walk me through your design.
62. A user reports that the agent is confidently returning outdated information from the knowledge base. How do you diagnose and fix this?
63. Your agentic RAG system's P95 latency has spiked from 3s to 12s after a recent deployment. Walk me through your debugging process.
64. How would you design an agentic RAG system that supports 50+ languages with varying levels of retrieval quality per language?
65. You need to add real-time web search as a fallback retrieval source for your agent. How do you integrate it without sacrificing groundedness?

---

*Good luck — go build something that retrieves AND reasons.* 🐶
