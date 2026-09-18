# Production-Grade RAG System — Full Project Spec

**Goal:** Build a domain-specific "ask my docs" system that goes beyond a RAG demo into something production-grade — hybrid retrieval, reranking, enforced citations, and a CI-gated evaluation pipeline. This is the differentiator: most portfolio RAG projects stop at "embed + retrieve + generate." This one proves you understand the full lifecycle of a production AI system.

**Timeline: 8-12 focused days across 3 phases.** Do not compress this by skipping Phase 3 — the eval harness and CI gate are the entire point of the project; without them this is just another RAG demo.

**Domain choice:** pick one corpus and commit — technical documentation, research papers (arXiv subset), legal contracts, or healthcare documents. Recommendation: pick something you can also use personally (e.g. your own coursework/research papers, or Columbia CS course docs) so you're motivated to keep improving it and can speak to real usage, not just a synthetic test.

---

## Phase 1 — Fundamentals (target: 1-2 days)

**1.1 Ingestion**
- Support PDF, Markdown, and web page input.
- Parse cleanly: strip headers/footers/page numbers from PDFs, preserve heading structure from Markdown/HTML where possible (headings become useful metadata later).

**1.2 Chunking**
- Chunk size: 500-800 tokens per chunk.
- Overlap: ~100 tokens between adjacent chunks.
- Why the overlap matters: prevents an important sentence or claim from being sliced across a chunk boundary and losing its supporting context — a naive non-overlapping chunker will silently break retrieval quality on exactly the sentences most likely to be cited.
- Store chunk metadata: source document, page/section, chunk index, and the raw text — you'll need all of this for citations later.

**1.3 Embedding + Vector Store**
- Vector store: ChromaDB or Weaviate (both are solid to start with; Chroma is faster to stand up locally, Weaviate is closer to what you'd see in a real infra stack if you want that on your resume instead).
- Embed each chunk, store alongside its metadata.

**1.4 Retrieval Pipeline**
- Given a query, retrieve top-k most relevant chunks (start with k=5, make it configurable).
- Generate an answer from the retrieved chunks, with the answer explicitly citing which chunk(s) it drew from.

**Phase 1 deliverable:** point the system at a document, ask a question, get an answer, and be able to click through to the exact paragraph the answer came from. This alone should be demoable.

---

## Phase 2 — Production Quality (target: 3-5 days)

**2.1 Hybrid Retrieval**
- Combine BM25 keyword search with vector/semantic search.
- Why both: vector search is strong on meaning/intent, but weak when a user searches for an exact term, phrase, or identifier (a spec name, a legal clause number, a specific drug name) — BM25 handles that case cleanly where pure embedding similarity often misses it.
- Combine scores (start with a simple weighted sum or reciprocal rank fusion; expect to spend real time here tuning the weighting by looking at actual query results, not just implementing the mechanism once).

**2.2 Cross-Encoder Reranking**
- Take the initial retrieved set (e.g. top 20-30 from hybrid retrieval) and rerank using a cross-encoder that scores the query and each chunk together as a pair (not independently, which is what the first-pass retrieval does).
- Use a reranker from sentence-transformers, or Cohere's reranker if you want a hosted option.
- This step consistently and measurably improves precision — expect to see it in your eval numbers in Phase 3, which is itself worth showing (before/after reranking faithfulness scores).

**2.3 Citation Enforcement**
- The system must explicitly decline to answer (not hallucinate a plausible-sounding response) when the retrieved chunks don't actually support a specific claim or question.
- Implement this as an explicit check: after generation, verify each claim in the answer is traceable to a retrieved chunk; if not, the system should say so rather than answer anyway.
- This is the single most important trust property of the system — call it out explicitly in your README and resume bullet.

**2.4 Prompt Versioning**
- Store all prompts in a version-controlled config file (not inline strings scattered through the codebase).
- Treat prompts as part of the system architecture, not throwaway strings — this is a real engineering-maturity signal and should be visible in the repo structure itself (e.g. a `prompts/` directory with versioned files).

**Phase 2 deliverable:** the same demo as Phase 1, but retrieval precision is visibly better, and the system refuses to answer when it should rather than confidently hallucinating.

---

## Phase 3 — Shippable / Production Discipline (target: 3-5 days)

**3.1 Golden Evaluation Dataset**
- Curate 50-200 question-answer pairs, manually verified for correctness against your corpus.
- This is slow, deliberate work — budget real time for it, don't rush it. Quality of this dataset determines whether your eval numbers mean anything.

**3.2 Offline Evaluation**
- Write an evaluation script measuring **faithfulness**: are the claims in the generated answer actually supported by the retrieved chunks?
- Use RAGAS (purpose-built for RAG evaluation) for this — it gives you faithfulness, answer relevance, and context precision/recall out of the box, so you don't have to build these metrics from scratch.

**3.3 CI Integration**
- Wire the evaluation script into your CI pipeline (GitHub Actions).
- Every pull request triggers an automatic evaluation run against the golden dataset.
- If quality (faithfulness score) drops below a defined threshold, the build fails.
- This is exactly how production AI teams operate, and having it visible in your repo (a red/green CI badge tied to *answer quality*, not just tests passing) is a concrete, checkable signal that you understand the full lifecycle of a production AI system — not just "I can call an LLM API."

**Phase 3 deliverable:** a repo where you can point to a CI run and say "this PR was blocked because it regressed faithfulness from 0.91 to 0.84" — that's the sentence that makes this project land in an interview.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain or LangGraph |
| Vector store | ChromaDB or Weaviate |
| Reranking | sentence-transformers cross-encoder, or Cohere Rerank |
| Evaluation | RAGAS |
| Backend | FastAPI |
| CI | GitHub Actions |

---

## What "done" looks like for the resume

- A public repo with a clear README: architecture diagram, what problem it solves, before/after numbers from reranking, and a screenshot of a CI run gating on faithfulness score.
- A live or easily-runnable demo (even a simple Streamlit UI is fine — the engineering depth is in the retrieval/eval pipeline, not the frontend).
- A resume bullet that can survive a technical follow-up, e.g.: *"Built a production-grade RAG system with hybrid BM25 + vector retrieval and cross-encoder reranking, improving retrieval precision by X%; implemented citation enforcement and a RAGAS-based evaluation harness gating CI on faithfulness score (50-200 golden QA pairs)."* — fill in X once you actually measure it in Phase 3, don't estimate it.

## Sequencing note

This is a 3-4 week project at evening/part-time pace, not a one-week project — don't let it be the thing you're relying on for applications due in the next two weeks. Run it alongside your other committed project(s) for immediate applications, and let this become the strongest item in your portfolio once it's actually finished end-to-end, including Phase 3. An unfinished Phase 3 (no CI gate, no golden eval set) is just a normal RAG demo with extra steps — the whole differentiation thesis depends on shipping all three phases, not stopping at Phase 2.
