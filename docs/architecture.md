# Architecture

## Query path

```mermaid
flowchart TB
    Q["Question"] --> DENSE["Dense retrieval<br/>BGE-small-en-v1.5 → ChromaDB<br/>candidate_k = 30"]
    Q --> SPARSE["Sparse retrieval<br/>BM25 Okapi, identifier-preserving tokenizer<br/>candidate_k = 30"]

    DENSE --> FUSE["Reciprocal rank fusion<br/>rank-based, needs no cross-retriever calibration"]
    SPARSE --> FUSE

    FUSE --> GATE{"Relevance gate<br/>best cross-encoder score ≥ −7.0?"}
    GATE -->|no| REFUSE["REFUSE<br/>corpus does not cover this"]
    GATE -->|yes| RERANK["Cross-encoder rerank<br/>ms-marco-MiniLM-L-6-v2<br/>30 candidates → top_k = 5"]

    RERANK --> CTX["Context assembly<br/>[S1]…[S5] markers + locators<br/>budget 6000 tokens"]
    CTX --> GEN["Generation<br/>versioned prompt answer/v2<br/>provider registry"]

    GEN --> SENT{"Refusal sentinel<br/>emitted?"}
    SENT -->|yes| REFUSE
    SENT -->|no| VERIFY["Citation enforcement<br/>per-claim: coverage · numeric · negation<br/>LLM judge on the ambiguous middle only"]

    VERIFY --> RATIO{"supported ratio<br/>≥ 0.80?"}
    RATIO -->|no| REFUSE
    RATIO -->|yes| ANS["ANSWER<br/>every sentence cited to an exact passage"]

    ANS --> CLICK["GET /chunk/{id}<br/>full source paragraph"]
```

## Ingestion path

```mermaid
flowchart LR
    subgraph SRC["Sources"]
        PDF["PDF<br/>40 arXiv papers"]
        MD["Markdown"]
        WEB["HTML / web page"]
    end

    PDF --> PP["PDFParser<br/>column-order recovery<br/>running-head stripping (parity-aware)<br/>heading detection (numbered · roman · font)"]
    MD --> MP["MarkdownParser<br/>ATX + setext, fence-aware"]
    WEB --> HP["HTMLParser<br/>chrome removal, main-content root"]

    PP --> IR["Block IR<br/>text + page + heading stack"]
    MP --> IR
    HP --> IR

    IR --> CH["Chunker<br/>650 tokens, 100 overlap<br/>sentence-aligned boundaries"]
    CH --> JSONL["chunks.jsonl<br/>1054 chunks"]
    JSONL --> EMB["BGE embedder"]
    EMB --> CHROMA[("ChromaDB")]
    CHROMA --> BM["BM25 index<br/>built FROM the store,<br/>so the two cannot drift"]
```

## Evaluation path

```mermaid
flowchart LR
    CORPUS[("Corpus")] --> DRAFT["Candidate drafting"]
    DRAFT --> REVIEW["Human verification<br/>accept · edit · reject"]
    REVIEW --> GOLD[("Golden dataset<br/>50–200 verified pairs<br/>incl. unanswerable")]

    GOLD --> EVAL["RAGAS<br/>faithfulness · answer relevance<br/>context precision · context recall"]
    GOLD --> REF["Refusal accuracy<br/>computed by us: RAGAS has no such metric"]

    EVAL --> GATEJOB{"CI gate<br/>threshold + regression"}
    REF --> GATEJOB
    GATEJOB -->|below threshold or regressed| RED["Build fails"]
    GATEJOB -->|pass| GREEN["Build passes"]
```

## Why the structure is shaped this way

**Parsers emit blocks, not text.** Each block carries the page it came from and
the heading stack it sits under, so a citation can name a page and section.
Provenance is attached at parse time, where the information exists, rather than
reconstructed later from a flat string where it is already lost.

**Wide then narrow.** Each retriever proposes 30 candidates and the reranker
cuts to 5. A cross-encoder can only reorder what it is handed, so a passage
dense retrieval ranked twelfth can never reach first place from a five-item
shortlist.

**The sparse index is built from the vector store, not the chunk file.** Two
independently built indexes drift, and a BM25 hit for a chunk the store lacks
yields a citation whose click-through 404s.

**Grounding and relevance are separate gates.** A faithful quotation of an
irrelevant passage passes every citation check and still fails the user, so
relevance is decided by the cross-encoder before generation is even attempted.

**Every backend sits behind a protocol.** LLM, embeddings and reranker are
swappable, and deterministic offline implementations satisfy the same
contracts — which is why the full pipeline, citation enforcement included, runs
in CI with no API key and no run-to-run variance.
