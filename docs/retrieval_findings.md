# Retrieval findings (Phase 4)

Measured on the 40-paper arXiv corpus, 1054 chunks, with the
known-item diagnostic in `src/ragpipe/eval/retrieval_bench.py`
(`make bench`). 176 synthesised queries across four families.

## The headline result is a trap, and that is the finding

| configuration | R@1 | R@5 | MRR@10 | latency |
|---|---|---|---|---|
| dense only | 0.256 | 0.500 | 0.358 | 62 ms |
| sparse only (BM25) | **0.710** | 0.909 | 0.810 | 1 ms |
| hybrid RRF | 0.449 | 0.750 | 0.578 | 17 ms |
| hybrid weighted 50/50 | 0.659 | 0.881 | 0.757 | 25 ms |
| dense + rerank | 0.483 | 0.659 | 0.560 | 322 ms |
| hybrid RRF + rerank | 0.557 | 0.812 | 0.674 | 501 ms |

Read naively, this says "drop the embeddings, ship BM25." A weight sweep
agrees, improving monotonically all the way to `sparse_weight=0.9`:

| fusion weights | R@1 | MRR@10 |
|---|---|---|
| weighted 0.9 dense / 0.1 sparse | 0.520 | 0.588 |
| weighted 0.5 / 0.5 | 0.657 | 0.754 |
| weighted 0.3 / 0.7 | 0.747 | 0.822 |
| weighted 0.1 dense / 0.9 sparse | **0.760** | 0.834 |

**That conclusion is wrong, and acting on it would damage the system.** The
diagnostic synthesises each query from the target chunk's own text, so the
query shares vocabulary with the answer by construction. That is close to the
best case imaginable for term matching and close to irrelevant for a user
typing a natural question.

I built a fourth family to escape the bias — `abstract_to_body`, where the
query is a sentence from a paper's abstract and only *body* passages of that
paper count as hits, with the abstract itself excluded so no verbatim match is
possible. BM25 scored **1.000** on it. That is not evidence of quality either:
accepting any of a paper's 10-25 body chunks makes the task document
identification, and each paper's distinctive jargon (`FL-Net`, `dQwen3.5`, a
bespoke metric name) identifies the document trivially by term match.

A synthesised query cannot separate these retrievers on this corpus. Doing so
requires natural questions labelled against the specific passage that answers
them — the human-verified golden dataset, which is Phase 6's job.

## What the numbers do legitimately support

**1. BM25 earns its place on exact identifiers.** On the `identifier` family
(a real identifier lifted from the corpus, wrapped in light natural framing):

| | R@1 | R@5 |
|---|---|---|
| dense only | 0.060 | 0.140 |
| sparse only | **0.300** | **0.740** |

Five times the R@1 and over five times the R@5. This is precisely the failure
mode hybrid retrieval exists to fix: embedding similarity blurs a specific
spec name, model version or clause number into its neighbourhood, and term
matching does not.

**2. Reranking rescues a weak first pass and cannot rescue a saturated one.**

| | R@1 before | R@1 after rerank | change |
|---|---|---|---|
| dense only | 0.256 | 0.483 | **+89%** |
| hybrid RRF | 0.449 | 0.557 | +24% |
| sparse only (already 0.710) | — | — | nothing left to gain |

Consistent with what a cross-encoder does: it reads query and passage together
in one pass instead of comparing them as independently-computed vectors, so it
fixes ordering mistakes the first pass made. It cannot invent a passage the
first pass never shortlisted — which is why `candidate_k` (30) is much larger
than `top_k` (5).

**3. Fusion is never worse than its weaker input**, so hybrid is a safe
default rather than a gamble.

**4. Reranking costs ~300 ms** per query on MPS for 30 candidates. Real, and
worth it at this corpus size.

## Decisions taken

- **Fusion stays RRF**, not the weighted sum that scores better here. RRF
  combines ranks and never compares a cosine similarity against a BM25 score,
  so it needs no cross-retriever calibration. The weighted sum's apparent win
  rides on BM25's per-query min-max normalisation, which forces the top hit to
  1.0 regardless of absolute match quality — exactly the artifact that makes
  it look strong on lexical queries.
- **Fusion weights stay 0.5/0.5.** Tuning them on this diagnostic would be
  fitting to its bias. Revisit against the golden dataset.
- **Reranking stays on**, with `candidate_k=30` feeding `top_k=5`.
- The honest "improved precision by X%" number comes from Phase 6. It is not
  in this file on purpose.
