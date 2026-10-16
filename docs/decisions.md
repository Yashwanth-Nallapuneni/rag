# Decision log

Decisions taken without the owner's sign-off while finishing the project
(the owner delegated them on 2026-10-16). Each entry is the decision, then
why. Newest at the bottom. Earlier design decisions live in `docs/STATE.md`
and the commit history; this file covers the finishing phase.

Standing limits that no decision here overrides: total OpenRouter spend stays
under the owner's $5 budget; commits carry no AI attribution; no force-push or
history rewrite; no number is published unless a run produced it.

---

### D1. Work is done by Sonnet subagents; the main session orchestrates

**Decision:** Hands-on work (running scripts, writing code and docs, analysis)
goes to Sonnet subagents. The main session writes their briefs, checks their
output against the files they produced, and does the merges and commits.

**Why:** The owner asked for this explicitly. Checking every agent claim
against the actual files (not the agent's summary) is what keeps the reported
numbers honest.

### D2. Accept the golden-set review as human-verified

**Decision:** The 171 verified pairs count as human-verified, and the README
and resume may say so.

**Why:** The tool's timestamps show the initial 180 decisions entered about
1 second apart, which on its own would not support the claim. The owner
confirmed they had already read every pair against its passage in
`eval/review_sheet.html` before entering decisions. The 5 corrections from
the audit took 20 to 45 seconds each. The timing and the owner's
confirmation are both written into `eval/review_record/README.md`, so
anyone checking can see exactly what the claim rests on.

### D3. Keep the relevance gate at -7.0 (MiniLM era; superseded by the bge gate of -2.5, see D9)

**Decision:** No change to `retrieval.relevance_gate`.

**Why:** On the 171 verified pairs, -7.0 refuses 0 of 145 answerable
questions (the lowest answerable top score is -3.26, so there is margin).
It stops 13 of 26 unanswerable questions before generation; the rest are left
to citation enforcement, which is the gate's intended division of labour.
Tightening to 0.0 would refuse 7 answerable questions. Loosening gains
nothing. File: `eval_results/gate_calibration_20261016T165554Z.json`.

### D4. Rejected pairs stay in the dataset file, excluded by the ledger

**Decision:** The 9 rejected pairs remain in `eval/golden_dataset.jsonl`
and are excluded through the review ledger, not deleted.

**Why:** `load_verified()` already reads only verified pairs, so deleting
them gains nothing and loses the record of what was rejected and why.

### D5. Archive the full review record in the repo

**Decision:** `eval/review_record/` holds the drafted dataset, the
pre-screen, the AI suggestions, the ledger and a per-pair change table.

**Why:** The owner asked for it, and a golden set is only as credible as
the record of how it was checked. The ledger keeps only the latest decision
per pair, so the first decisions on the 5 re-reviewed pairs are gone. The
record says so rather than reconstructing them.

### D6. Replace deprecated Streamlit calls

**Decision:** `st.components.v1.html` becomes `st.iframe`, and
`use_container_width=True` becomes `width="stretch"`, in the review tool and
the demo app.

**Why:** Streamlit 1.64 warns on every page load that `components.v1.html`
is past its removal date, so a routine upgrade would break the review tool.
`st.iframe` with an HTML string keeps the `allow-scripts allow-same-origin`
sandbox, so the keyboard shortcuts still work. This was checked in the
installed package source; a headless run raises no exception and no warning.

### D7. Merge Dependabot PR #2 (five GitHub Actions bumps) as-is

**Decision:** Merge the grouped bump: checkout 5→7, setup-python 6→7,
cache 5→6, upload-artifact 6→7, github-script 8→9.

**Why:** Each breaking change in the release notes was checked against how
the workflows use the action:
- checkout v7 blocks fork checkouts under `pull_request_target`, which no
  workflow uses.
- setup-python v7 drops a `pip-install` input that the workflows don't use.
- github-script v9 breaks `require('@actions/github')` and `getOctokit`
  declarations. The workflow script uses only `require('fs')` and the
  injected `github`, `context` and `core`.

The PR's `test` check passed. Its `eval` check was skipped because Dependabot
PRs never receive the secret, so the bumped eval steps first run on the next
push to main. I check that run, not assume it.

### D8. Fix the verifier's leading-dot decimal bug

**Decision:** The numeric-agreement check treats ".034" and "0.034" as the
same number (commit `7c78b34`).

**Why:** Statistics papers drop the leading zero and models restore it. Two
correct answers in the verified-set eval were refused for citing "figures
absent from the passage". This is a parsing bug, not a threshold question, so
it is fixed rather than tolerated. Replaying all 4,894 claim/chunk pairs from
the eval: 7 checks flip fail→pass, all this case, and 0 flip pass→fail.
Tests prove that different numbers (0.035 vs .034, 1.034 vs 0.034,
2.5 vs 3.1) still fail.

### D9. Switch the reranker to BAAI/bge-reranker-base (fusion unchanged)

**Decision:** The cross-encoder changes from
`cross-encoder/ms-marco-MiniLM-L-6-v2` to `BAAI/bge-reranker-base`. Fusion
stays RRF 0.5/0.5, candidate_k 30 and top_k 5. The relevance gate is
re-calibrated for the new model's score scale.

**Why:**
- In 21 of the 32 false refusals in the first verified-set eval, the gold
  passage never reached the model's top 5. The model then correctly said it
  could not answer, so the fault was retrieval.
- On held-out verified pairs, MiniLM-L6 reranking scored recall@5 0.707,
  below the fusion order it was reranking.
- The retrieval experiment (`eval_results/retrieval_experiment_20261016T200538Z.json`)
  chose its winner on the tune split only (87 pairs), using the same rule as
  the fusion tuner. bge-reranker-base won. On the held-out 58 pairs it scores
  recall@5 **0.879 vs 0.707**, +0.172, 95% CI [+0.05, +0.29]. The CI
  excludes 0, so the pre-set rule says change.

**Rejected alternatives:**
- *Drop reranking and order by weighted fusion* (the experiment agent's
  preferred option, test recall@5 0.897). The spec (§2.2) requires
  cross-encoder reranking and before/after reranking numbers. Its held-out
  edge over bge (0.897 vs 0.879) is one question out of 58, well inside
  the CI.
- *bge with weighted-0.6 fusion and a rank blend* (test 0.948). It scored
  third on tune. Picking it would mean choosing on the test split after
  looking at it, which is exactly the selection bias the split exists to
  prevent.
- *bge-reranker-v2-m3.* It did not run on this 8 GB Mac: it swapped for over
  20 minutes on MPS and took 16 s per query on CPU.

**Costs accepted:**
- About 1.6–2.1 s per query against 0.3 s.
- Weaker gate separation (AUROC 0.83 vs 0.88), so more in-domain
  unanswerable questions reach citation enforcement. That stage has produced
  0 false answers so far, and the re-run eval measures whether that still
  holds.

**Gate re-calibration (`make calibrate`, `eval_results/gate_calibration_20261016T201503Z.json`):**
rule fixed beforehand: highest threshold with 0 false refusals on the 145
answerable verified pairs, backed off to a round value with at least 1 logit
of margin. Lowest answerable top score is -1.32, so the ceiling is just under
that; -2.5 gives margin 1.18 (the MiniLM gate had about 3.7). It stops 9 of 26
unanswerable questions before generation (MiniLM at -7.0 stopped 13). The
provider pins identity activation because sentence-transformers defaults
bge-reranker-base to a sigmoid.

### D10. Keep top_k at 5

**Decision:** The model still sees 5 passages.

**Why:** Going to 8 adds about 0.03–0.08 recall for about 1.6 times the
context tokens (≈2,300 → ≈3,750). The spec starts at k=5. With the stronger
reranker, recall@5 is already 0.879 on held-out pairs. Changing two things at
once would also blur what the reranker switch alone did.

### D11. Relevance gate -2.5 for bge-reranker-base, accepted as a weaker first line

**Decision:** The gate becomes -2.5 on bge logits, with the scorer pinned to
raw logits (identity activation). The reranker is still the only model
scoring the gate. MiniLM is not kept as a separate gate scorer.

**Why:**
- sentence-transformers applies a sigmoid to bge-reranker-base by default.
  Left unpinned, the gate would compare a probability against a logit
  threshold. Pinning identity makes the scale explicit, and a test asserts
  that scores fall outside (0,1).
- The value follows a rule set before calibrating: the highest threshold
  with 0 false refusals on the 145 answerable verified pairs, backed off to a
  round value with at least 1 logit of margin. The lowest answerable score is
  -1.32, so -2.5 gives 1.18 of margin. It stops 9 of 26 unanswerable questions
  (MiniLM at -7.0 stopped 13). Generic off-topic probes ("capital of
  Mongolia", "what is self-attention?") still score around -5.5 to -5.9 and
  are refused at the gate.
- Weaker separation is the cost already accepted in D9. A question like
  "side effects of ibuprofen" scores 5.45 because the corpus has a women's
  health paper, so it is left to citation enforcement. A second gate-only
  model would add complexity and a second calibration. That is only worth it
  if enforcement actually lets an unanswerable question through.

**Test that decides it:** the re-run eval counts false answers
(unanswerable but answered). If that is above 0, the fallback is MiniLM-L6
as a gate-only scorer at -7.0 over the shown passages: AUROC 0.88, 0/145
false refusals, measured in the retrieval experiment.

### D12. Fix three verifier false negatives; let contrast words send negation mismatches to the judge

**Decision:**
- Read space-grouped thousands ("1 000", "5 000") as one number.
- Stop reading inline enumerators ("(1)", "(2)", "(i)") as figures.
- When a claim negates and its cited passage has no negation word but does
  contain a contrastive construction ("otherwise", "rather than",
  "instead of", "in contrast"), escalate the claim to the LLM judge instead of
  failing it outright. A claim that negates against a passage with neither
  negation nor contrast still fails hard.

**Why:**
- After the reranker switch, 6 of 23 false refusals were the verifier
  rejecting correct answers:
  - "N = 1 000" parsed as 1 and 0.
  - "(2)" read as the figure 2.
  - Four faithful paraphrases rejected as contradictions. "Set to 0 when
    (j,i) is not in E" against "and 0 otherwise", for example.
- The original rule ("a model must not talk itself into accepting a negation
  the passage does not contain") is kept where it matters: the passage gives
  no hint of contrast. A contrastive passage can genuinely support a negated
  paraphrase, and the judge exists to settle exactly that kind of ambiguity.
- Numeric mismatches are still never escalated.

### D13. Answer prompt v3: the first sentence must carry a citation too

**Decision:** A new prompt version (v2 stays as it was) adds an explicit
rule: no introductory or summary sentence without a marker, including the
first sentence. It also includes one short example.

**Why:** 10 of 23 false refusals had a correct, cited body with an uncited
opening sentence ("On-Demand Attention (ODA) is a decoding method...").
Under the sentence-level traceability rule those refusals are correct, and
the rule stays. The model was breaking a rule v2 already states, so the fix
belongs in the instruction, not in the verifier. Prompts are versioned and
the version goes into the config fingerprint, so the change is traceable in
every eval result.
