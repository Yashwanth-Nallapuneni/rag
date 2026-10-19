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

**D13 caveat (recorded after the live check):** the v3 check used the 10
golden pairs that showed the problem, and the final eval scores the same
171 pairs. v3 is a generic instruction with a generic example, and nothing in
it comes from the golden questions, so the risk of fitting to the test set is
small, but it is not zero. Result of the check: v2 answered 3 of the 10, v3
answered 9. The 5 previously answered pairs still answered. That check
overspent its own $0.10 sub-cap by about $0.07 because of a script crash.
Total spend stays well under $5.

### D14. Install `parskip` and `cm-super` into the owner's TinyTeX

**Decision:** Run `tlmgr install parskip cm-super` (CTAN, TeX's official
package archive) so the resume compiles locally.

**Why:** The resume `.tex` was compiled elsewhere (Overleaf, TeX Live 2026).
The local TinyTeX lacks `parskip` and the cm-super Type 1 fonts, so a local
build runs to 2 pages in bitmap fonts and cannot be checked against
`cv/resume.pdf`. The change is limited to the owner's user-level TeX tree, can
be undone with `tlmgr remove`, and is needed to produce
`resume-updated.pdf` in the same typography.

### D15. Add a document-attribution check instead of the MiniLM gate fallback

**Decision:** If a question names exactly one corpus document by title,
every passage the answer cites must come from that document. Otherwise the
answer is refused as "cites a different document than the one asked about".
D11's planned fallback (MiniLM-L6 as a gate-only scorer) is **not** used.

**Why:**
- The final eval had one false answer (`qa-unans-0013`), the worst failure
  class. Asked about Paint-Anything's licensing, the model quoted *another
  paper's* reproducibility checklist ("we follow the licenses and usage terms
  of the underlying datasets"), cited it, and the verifier approved it on 77%
  term overlap.
- The relevance gate behaved correctly. The question is on-topic and
  Paint-Anything's own passage scored 1.92, so a stricter gate would have
  refused answerable questions without fixing the real fault. That fault is
  attribution: the cited passage was faithful to *a* paper, just not the
  one asked about. Grounding and relevance checks both miss this.
- The check applies only when a title match is unambiguous, so
  cross-paper or untitled questions behave exactly as before.

**Caveat:** this failure was found on the golden set, and the fix is
measured on the same set. The check targets a general failure class
(misattributed citations), not this one question. The before/after replay
must still show it newly refuses no answered answerable pair.

### D16. Fix the sentence splitter on "Eqs."

**Decision:** Common scholarly abbreviations ("Eq.", "Eqs.", "Fig.",
"Figs.", "Sec.", "Tab.", "et al.", "e.g.", "i.e.", "cf.", "vs.") do not end
a sentence.

**Why:** One correct answer was refused because the splitter cut "weighted
L1 loss (Eqs. 3, 4) [S1]" into an uncited fragment and a cited one. That is
a parsing bug of the same family as the earlier "U.S." fix.

### D17. Leave two remaining false-refusal causes alone

**Decision:** No change for:
- answers that cite with "(S1)" in place of "[S1]";
- the judge misreading garbled summation symbols in PDF text.

**Why:**
- "(S1)" is ambiguous with papers' own supplementary references ("Table
  S1"). Accepting it would weaken the citation contract that makes every
  marker resolvable.
- The garbled-math case is a PDF-extraction artifact in one chunk. Working
  around it in the judge prompt would be fitting to one example.
- Both stay counted as false refusals in the reported numbers.

### D18. Spend about $0.60 on a rerank-OFF run of the final commit

**Decision:** Run the rerank-off eval on the final code (d276390) for the
before/after reranking comparison. Projected total spend: about $3.98 of $5.

**Why:** The spec (§2.2, §5) requires before/after reranking numbers.
The only existing rerank-off run used the previous commit, without the
attribution check or the splitter fix, so comparing it to the final
rerank-on run would mix the reranker's effect with those code changes. The
remaining ~$1 covers the two CI eval runs (the push and the Dependabot
merge).

Note on cost tracking: the harness counts only generation and RAGAS judge
calls. The real OpenRouter delta per full run is about $0.60 against a
reported ~$0.33, because the claim verifier's judge calls aren't
attributed. Budget decisions use the real `/key` delta.

### D19. Stop iterating on false refusals; report run-to-run variance

**Decision:** No further fixes after d276390. The 8 remaining false
refusals are reported as they are.

**Why:**
- None of the 8 shows a verifier bug:
  - 2 are the model declining when the gold passage was not retrieved;
  - 6 are genuinely uncited or overbroad claims, which the traceability
    rule is right to reject.
- Two back-to-back runs of nearly identical code differed on 13 pairs, so
  the run-to-run noise is about the size of any further fix. Changes chased
  inside that noise would be fitting the golden set.
- The README reports single-run numbers and says plainly that the
  generator is non-deterministic.

**Progression on the rerank-on runs** (171 human-verified pairs):

| Run | False refusals | False answers |
|---|---|---|
| MiniLM baseline | 32 | 0 |
| bge reranker + numeric fix | 23 | 0 |
| prompt v3 + verifier fixes | 9 | 1 |
| attribution check, final | 8 | 0 |

**Final rerank-on run** (eval_20261017T011212Z_5f0a41881655):

| Metric | Value |
|---|---|
| Refusal accuracy | 0.953 |
| Faithfulness | 0.933 |
| Answer relevancy | 0.820 |
| Context precision | 0.910 |
| Context recall | 0.964 |

**D18 status (2026-10-17):** the rerank-OFF run was stopped before
completing because the session hit its usage limit. Nothing was published
from it. Resume step: run the OFF eval on d276390 (see STATE.md §5) and
compare it like-for-like with eval_20261017T011212Z_5f0a41881655.

### D18 result. The rerank-OFF run completed

**Decision:** Report the OFF run
(`eval_results/eval_20261017T033055Z_4d290a449c2f.json`, commit 50fc0ca,
which has the same code as d276390 plus only a decision-log commit,
`--no-rerank`) next to the ON run
(`eval_20261017T011212Z_5f0a41881655`). Both use the same 171 human-verified
pairs (145 answerable, 26 unanswerable).

**Numbers (ON / OFF):**
- Faithfulness 0.933 / 0.931; answer relevancy 0.820 / 0.813
- Context precision 0.910 / 0.871; context recall 0.964 / 0.985
- Refusal accuracy 0.953 / 0.930; refusal precision 0.765 / 0.684; refusal
  recall 1.0 / 1.0; F1 0.867 / 0.813
- Answered 137 / 133; false refusals 8 / 12; false answers 0 / 0
- On the 128 pairs answered in both: faithfulness 0.929 / 0.929, context
  precision 0.905 / 0.875, context recall 0.961 / 0.984

**Cost:** $0.422 real (OpenRouter `/key` delta) against $0.329 reported by
the harness; the gap is the verifier judge calls the harness does not count.
Total project spend is $3.80 of $5.

**Conclusion:** Reranking does not improve faithfulness. It improves context
precision, refusal accuracy and false refusals (12 to 8). Context recall is
slightly higher with rerank off. Generation is nondeterministic (two
near-identical earlier runs differed on 13 pairs), so differences of a few
pairs are within noise. The README states this reading.

### D20. Resume bullets: variant D

**Decision:** Four variants were drafted and variant D was chosen. It is
balanced: the retrieval gain with its CI, 0 false answers on the 26
unanswerable questions, faithfulness 0.933 and context recall 0.964, false
refusals 32 to 8, CI gating and 363 tests. Output is
`cv/resume-updated.pdf`; the original resume is untouched.

**Why:**
- Variant B was rejected: placing faithfulness next to the reranker result
  implies a causation that the OFF run shows does not exist (0.933 vs 0.931).
- The old Recall@1 figure was dropped because it came from the superseded
  MiniLM setup.

### D21. Ignore the local review-tool outputs

**Decision:** Add `eval/review_sheet.html` and
`eval/review_suggestions.jsonl` to `.gitignore` so the working tree is clean.

**Why:** Identical copies are already committed in `eval/review_record/`;
the top-level files are regenerated by the review tooling.

### D22. Publish the final eval evidence

**Decision:** Copy the two final eval runs (rerank on and off) and the
retrieval experiment into `docs/results/`, with a short README saying what
each file is and where the headline numbers live.

**Why:**
- The README and resume quote numbers from `eval_results/`, which is
  git-ignored, so nobody could check them from a clone.
- Copies rather than un-ignoring `eval_results/`: that folder also holds
  scratch and superseded runs that should stay out of git.

### D23. Close out D7 and keep the CI gate sample out of the headline numbers

**Decision:** D7 is closed. PR #2 was squash-merged as `095a82a`, and on that
commit both workflows passed (tests run 38999802156, eval run 38999802162).
The CI gate figures are not quoted next to the full-run numbers.

**Why:**
- The eval step in CI ran on the new action versions and passed its gate:
  faithfulness 0.8838 against a 0.80 threshold, answer_relevancy 0.8165,
  context_precision 0.9451, context_recall 1.0000, refusal_accuracy 1.0000.
  That is the check D7 said I would make rather than assume.
- The gate runs a 20-question sample (`SAMPLE_SIZE: 20`) to keep CI cheap. The
  full runs use 171 pairs, so the two sets of numbers differ in sample size and
  are not comparable. Quoting 0.8838 next to 0.933 would imply a drop that the
  data does not show.
