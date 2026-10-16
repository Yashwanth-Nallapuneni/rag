# Golden dataset review record

This directory archives the full record of the review of the ragpipe golden dataset (`eval/golden_dataset.jsonl`). It is self-contained: every input and every derived table is copied here, so the record survives later changes to the originals.

## Purpose and process

The golden dataset holds 180 question and ground-truth pairs drafted by an LLM (153 answerable, 27 unanswerable). Each answerable pair is tied to one or more source passages in `data/processed/chunks.jsonl`. The review checked that every pair is supported by its passage, makes sense as a standalone question and has a correct answer.

1. LLM pre-screen (meta-llama/llama-3.3-70b-instruct, advisory only): flagged 39 pairs as suspect and 141 as ok. See `golden_prescreen.jsonl`.
2. Sonnet subagent suggestions (advisory only): independently read every pair against its source passage and produced approve, fix or reject suggestions (115/56/9). See `review_suggestions.jsonl` and the rendered `review_sheet.html`.
3. Owner review: the owner (Yashwanth Nallapuneni) reviewed every pair against its source passage, first in `review_sheet.html`, then recorded decisions in the Streamlit tool (`scripts/review_golden.py`).
4. Orchestrator audit: Claude (the main session) re-checked the recorded decisions and flagged 5 pairs.
5. Owner re-review: the owner re-reviewed the 5 flagged pairs and the ledger was updated.

### Timing of the decisions

The initial 180 decisions were entered in the tool between 2026-10-16 16:37:41 and 16:46:12 UTC, with a median gap of 1 second between entries. They were entered that quickly because the owner had already read the pairs in `review_sheet.html`; the owner confirmed this. The tool timestamps therefore record when decisions were typed in, not how long each pair was read. The 5 corrections were made between 16:50:42 and 16:52:29 UTC.

Note on history: `golden_review.jsonl` keeps one row per pair (the latest decision wins), so the original timestamps and statuses of the 5 corrected pairs were overwritten and are not recoverable from the file. In `changes.jsonl` their `human_decision_history` therefore holds a single entry. The corrected rows also have an empty `reviewer` field, unlike the other rows (`Y`).

## Final counts

- Verified: 171 (145 answerable, 26 unanswerable)
- Rejected: 9 (8 answerable, 1 unanswerable)
- Pairs whose question changed: 56; ground truth changed: 5; both: 4

## AI suggestion by human final decision

| AI suggestion | Verified | Rejected | Total |
|---|---|---|---|
| approve | 115 | 0 | 115 |
| fix | 56 | 0 | 56 |
| reject | 0 | 9 | 9 |
| total | 171 | 9 | 180 |

All 56 `fix` suggestions ended verified with an edit applied. The one pair where the human final result differs from the AI suggestion is `qa-eb462010f96a0386-00004-numeric` (suggested approve, edited by the audit, see below). `human_vs_ai` in `changes.jsonl` is "agree" when the final status matches the suggestion (a `fix` counts as agreed when it was verified with an edit) and "disagree" otherwise.

## Orchestrator audit findings and outcomes

a. `qa-30f3358fb5e10270-00003-definition`: initially approved although the AI suggested reject. It is an exact duplicate question of `-00002` with a subset answer. Re-reviewed and REJECTED.
b. `qa-05aa00352d3d8258-00000-multi_hop`: approved without its suggested fix (the question named no paper). Fixed and approved.
c. `qa-05aa00352d3d8258-00008-factual` and `-00009-comparison`: the AI-suggested question added "(Bulut Doctrine)", which is not in either source passage. The phrase was removed and the pairs approved. (`-00004` keeps "Bulut Doctrine" correctly, because it is in that passage.)
d. `qa-eb462010f96a0386-00004-numeric`: the question named no paper. The paper title was added and the pair approved.
e. Spot checks of numbers against passages all matched. Examples: Harm Laundering rho=+0.55 p=.034 and 15 clusters (14 in GPT-2 to text-davinci-002); RF-Fingerprinting 97%; JEPA-Anything 7 domains; ActObs lambda=0; RetireOPD 3 phases; Recursive QLSTM MAE below 3.9 C.

## Rejected pairs

- `qa-20c7397a39c7d029-00009-numeric` (answerable): Question asks 'how many' values are sampled but passage gives no count; GT '24-bit' does not answer it. Unanswerable as written.
- `qa-20c7397a39c7d029-00012-multi_hop` (answerable): GT wrong against the paper: the full Table 1 (next chunk) shows FLUX.2-4B + Ours scoring 75.57 on ACBench-Edit, above FLUX.2-dev's 68.87; the cited chunk is truncated before those rows. 'Overall' column is also ambiguous. Only valid if restricted to baselines.
- `qa-30f3358fb5e10270-00003-definition` (answerable): Exact duplicate question of qa-30f3358fb5e10270-00002 with a subset answer (found in the orchestrator audit; the AI also suggested reject: Identical question text to #34 (qa-30f3358f...-00002); GT is a strict subset (omits recall head). Duplicate; drop one.)
- `qa-6a630c46bfcbfcbb-00011-comparison` (answerable): Malformed question quoting a section heading and a truncated sentence; asks for a 'comparison or contrast' but none exists. GT is just a copy of the sentence; chunk is mostly a results table. Unusable.
- `qa-6a630c46bfcbfcbb-00012-definition` (answerable): GT is circular and vacuous ('a category of error ... characterized by counts'); the chunk is a bare table with no definition of 'Explicit overclaim'. The real definition is in chunk 00003 (an agent overclaims when its final response asserts an action or completion contradicted by evidence in its own context).
- `qa-a4b810cad99df312-00005-multi_hop` (answerable): GT likely wrong: the passage lists exact gains in descending order (X310: 28.4, 25.2, 22.1, 18.9 = 90/80/70/60% of 31.5; B210: 80.8, 71.8, 62.9, 53.9 = ~90/80/70/60% of 89). 70% of 89 is about 62.3, matching 62.9, not 71.8 (which is ~80%). Passage ordering is ambiguous and the question requires arithmetic inference; drop it.
- `qa-c8bdddc202df7186-00006-numeric` (answerable): Source is garbled figure-label text ('Reach ... Grasp Transport 50% cumulative positive growth ... Align'); the 50% looks like a threshold marker in the figure, not a value associated with the Grasp Transport step. GT not reliably supported and question names no paper.
- `qa-eb462010f96a0386-00003-numeric` (answerable): GT '30.6B' is a text-extraction artifact: the chunk reads 'Qwen30.6B' (Qwen3 + 0.6B, confirmed by other chunks: 'Qwen3-0.6B-Instruct on Countdown'). Real answer is 0.6B. Rewrite as Q: 'Which Qwen model size is used for the Countdown experiments in the score centering paper?' GT: 'Qwen3-0.6B' only if the human accepts corroboration from other chunks; the cited chunk alone does not support it.
- `qa-unans-0025` (unanswerable): Corpus DOES answer this: chunk a4b810cad99df312::00015 states 'This work was supported by the US ARO under cooperative agreement W911NF-23-1-0214 and W911NF-24-2-0133, and by US NSF award 2302469, 2428761.' Replace with a question the paper truly does not answer.

## Pairs whose question or ground truth changed (57)

Q is the question and GT the ground truth. Full texts are in `changes.jsonl`.

- `qa-05aa00352d3d8258-00000-multi_hop`
  - Q before: According to the version note, what specific status update regarding the detector study is reflected in the changes to Section 3 and the reference list in version 1.1?
  - Q after: In the version 1.1 note of the paper 'Summarization Bias: The Directional Collapse of Objective Projection into Told-Mode Labels in Large Language Models', what status update regarding the detector study is reflected in the changes to Section 3 and the reference list?
- `qa-05aa00352d3d8258-00008-factual`
  - Q before: What is the target Cohen's h or odds ratio fixed for the model told-preference rate to exceed the human told-preference rate in the evaluative test?
  - Q after: In the pre-registered test protocol of the 'Summarization Bias' paper, what target Cohen's h or odds ratio is fixed for the model told-preference rate to exceed the human told-preference rate in the evaluative test?
- `qa-05aa00352d3d8258-00009-comparison`
  - Q before: How does the protocol distinguish a genuine told-preference from verbosity or sycophancy in Stage 4?
  - Q after: In the pre-registered test protocol of the 'Summarization Bias' paper, how does Stage 4 distinguish a genuine told-preference from verbosity or sycophancy?
- `qa-065ffa842a655e37-00004-multi_hop`
  - Q before: In the five real-world experiments, what was the inference latency achieved by the model that explicitly models the future evolution of tactile observations to yield a 29.4% relative gain in overall success rates?
  - Q after: In the Agile-WAM paper's five real-world experiments, what inference latency was achieved alongside the 29.4% relative gain in overall success rates?
- `qa-065ffa842a655e37-00006-factual`
  - Q before: What specific representation does Agile-WAM use to replace Gaussian noise in its noisefree flow-matching policies?
  - Q after: In the Agile-WAM paper, what do recent noise-free flow-matching policies use to replace Gaussian noise?
- `qa-08c84f25326f5143-00001-comparison`
  - Q before: How does the token efficiency of the hybrid backbone compare to the full-attention control when reaching a specific training loss?
  - Q after: When adapting Qwen3.5 backbones into diffusion language models (dQwen3.5), how does the token efficiency of the hybrid backbone compare to the full-attention control in reaching a given training loss?
- `qa-16f3620e21880ee0-00001-comparison`
  - Q before: How do the authorship designations for the first two authors compare to those of the last two authors?
  - Q after: In the FL-Net paper ('Multi-center Medical Data Mining with FL-Net'), how do the authorship designations for the first two authors compare to those of the last two authors?
- `qa-16f3620e21880ee0-00017-numeric`
  - Q before: How many participating hospitals are expected to take part in the two EU projects?
  - Q after: In the FL-Net paper, how many participating hospitals are expected to take part in the two EU projects (dAIbetes and Microb-AI-ome)?
- `qa-18b1407b9c238edf-00009-factual`
  - Q before: What specific optimization algorithm is used to train the teacher policy to obtain a skilled teacher?
  - Q after: In the RetireOPD framework, what optimization algorithm is used to train the teacher policy to obtain a skilled teacher?
- `qa-20c7397a39c7d029-00006-numeric`
  - Q before: What is the name of the dataset pipeline illustrated in Figure 4?
  - Q after: In the Paint-Anything paper, what is the name of the data pipeline illustrated in Figure 4?
- `qa-25147abfdad7da75-00011-numeric`
  - Q before: What is the maximum number of steps allowed for each task to run?
  - Q after: In 'An Empirical Study of Harness Design for Coding Agents', what is the maximum number of steps each task may run?
- `qa-2803ed2f586f5cb9-00000-definition`
  - Q before: According to the study, what temporal ordering do the mechanisms of intergroup hostility rhetoric tend to follow?
  - Q after: In 'Unifying Models of Intergroup Hostility in Online Discourse', what temporal ordering do the mechanisms of intergroup hostility rhetoric tend to follow?
- `qa-2803ed2f586f5cb9-00001-comparison`
  - Q before: How does the temporal appearance of boundary construction compare to that of scapegoating in the observed intergroup hostility rhetoric?
  - Q after: In 'Unifying Models of Intergroup Hostility in Online Discourse', how does the temporal appearance of boundary construction compare to that of scapegoating?
- `qa-2803ed2f586f5cb9-00005-numeric`
  - Q before: How many annotated posts are contained in the multi-platform corpus studied during the 2024 U.S. presidential election?
  - Q after: In the 'Unifying Models of Intergroup Hostility in Online Discourse' paper, how many annotated posts does the multi-platform corpus from the 2024 U.S. presidential election contain?
- `qa-2803ed2f586f5cb9-00017-comparison`
  - Q before: How does the study distinguish between the measured linguistic manifestations of mechanisms and the underlying psychological processes?
  - Q after: In 'Unifying Models of Intergroup Hostility in Online Discourse', how does the study distinguish between the measured linguistic manifestations of mechanisms and the underlying psychological processes?
- `qa-2b88d5a23b858daf-00007-definition`
  - Q before: According to the provided definition, what specific hyperparameters define a multi-layer perceptron (MLP) architecture?
  - Q after: In the paper 'Beyond PINNs: A Unified Gauss-Newton and Petrov-Galerkin Framework for Neural and Hybrid PDE Solvers', which hyperparameters define a multi-layer perceptron (MLP) architecture in Definition 2.2?
- `qa-2b88d5a23b858daf-00025-definition`
  - Q before: What specific modification is made to the eigenGreen sections to obtain the shifted Sobolev family used for the RC problem?
  - Q after: In 'Beyond PINNs: A Unified Gauss-Newton and Petrov-Galerkin Framework for Neural and Hybrid PDE Solvers', what modification is made to the truncated eigenGreen sections to obtain the shifted Sobolev family used for the RC (reentrant corner) problem?
- `qa-30f3358fb5e10270-00008-numeric`
  - Q before: What is the complexity of the per-step attention costs for the Local view?
  - Q after: In On-Demand Attention (ODA), what is the per-step attention cost complexity of the Local view?
- `qa-3b3fbd6bf534abc2-00003-comparison`
  - Q before: How do GP-based methods differ from NN-based approaches in terms of error bounds, data requirements, and scalability?
  - Q after: In the COIN-GP paper on cooperative online learning with Gaussian process regression, how do GP-based methods differ from NN-based approaches in terms of error bounds, data requirements, and scalability?
- `qa-3b3fbd6bf534abc2-00006-comparison`
  - Q before: How does the definition of the adjacent matrix element a_{i,j} differ between the condition where (j, i) is in the edge set E and where it is not?
  - Q after: In the COIN-GP paper on cooperative online learning in networked systems, how does the definition of the adjacent matrix element a_{i,j} differ between the case where (j, i) is in the edge set E and where it is not?
- `qa-41e2a90782edf723-00003-definition`
  - Q before: What is the definition of the contact strategy in the context of obstacle-aware contact execution?
  - Q after: In SAFEHARNESS ('Coding Agents with an Obstacle-Aware Harness for Safe Robot Manipulation'), what is the contact strategy in obstacle-aware contact execution?
- `qa-41e2a90782edf723-00010-numeric`
  - Q before: How many distinct points trigger replanning in the obstacle-aware route planning process?
  - Q after: In SAFEHARNESS's obstacle-aware route planning, at how many distinct points is replanning triggered?
- `qa-45b8a84b970cd1b4-00003-factual`
  - Q before: How many recorded failures were used in the benchmark where full replay made no model call and reproduced each run identically across 20 repetitions?
  - Q after: In the Chronicle paper on cut-point replay for LLM agents, how many recorded failures were in the benchmark on which full replay made no model call and reproduced each run identically across 20 repetitions?
- `qa-591684f862ff7c59-00007-factual`
  - Q before: Which specific model variant is listed in Table 1 as the large variant of the Mistral model?
  - Q after: In 'WiC is Not WSD: A Study on LLMs and Lexical Ambiguity Resolution', which model variant does Table 1 list as the large (L) Mistral model?
- `qa-620c94c4438e5e62-00000-comparison`
  - Q before: How does the application of the policy network's actions differ between the offline training phase and the deployment on the real vehicle?
  - Q after: In the MILER framework, how does the application of the policy network's actions differ between offline training and deployment on the real vehicle?
- `qa-620c94c4438e5e62-00001-comparison`
  - Q before: How does the total distance driven with two different vehicles compare to the length of the test track used for evaluation?
  - Q after: In the MILER paper, how does the total distance driven with two vehicles compare to the length of the test track used for evaluation?
- `qa-648832a767918090-00004-multi_hop`
  - Q before: According to the study, what specific factor explains most of the apparent gain from broader loss coverage, and how does the student's next-token distribution differ from the teacher's in terms of the inputs used for scoring?
  - Q after: In 'What Does Privileged Information Add to On-Policy Self-Distillation?', what factor explains most of the apparent gain from broader loss coverage, and how does the student's next-token distribution differ from the teacher's in terms of the inputs used for scoring?
- `qa-648832a767918090-00005-multi_hop`
  - Q before: What specific value is used to clip each vocabulary term in the teacher-to-student forward KL for the OPSD loss, and what is the maximum number of completion tokens allowed for direct-response profiles?
  - Q after: In 'What Does Privileged Information Add to On-Policy Self-Distillation?', what value is used to clip each vocabulary term in the teacher-to-student forward KL of the OPSD loss, and what is the maximum number of completion tokens for direct-response profiles?
- `qa-648832a767918090-00008-factual`
  - Q before: How many problem-cluster bootstrap resamples were used for the paired 95% intervals in the principal reference comparisons?
  - Q after: In 'What Does Privileged Information Add to On-Policy Self-Distillation?', how many problem-cluster bootstrap resamples were used for the paired 95% intervals in the principal reference comparisons?
- `qa-67ab5e869fe15b54-00001-comparison`
  - Q before: How does the sample efficiency of the pretrained model compare between the same-SA target and the transition-modeled target at N = 1000, and how does this comparison change by N = 5000?
  - Q after: In 'How Does Distribution Shift Shape Pretraining Gains in Neural PDE Surrogates?', how does the sample efficiency of the pretrained model compare between the same-SA target and the transition-modeled target at N = 1000, and how does this comparison change by N = 5000?
- `qa-67ab5e869fe15b54-00009-factual`
  - Q before: Under the same-closure geometry shift, how does the pretraining gain change when the allocation is switched to breadth?
  - Q after: In 'How Does Distribution Shift Shape Pretraining Gains in Neural PDE Surrogates?', under the same-closure geometry shift, how does the pretraining gain change when the target-data allocation is switched to breadth?
- `qa-6dbe02b2bc472b35-00019-definition`
  - Q before: According to the discussion, what is the specific setting of the problem where the number of steps T is no larger than the dimension d?
  - Q after: In 'Stable Movement for Nondual Lipschitz Convex Optimization', what setting does the paper describe where the number of steps T is no larger than the dimension d?
- `qa-7711e2ab2f019d39-00005-multi_hop`
  - GT before: SAFARI covers open-ended hazard reasoning, standards-grounded E/S/C assessment, ASIL allocation, safety goal generation, and FTTI estimation, building on prior work in ontology-based scene modeling and hazardous-event identification.
  - GT after: SAFARI covers open-ended hazard reasoning, standards-grounded E/S/C assessment, ASIL allocation, safety goal generation, and FTTI estimation; it builds on prior work on ontology and representation (Ulbrich et al., 2015; Menzel et al., 2018), ontology-based scene modeling (Bagschik et al., 2018), and hazardous-event identification (Bagschik et al., 2016; Graubohm et al., 2020).
- `qa-7a10e4cedd77494a-00005-factual`
  - Q before: In the context of diffusion posterior sampling, what are the two terms that constitute the posterior-score decomposition for ∇x log p(x | yobs)?
  - Q after: In the PosteriorBench paper, what two terms constitute the posterior-score decomposition for the gradient of log p(x | y_obs) used in diffusion posterior sampling?
- `qa-7a10e4cedd77494a-00007-comparison`
  - Q before: How does the prior distribution used in the Poisson source recovery task differ from the prior used in the Darcy flow inversion task in terms of structural diversity and spectral variability?
  - Q after: In PosteriorBench, how does the prior distribution used in the Poisson source recovery task differ from the prior used in the Darcy flow inversion task in terms of structural diversity and spectral variability?
- `qa-7e6dfa7f1be9cdee-00009-multi_hop`
  - Q before: According to the proof of Corollary 1, what specific property of matrix P, established by Proposition 1, ensures that the error norm ||b_z - z||_2 equals ||e||_2, and what happens to the condition number of P if the cross-factor orthogonality constraint is removed and directions become nearly repeated?
  - Q after: In the JEPA-Anything paper, according to the proof of Corollary 1, what property of matrix P (established by Proposition 1) ensures that the error norm ||b_z - z||_2 equals ||e||_2, and what happens to the condition number of P if the cross-factor orthogonality constraint is removed and directions become nearly repeated?
- `qa-894ef52e8f805feb-00007-comparison`
  - Q before: How does the distribution-induced view proposed in the paper differ from the approach of treating relation prediction as a flat classification problem over the thirteen Allen labels?
  - Q after: In the Probabilistic Allen Algebra (PAA) paper, how does the distribution-induced view differ from treating relation prediction as a flat classification problem over the thirteen Allen labels?
  - GT before: The distribution-induced view assigns probabilities based on how much of the distribution of uncertain temporal quantities falls into the region where a specific relation holds, allowing for smooth variation and distinction between similar configurations, whereas the flat classification approach assigns probabilities to relations directly or treats them as discrete labels without accounting for the underlying uncertainty in the temporal objects.
  - GT after: The distribution-induced view derives each relation's probability from how much of the distribution over the uncertain temporal quantities (point location, or interval midpoint and duration) falls in the region where that relation holds, so probabilities vary smoothly and distinguish configurations that one crisp label conflates. Prior work either assigns probabilities to relations directly or treats relation prediction as flat classification over the thirteen Allen labels, where crisp relations are brittle (a sub-threshold boundary shift flips the label) and inadequate (one label can cover markedly different configurations).
- `qa-9a2a29cd0f8a3b1a-00009-multi_hop`
  - Q before: In the Phase 2 local refinement stage, what is the specific value of the scalarization weight λ(2) used, and what is the upper bound of the constraint interval for the O1 fractional coordinate?
  - Q after: In the CrystalMO-TuRBO paper, what scalarization weight lambda(2) is used in the Phase 2 local refinement stage, and what is the upper bound of the constraint interval for the O1 fractional coordinate?
- `qa-9fd7e8cc5a3af6e6-00008-comparison`
  - Q before: How does the computational cost of RL fine-tuning compare to supervised learning in terms of the amount of interaction required?
  - Q after: In the OPTED paper on on-policy fine-tuning for end-to-end driving, how does the computational cost of closed-loop RL fine-tuning compare to supervised learning in terms of the amount of interaction required?
- `qa-a4b810cad99df312-00000-numeric`
  - Q before: What is the highest accuracy percentage reported for the proposed method after calibration?
  - Q after: In 'Calibrated RF-Fingerprinting Under Interference With Heterogeneous Transmission Protocols', what is the highest accuracy reported for the proposed method after calibration?
- `qa-a4b810cad99df312-00003-numeric`
  - Q before: How many sections are described in the remainder of the paper's structure?
  - Q after: In 'Calibrated RF-Fingerprinting Under Interference With Heterogeneous Transmission Protocols', how many sections are described in the remainder-of-the-paper outline?
- `qa-afb3c5e4e569f6b2-00000-comparison`
  - Q before: How does the performance of the proposed estimators compare to direct estimators in terms of point and interval estimation accuracy and coverage?
  - Q after: In 'Prediction-Powered Smoothing and Validation for Disaggregated AI Evaluation', how does the performance of the proposed estimators compare to direct estimators in point and interval estimation, and what coverage do they achieve?
- `qa-b3230f942fb55cfe-00004-numeric`
  - Q before: What is the minimum value of s for the finite ordered support of the ordinal random variable Xi?
  - Q after: In the structured statistical model (SSM) of 'Epidemiological Causal Graph Identification: Challenges, Identifiability and Algorithms', what is the minimum value of s for the finite ordered support {1, ..., s} of an ordinal random variable Xi?
- `qa-b3230f942fb55cfe-00008-multi_hop`
  - Q before: In the context of the identifiability proof for the models MX→Y and MY →X, which specific lemma rules out the condition where the log-posterior ratio is affine in T(y) for w ≠ 0?
  - Q after: In the identifiability proof of Theorem 1 in 'Epidemiological Causal Graph Identification: Challenges, Identifiability and Algorithms', which lemma rules out the log-posterior ratio being affine in T(y) for w != 0?
- `qa-c0039283c127ceac-00013-numeric`
  - Q before: What is the standard nonsmooth rate for the generic dual problem with the full ℓp∗ unit ball of subgradients?
  - Q after: In 'The First-Order Oracle Complexity of Lipschitz Convex Optimization in Nondual Settings', what is the standard nonsmooth rate for the generic dual problem with the full l_{p*} unit ball of subgradients?
- `qa-c0039283c127ceac-00023-multi_hop`
  - Q before: According to Theorem 10, what specific condition on the relationship between p and q, combined with the constraint T ≤ d, allows the derived optimization result to nearly match the lower bounds established in Guz15b and Guz15a?
  - Q after: In 'The First-Order Oracle Complexity of Lipschitz Convex Optimization in Nondual Settings', according to Theorem 10, what condition on the relationship between p and q, combined with T <= d, lets the optimization result nearly match the lower bounds of Guzman (Guz15b, Guz15a)?
- `qa-c8bdddc202df7186-00007-multi_hop`
  - Q before: In the context of Flow Matching VLA policies, what specific intermediate outputs generated during numerical integration are restricted to form a prefix-wise representation for adaptive action chunking, and what prior relationship involving these trajectories does the paper investigate?
  - Q after: In the GeoAAC paper, which intermediate outputs of Flow Matching numerical integration are restricted to an action prefix to form a prefix-wise representation for adaptive action chunking, and what prior relationship involving denoising trajectories does the paper investigate across prefixes?
- `qa-cd1e4f5c0b211131-00007-numeric`
  - Q before: How many patient–doctor datasets were used to draw PCOS or hormonal and fertility questions?
  - Q after: In HerHealthEval, how many patient-doctor datasets were used to draw the PCOS or hormonal and fertility questions?
- `qa-d15c40c94baead79-00004-definition`
  - Q before: How is the generalization gap defined in the context of the experimental evaluation?
  - Q after: In 'Recursive Quantum Long Short-Term Memory for Stable Short-Horizon Temperature Forecasting', how is the generalization gap defined in the experimental evaluation?
- `qa-d15c40c94baead79-00006-numeric`
  - Q before: At what temperature threshold does Recursive QLSTM achieve MAE for all window lengths?
  - Q after: In the Recursive QLSTM temperature-forecasting paper, what MAE bound does Recursive QLSTM stay below for all window lengths, compared with QLSTM's roughly 4.1-4.2 C?
  - GT before: 3.9◦C
  - GT after: Recursive QLSTM achieves an MAE below 3.9 C for all window lengths.
- `qa-e75aa8208909715d-00005-multi_hop`
  - Q before: What specific normalization is applied to the linear readout before it passes through its sigmoid output gate, and what is the name of the method that extends the delta rule to entire video frames?
  - Q after: In the Video DeltaNet paper, what normalization is applied to the linear readout before its sigmoid output gate, and what is the name of the method that extends the delta rule from individual tokens to entire video frames?
- `qa-eb462010f96a0386-00004-numeric`
  - Q before: In the example provided for fp4_e2m1 precision, what is the result of the calculation 0.5 + (0.5 + 2)?
  - Q after: In the paper "Score Centering Stabilizes Off-policy Reinforcement Learning "  the example provided for fp4_e2m1 precision, what is the result of the calculation 0.5 + (0.5 + 2)?
- `qa-ebe6d9cf813777ce-00001-factual`
  - Q before: What specific type of experimental data does TetrisCNN use to learn sparse, interpretable latent representations?
  - Q after: In the TetrisCNN paper, in terms of what quantities does TetrisCNN learn its sparse, interpretable latent representations?
  - GT before: Spin correlators
  - GT after: Spin correlators (applied to experimental snapshots of two-dimensional Ising and XY quantum simulators).
- `qa-ebe6d9cf813777ce-00005-comparison`
  - Q before: How does the interaction strength of the nearest-neighbor coupling in the 2D Ising Hamiltonian compare to the interaction mechanism in the XY dataset?
  - Q after: In the TetrisCNN paper, what is the nearest-neighbor interaction strength U/h for the 2D Ising Rydberg-atom dataset (lattice spacing a = 10 um, n = 75 Rydberg state)?
  - GT before: The 2D Ising Hamiltonian has a nearest-neighbor interaction strength of U/h ≈ 1.95 MHz, whereas the XY dataset utilizes strong electric dipole–dipole interactions that generate the Rydberg blockade mechanism.
  - GT after: U/h is approximately 1.95 MHz.
- `qa-fb85b2f19bbf8e6f-00001-comparison`
  - Q before: How do the measurement patterns of the tested embedding models compare to the ideal measurement space depicted in Panel A of Figure 1?
  - Q after: In 'Embedding Models Measure in Peculiar Ways', how do the measurement patterns of the tested embedding models compare to the ideal measurement space depicted in Panel A of Figure 1?
- `qa-fb85b2f19bbf8e6f-00002-comparison`
  - Q before: How does the paper describe the association of embedding similarity for physical measurements in comparison to lexical and semantic/numerical similarity?
  - Q after: In 'Embedding Models Measure in Peculiar Ways', with which kind of similarity is the embedding similarity of physical measurements found to be strongly associated, lexical or semantic/numerical?
- `qa-fb85b2f19bbf8e6f-00005-numeric`
  - Q before: What is the upper limit of the 'log' measurement range described in the study?
  - Q after: In 'Embedding Models Measure in Peculiar Ways', what is the upper limit of the 'log' measurement range?

## Rules for the record

- AI suggestions and the LLM pre-screen are advisory. They never set "verified". Only the owner's clicks in the review tool did.
- This is enforced in code: `load_verified()` returns only pairs whose ledger status is verified, and nothing reads the suggestion or pre-screen files for that purpose.
- Rejected pairs stay in `golden_dataset.jsonl` but are excluded by `load_verified()`.

## File index

- `golden_dataset.before_review.jsonl`: Dataset before review (commit b8128b3), byte-identical
- `golden_review.jsonl`: Human decisions ledger (one row per pair, latest decision)
- `review_suggestions.jsonl`: Sonnet subagent advisory suggestions
- `review_sheet.html`: Rendered sheet the owner read first
- `golden_prescreen.jsonl`: LLM pre-screen (advisory)
- `changes.jsonl`: One row per pair joining all of the above with before and after text
- `README.md`: This file
