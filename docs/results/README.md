# Result files

Raw evidence behind the numbers quoted in the top-level README. These are
unchanged copies of files from the git-ignored `eval_results/` folder. The
`dataset_path` field and the header line in the `.txt` files hold the absolute
path of the machine that ran them.

| File | What it is |
| --- | --- |
| `eval_20261017T011212Z_5f0a41881655.json` and `.txt` | Final run with rerank on (`BAAI/bge-reranker-base`), commit d276390 |
| `eval_20261017T033055Z_4d290a449c2f.json` and `.txt` | Final run with rerank off (`--no-rerank`), commit 50fc0ca |
| `retrieval_experiment_20261016T200538Z.json` | Held-out reranker comparison (145 answerable and 26 unanswerable questions, 87 tune and 58 test) |

Both eval runs used the 171 human-verified golden pairs
(`pairs_human_verified` is true, `sample_size` is 171), generation model
`openrouter:openai/gpt-oss-120b` and judge model
`openrouter:meta-llama/llama-3.3-70b-instruct`. The `.txt` file is the human
readable summary of the matching `.json`.

## Where the headline numbers live

In each eval `.json`:

- `ragas.aggregate`: faithfulness, answer_relevancy, context_precision and
  context_recall
- `refusal_accuracy`: accuracy, precision, recall, f1, `false_answer_count`
  and `false_refusal_count`
- `excluded_from_faithfulness_relevancy.count`: refused answers left out of the
  RAGAS averages
- `config_summary.rerank`: which reranker the run used
- `git_commit`: the commit the run used
- `cost.total_usd`: measured spend

In `retrieval_experiment_20261016T200538Z.json`: each entry in `configs` has a
`test` block (held-out recall@1, recall@5, recall@8, mrr@10) and a `test_diff`
block with bootstrap confidence intervals against the baseline. The
`bge-base` entry is the reranker that was chosen.
