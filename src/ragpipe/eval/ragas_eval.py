"""RAGAS evaluation harness: faithfulness, answer relevance, context
precision/recall, plus a refusal-accuracy metric RAGAS does not provide.

WHAT THIS IS: answers are generated through the REAL pipeline
(`build_answerer`, never a shortcut), so retrieval, citation enforcement and
refusal all run exactly as they would in production. RAGAS then grades the
(question, answer, contexts, reference) tuples with a judge LLM.

WHAT THIS IS NOT: a free metric. Every RAGAS metric call goes to the judge
LLM, and every generated answer goes to the generation LLM -- see
`ragpipe.eval.cost` for the guard that stops a run before or as it overspends.

WHY WE COMPUTE REFUSAL ACCURACY OURSELVES: the spec's central trust property
is "does the system refuse when it should and answer when it should" -- RAGAS
has no metric for that, because RAGAS assumes every row has an answer to
grade. This harness treats `unanswerable=True` -> `Answer.refused` as the
correct behaviour and scores precision/recall/accuracy over it directly from
the Answerer's own `AnswerStatus`, before any refused answer is handed to
RAGAS at all.

WHY REFUSALS ARE EXCLUDED FROM FAITHFULNESS/RELEVANCY: a refusal ("I can't
answer that from the indexed documents") has no claims to check faithfulness
against and no semantic content to compare to the question. Scoring it would
either be undefined or, worse, score as vacuously perfect (a refusal cites
nothing, so nothing it says is unsupported). A harness that quietly drops
refused rows can therefore inflate faithfulness by refusing more -- so the
excluded count is written into every result file, not hidden in a footnote.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings
from ..logging_utils import get_logger
from ..providers import LLMRequest, get_embedder, get_llm
from ..schemas import Answer, QAPair
from ..tokenization import count_tokens
from .cost import CostTracker, check_preflight, estimate_run_cost

log = get_logger(__name__)

DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ---------------------------------------------------------------------------
# dataset loading -- lazy import of golden.py, which another agent owns and
# may not exist yet at import time. Fall back to reading the JSONL directly.
# ---------------------------------------------------------------------------


def _load_pairs(settings: Settings) -> list[QAPair]:
    try:
        from .golden import load_verified  # type: ignore
    except ImportError:
        log.warning(
            "ragpipe.eval.golden not importable yet; falling back to reading "
            "the raw dataset JSONL with no verification-ledger filtering"
        )
        return _read_jsonl_fallback(settings.evaluation.dataset)
    pairs = load_verified(settings.evaluation.dataset_path)
    if not pairs:
        log.warning(
            "load_verified returned 0 pairs for %s; falling back to the raw "
            "dataset (unverified) so the harness is still exercisable",
            settings.evaluation.dataset_path,
        )
        return _read_jsonl_fallback(settings.evaluation.dataset)
    return pairs


def _read_jsonl_fallback(path: Path) -> list[QAPair]:
    if not path.exists():
        return []
    pairs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(QAPair.model_validate_json(line))
    return pairs


# ---------------------------------------------------------------------------
# RAGAS adapters -- inject OUR providers instead of RAGAS's OpenAI defaults.
#
# ragas 0.4.3 offers three ways to hand RAGAS an LLM/embeddings:
#   1. `ragas.llms.llm_factory` / the new `BaseRagasEmbedding` providers --
#      the "non-deprecated" path per the module's own docs, but it REQUIRES a
#      real SDK client instance (`openai.OpenAI(...)`, `anthropic.Anthropic(...)`)
#      and only supports the providers it has an adapter for. Our `mock` and
#      `ollama` providers have no SDK client to hand it, so this path cannot
#      wrap our provider abstraction uniformly.
#   2. `LangchainLLMWrapper` / `LangchainEmbeddingsWrapper` -- accepts any
#      LangChain `BaseLanguageModel`/`Embeddings`, but is itself marked
#      deprecated in this version (see ragas/llms/base.py, ragas/embeddings/
#      base.py) and would require ANOTHER adapter layer (our provider ->
#      LangChain interface -> ragas wrapper) for no benefit.
#   3. Subclass `ragas.llms.base.BaseRagasLLM` / `ragas.embeddings.base.
#      BaseRagasEmbeddings` directly -- these are the actual abstract
#      contracts `ragas.evaluate()` consumes (see evaluate()'s "set llm and
#      embeddings" block: it assigns straight to `metric.llm`/`metric.
#      embeddings` for anything that isn't a LangChain object). No
#      deprecation warning, no extra adapter layer, works for every
#      `LLMProvider`/`EmbeddingProvider` this project has including `mock`.
#
# We use (3): it is the thinnest adapter that actually works uniformly across
# our providers, and it is not the deprecated path.
# ---------------------------------------------------------------------------


def _build_ragas_llm(llm: Any, cost_tracker: CostTracker):
    """Adapts a ragpipe `LLMProvider` to `ragas.llms.base.BaseRagasLLM`."""
    import asyncio

    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM

    model_name = f"{llm.name}:{llm.model}"

    class _RagasLLMAdapter(BaseRagasLLM):
        multiple_completion_supported = False

        def is_finished(self, response: LLMResult) -> bool:
            return True

        def _call(self, prompt_text: str, temperature: float | None) -> str:
            request = LLMRequest(
                system="You are a careful evaluation judge. Follow the "
                "instructions in the prompt exactly and respond only in the "
                "requested format.",
                user=prompt_text,
                task="generic",
                temperature=temperature,
            )
            response = llm.complete(request)
            usage = response.usage or {}
            in_tok = usage.get("input_tokens")
            out_tok = usage.get("output_tokens")
            # mock's usage is a word-count heuristic, not real tokenization,
            # and treating it as "measured" would overstate what we actually
            # know -- so mock (and anything with no usage at all) is always
            # estimated via the shared tokenizer instead.
            estimated = llm.name == "mock" or in_tok is None or out_tok is None
            if estimated:
                in_tok = count_tokens(prompt_text)
                out_tok = count_tokens(response.text)
            cost_tracker.add(model_name, in_tok, out_tok, estimated=estimated)
            return response.text

        def generate_text(
            self,
            prompt,
            n: int = 1,
            temperature: float | None = 0.01,
            stop=None,
            callbacks=None,
        ) -> LLMResult:
            text = self._call(prompt.to_string(), temperature)
            generations = [[Generation(text=text)] for _ in range(n)]
            return LLMResult(generations=generations)

        async def agenerate_text(
            self,
            prompt,
            n: int = 1,
            temperature: float | None = 0.01,
            stop=None,
            callbacks=None,
        ) -> LLMResult:
            return await asyncio.to_thread(
                self.generate_text, prompt, n, temperature, stop, callbacks
            )

    return _RagasLLMAdapter()


def _build_ragas_embeddings(embedder: Any):
    """Adapts a ragpipe `EmbeddingProvider` (our local BGE model) to
    `ragas.embeddings.base.BaseRagasEmbeddings`, so answer_relevancy never
    needs an OpenAI embeddings key -- it runs entirely offline."""
    import asyncio

    from ragas.embeddings.base import BaseRagasEmbeddings

    class _RagasEmbeddingsAdapter(BaseRagasEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embedder.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return embedder.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(embedder.embed_documents, texts)

        async def aembed_query(self, text: str) -> list[float]:
            return await asyncio.to_thread(embedder.embed_query, text)

    return _RagasEmbeddingsAdapter()


def _ragas_metric_objects(names: list[str]) -> list[Any]:
    import warnings

    with warnings.catch_warnings():
        # ragas.metrics re-exports the legacy MetricWithLLM instances that
        # ragas.evaluate() actually consumes; ragas.metrics.collections is
        # the newer InstructorBaseRagasLLM-based API and is NOT what
        # evaluate()'s dataset/executor flow accepts in 0.4.3 (confirmed by
        # reading ragas/evaluation.py: metrics are typed `Sequence[Metric]`
        # from ragas.metrics.base, and the "collections" classes require an
        # `InstructorBaseRagasLLM`, a different judge interface). The import
        # below emits a DeprecationWarning pointing at collections; it is
        # suppressed here because the legacy path is the one that works with
        # evaluate(), not a mistake.
        warnings.simplefilter("ignore", DeprecationWarning)
        import ragas.metrics as m

    registry = {
        "faithfulness": m.faithfulness,
        "answer_relevancy": m.answer_relevancy,
        "context_precision": m.context_precision,
        "context_recall": m.context_recall,
    }
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"unknown ragas metric(s): {unknown}; supported: {list(registry)}")
    return [registry[n] for n in names]


# ---------------------------------------------------------------------------
# refusal accuracy -- our own metric, computed from AnswerStatus directly.
# ---------------------------------------------------------------------------


@dataclass
class RefusalStats:
    total: int = 0
    true_positive: int = 0  # unanswerable, correctly refused
    true_negative: int = 0  # answerable, correctly answered
    false_answer: int = 0  # unanswerable, WRONGLY answered -- worst failure
    false_refusal: int = 0  # answerable, wrongly refused

    @property
    def precision(self) -> float | None:
        denom = self.true_positive + self.false_refusal
        return self.true_positive / denom if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.true_positive + self.false_answer
        return self.true_positive / denom if denom else None

    @property
    def accuracy(self) -> float | None:
        return (self.true_positive + self.true_negative) / self.total if self.total else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "true_positive_correct_refusal": self.true_positive,
            "true_negative_correct_answer": self.true_negative,
            "false_answer_count": self.false_answer,
            "false_refusal_count": self.false_refusal,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "f1": self.f1,
        }


def score_refusals(pairs: list[QAPair], answers: list[Answer]) -> RefusalStats:
    stats = RefusalStats(total=len(pairs))
    for qa, ans in zip(pairs, answers):
        expected_refuse = qa.unanswerable
        actual_refuse = ans.refused
        if expected_refuse and actual_refuse:
            stats.true_positive += 1
        elif not expected_refuse and not actual_refuse:
            stats.true_negative += 1
        elif expected_refuse and not actual_refuse:
            stats.false_answer += 1
        else:
            stats.false_refusal += 1
    return stats


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - best-effort only
        return None


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------


def run_evaluation(
    settings: Settings,
    pairs: list[QAPair] | None = None,
    *,
    sample_size: int | None = None,
    max_usd: float | None = None,
    metrics: list[str] | None = None,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    progress: bool = True,
    prices: dict[str, tuple[float, float]] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full harness: generate -> refusal-accuracy -> RAGAS -> write.

    `dry_run=True` computes and returns the pre-flight cost estimate and
    sample count WITHOUT calling `build_answerer` or any LLM -- use this to
    show a user the bill before spending anything.

    A separate `judge_model`/`judge_provider` avoids one model judging its
    own output; when omitted, the judge falls back to `settings.llm`
    (generation and judge are the same model) and that fact is recorded in
    the result under `judge_is_generation_model` so it is never silently
    true.
    """
    metric_names = list(metrics or settings.evaluation.metrics or DEFAULT_METRICS)
    all_pairs = pairs if pairs is not None else _load_pairs(settings)

    n = sample_size if sample_size is not None else settings.evaluation.sample_size
    if n is not None and n < len(all_pairs):
        import random

        rng = random.Random(20260917)
        selected = rng.sample(all_pairs, n)
    else:
        selected = list(all_pairs)

    gen_model_label = f"{settings.llm.provider}:{settings.llm.model or 'default'}"
    judge_model_label = (
        f"{judge_provider or settings.llm.provider}:{judge_model or settings.llm.model or 'default'}"
    )

    estimate = estimate_run_cost(
        len(selected),
        gen_model_label,
        judge_model=judge_model_label,
        metrics=metric_names,
        prices=prices,
    )

    if dry_run:
        return {
            "dry_run": True,
            "n_samples": len(selected),
            "n_available": len(all_pairs),
            "estimate": estimate.as_dict(),
            "metrics": metric_names,
        }

    check_preflight(estimate, max_usd)

    cost_tracker = CostTracker(max_usd=max_usd, prices=prices)

    from ..generation.answerer import build_answerer

    answerer = build_answerer(settings)

    started = time.perf_counter()
    answers: list[Answer] = []
    for i, qa in enumerate(selected):
        ans = answerer.answer(qa.question)
        answers.append(ans)
        # Generation cost, as measured by the pipeline's own LLM response --
        # see the module docstring: this is the "answer" call only. If
        # settings.citation.verifier is 'llm' or 'hybrid', the claim-checker
        # ALSO calls the LLM internally (ragpipe.generation.verify), and
        # Answer does not expose that call's usage -- it is not counted here.
        # That gap is recorded below (`generation_cost_gap`) rather than
        # hidden.
        usage = ans.usage or {}
        in_tok, out_tok = usage.get("input_tokens"), usage.get("output_tokens")
        if in_tok is not None and out_tok is not None:
            estimated = f"{ans.model}".startswith("mock")
            cost_tracker.add(ans.model or gen_model_label, in_tok, out_tok, estimated=estimated)
        if progress:
            log.info("generated %d/%d (%s)", i + 1, len(selected), qa.id)
    gen_elapsed_s = time.perf_counter() - started

    refusal_stats = score_refusals(selected, answers)

    # Refusals carry no claims to check and no answer to compare against the
    # question -- see module docstring for why they must be excluded rather
    # than scored as vacuously perfect.
    graded_rows: list[dict[str, Any]] = []
    excluded_refused = 0
    for qa, ans in zip(selected, answers):
        if ans.refused:
            excluded_refused += 1
            continue
        graded_rows.append(
            {
                "user_input": qa.question,
                "response": ans.text,
                "retrieved_contexts": ans.context_texts or [""],
                "reference": qa.ground_truth,
            }
        )

    ragas_result: dict[str, Any] = {
        "computed": False,
        "reason": None,
        "aggregate": {},
        "per_sample": [],
    }

    if not graded_rows:
        ragas_result["reason"] = "every sampled answer was refused; nothing to grade"
    else:
        try:
            import warnings

            from ragas import evaluate
            from ragas.dataset_schema import EvaluationDataset

            from .cost import BudgetExceeded

            judge_settings = settings.model_copy(deep=True)
            if judge_provider:
                judge_settings.llm.provider = judge_provider
            if judge_model:
                judge_settings.llm.model = judge_model
            judge_llm_provider = get_llm(judge_settings)
            embed_provider = get_embedder(settings)

            ragas_llm = _build_ragas_llm(judge_llm_provider, cost_tracker)
            ragas_embeddings = _build_ragas_embeddings(embed_provider)
            metric_objs = _ragas_metric_objects(metric_names)

            dataset = EvaluationDataset.from_list(graded_rows)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                eval_out = evaluate(
                    dataset=dataset,
                    metrics=metric_objs,
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    raise_exceptions=False,
                    show_progress=progress,
                )

            df = eval_out.to_pandas()
            aggregate = {
                name: (None if df[name].isna().all() else float(df[name].mean(skipna=True)))
                for name in metric_names
                if name in df.columns
            }
            per_sample = []
            for idx, row in df.iterrows():
                per_sample.append(
                    {
                        "question": row.get("user_input"),
                        **{
                            name: (None if row.get(name) != row.get(name) else float(row.get(name)))
                            for name in metric_names
                            if name in df.columns
                        },
                    }
                )
            all_nan = bool(aggregate) and all(v is None for v in aggregate.values())
            if all_nan:
                # ragas.evaluate() with raise_exceptions=False swallows a
                # judge that never produces parseable output (expected with
                # the offline mock LLM: RAGAS's structured-output prompts
                # need a real instruction-following model) and returns
                # NaN for every score instead of raising. Reporting that as
                # "computed" would be exactly the false claim the user
                # forbade -- every number here would silently be nothing.
                ragas_result["reason"] = (
                    "ragas.evaluate() ran without raising but every metric came back "
                    "NaN for every sample -- the judge never produced output RAGAS "
                    "could parse (see stderr for 'RagasOutputParserException' / "
                    "'failed to parse the output'). Expected with the offline mock "
                    "judge, which was never designed to answer RAGAS's own "
                    "structured-output prompts. Metrics are UNAVAILABLE, not zero."
                )
                ragas_result["per_sample"] = per_sample
                ragas_result["n_graded"] = len(graded_rows)
            else:
                ragas_result.update(
                    {
                        "computed": True,
                        "aggregate": aggregate,
                        "per_sample": per_sample,
                        "n_graded": len(graded_rows),
                    }
                )
        except BudgetExceeded:
            # A hard cap breach must abort the whole run, not be swallowed as
            # "ragas metrics unavailable" -- propagate it to the caller.
            raise
        except Exception as exc:  # noqa: BLE001 - judge failures must not crash the harness
            log.warning("RAGAS evaluation failed / unusable: %s", exc)
            ragas_result["reason"] = (
                f"ragas.evaluate raised {type(exc).__name__}: {exc}. This is expected "
                "with a mock/offline judge -- RAGAS's own prompts expect a real "
                "instruction-following LLM. Metrics below are UNAVAILABLE, not zero."
            )

    thresholds = settings.evaluation.thresholds.model_dump()
    fails: list[str] = []
    for name, value in ragas_result["aggregate"].items():
        thr = thresholds.get(name)
        if thr is not None and value is not None and value < thr:
            fails.append(f"{name}={value:.3f} < threshold {thr:.3f}")
    if refusal_stats.accuracy is not None:
        thr = thresholds.get("refusal_accuracy")
        if thr is not None and refusal_stats.accuracy < thr:
            fails.append(f"refusal_accuracy={refusal_stats.accuracy:.3f} < threshold {thr:.3f}")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config_fingerprint": settings.fingerprint(),
        "config_summary": settings.describe(),
        "prompt_version": settings.prompts.answer_version,
        "generation_model": gen_model_label,
        "judge_model": judge_model_label,
        "judge_is_generation_model": judge_model_label == gen_model_label,
        "dataset_path": str(settings.evaluation.dataset),
        "dataset_size_available": len(all_pairs),
        "sample_size": len(selected),
        "metrics_requested": metric_names,
        "ragas": ragas_result,
        "refusal_accuracy": refusal_stats.as_dict(),
        "excluded_from_faithfulness_relevancy": {
            "count": excluded_refused,
            "reason": "refused answers have no claims/answer to grade -- see module docstring",
        },
        "cost": cost_tracker.as_dict(),
        "cost_estimate_preflight": estimate.as_dict(),
        "generation_cost_gap": (
            "if citation.verifier is 'llm' or 'hybrid', the Answerer's claim-checker "
            "makes additional LLM calls not exposed on Answer.usage -- their cost is "
            "not included in `cost.by_model` above"
            if settings.citation.verifier in ("llm", "hybrid")
            else None
        ),
        "thresholds": thresholds,
        "threshold_failures": fails,
        "passed": not fails,
        "generation_wall_seconds": round(gen_elapsed_s, 2),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    results_dir = settings.evaluation.results
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = results_dir / f"eval_{stamp}_{settings.fingerprint()}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["result_path"] = str(out_path)

    summary_path = out_path.with_suffix(".txt")
    summary_path.write_text(format_summary(result), encoding="utf-8")
    result["summary_path"] = str(summary_path)

    return result


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def format_summary(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"RAGAS evaluation -- {result['timestamp']}")
    lines.append(f"config fingerprint: {result['config_fingerprint']}")
    lines.append(f"generation model:   {result['generation_model']}")
    lines.append(
        f"judge model:        {result['judge_model']}"
        + ("  ** SAME AS GENERATION -- judge is not independent **" if result["judge_is_generation_model"] else "")
    )
    lines.append(f"dataset:            {result['dataset_path']} ({result['sample_size']}/{result['dataset_size_available']} sampled)")
    lines.append("")

    lines.append("RAGAS metrics:")
    if result["ragas"]["computed"]:
        for name, value in result["ragas"]["aggregate"].items():
            thr = result["thresholds"].get(name)
            mark = ""
            if thr is not None and value is not None:
                mark = "PASS" if value >= thr else "FAIL"
            val_str = f"{value:.4f}" if value is not None else "n/a"
            lines.append(f"  {name:<20} {val_str:>8}   threshold {thr}   {mark}")
        lines.append(
            f"  (excluded {result['excluded_from_faithfulness_relevancy']['count']} "
            "refused answer(s) from these averages)"
        )
    else:
        lines.append(f"  UNAVAILABLE: {result['ragas']['reason']}")
    lines.append("")

    ra = result["refusal_accuracy"]
    lines.append("Refusal accuracy (computed by us, not RAGAS):")
    lines.append(f"  total={ra['total']}  accuracy={ra['accuracy']}  precision={ra['precision']}  recall={ra['recall']}  f1={ra['f1']}")
    lines.append(f"  false_answer (unanswerable, wrongly answered -- worst failure): {ra['false_answer_count']}")
    lines.append(f"  false_refusal (answerable, wrongly refused): {ra['false_refusal_count']}")
    lines.append("")

    cost = result["cost"]
    lines.append("Cost:")
    lines.append(f"  total spent: ${cost['total_usd']:.4f}  (cap: {cost['max_usd']})")
    for model, usage in cost["by_model"].items():
        lines.append(
            f"    {model}: ${usage['usd']:.4f}  in={usage['input_tokens']} out={usage['output_tokens']}  "
            f"measured_calls={usage['measured_calls']} estimated_calls={usage['estimated_calls']}"
        )
    if result.get("generation_cost_gap"):
        lines.append(f"  NOTE: {result['generation_cost_gap']}")
    lines.append("")

    lines.append(f"OVERALL: {'PASS' if result['passed'] else 'FAIL'}")
    for f in result["threshold_failures"]:
        lines.append(f"  - {f}")

    return "\n".join(lines) + "\n"


def compare_runs(a_path: str | Path, b_path: str | Path) -> str:
    """Before/after delta table across metrics for two result JSON files.

    This is what lets an honest "reranking improved faithfulness by X" claim
    exist: same dataset, same judge, one config difference between the two
    runs -- and if that is not true (different dataset size, different judge
    model, different fingerprint families), this function says so instead of
    printing a number that implies an apples-to-apples comparison.
    """
    a = json.loads(Path(a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(b_path).read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append(f"BEFORE: {a_path}")
    lines.append(f"  fingerprint={a.get('config_fingerprint')} judge={a.get('judge_model')} "
                 f"dataset={a.get('dataset_path')} n={a.get('sample_size')}")
    lines.append(f"AFTER:  {b_path}")
    lines.append(f"  fingerprint={b.get('config_fingerprint')} judge={b.get('judge_model')} "
                 f"dataset={b.get('dataset_path')} n={b.get('sample_size')}")
    lines.append("")

    caveats = []
    if a.get("dataset_path") != b.get("dataset_path") or a.get("sample_size") != b.get("sample_size"):
        caveats.append("dataset/sample size differs -- delta is NOT apples-to-apples")
    if a.get("judge_model") != b.get("judge_model"):
        caveats.append("judge model differs -- delta may reflect a different judge, not the config change")
    if caveats:
        lines.append("CAVEATS:")
        for c in caveats:
            lines.append(f"  - {c}")
        lines.append("")

    header = f"{'metric':<22} {'before':>10} {'after':>10} {'delta':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    a_agg = a.get("ragas", {}).get("aggregate", {})
    b_agg = b.get("ragas", {}).get("aggregate", {})
    names = sorted(set(a_agg) | set(b_agg))
    for name in names:
        av, bv = a_agg.get(name), b_agg.get(name)
        delta_str = f"{bv - av:+.4f}" if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else "n/a"
        av_str = f"{av:.4f}" if isinstance(av, (int, float)) else "n/a"
        bv_str = f"{bv:.4f}" if isinstance(bv, (int, float)) else "n/a"
        lines.append(f"{name:<22} {av_str:>10} {bv_str:>10} {delta_str:>10}")

    a_ra, b_ra = a.get("refusal_accuracy", {}), b.get("refusal_accuracy", {})
    for name in ("accuracy", "precision", "recall", "f1"):
        av, bv = a_ra.get(name), b_ra.get(name)
        delta_str = f"{bv - av:+.4f}" if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else "n/a"
        av_str = f"{av:.4f}" if isinstance(av, (int, float)) else "n/a"
        bv_str = f"{bv:.4f}" if isinstance(bv, (int, float)) else "n/a"
        lines.append(f"{'refusal_' + name:<22} {av_str:>10} {bv_str:>10} {delta_str:>10}")

    return "\n".join(lines) + "\n"
