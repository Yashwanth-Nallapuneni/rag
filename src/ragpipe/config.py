"""Typed configuration loaded from YAML, overridable by environment.

Precedence (low -> high): config/default.yaml, config/<RAGPIPE_ENV>.yaml,
explicit overrides, RAGPIPE_* environment variables.

`Settings.fingerprint()` hashes the whole resolved config. Evaluation results
record that hash, so a change in chunk size or reranker model is visible as a
different fingerprint rather than an unexplained metric shift.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_dotenv_once() -> None:
    """Load .env into the process environment at import time.

    pydantic-settings reads .env for RAGPIPE_* fields only, but provider
    credentials are plain env vars that the provider classes fetch with
    os.getenv -- so without this, a GROQ_API_KEY sitting in .env is invisible
    and every script fails with "requires GROQ_API_KEY" while the key is right
    there. Existing environment variables win, so an explicitly exported key
    is never silently overridden by a stale file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv ships with pydantic-settings
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


_load_dotenv_once()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class CorpusConfig(BaseModel):
    name: str = "arxiv-ml"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    @property
    def raw_path(self) -> Path:
        return _resolve(self.raw_dir)

    @property
    def processed_path(self) -> Path:
        return _resolve(self.processed_dir)


class IngestConfig(BaseModel):
    strip_pdf_headers_footers: bool = True
    header_footer_margin_ratio: float = 0.08
    min_repeat_ratio: float = 0.5
    drop_references_section: bool = True
    min_chars_per_doc: int = 500


class ChunkingConfig(BaseModel):
    strategy: Literal["token_window", "heading_aware"] = "token_window"
    chunk_size: int = 650
    chunk_overlap: int = 100
    min_chunk_tokens: int = 40
    respect_headings: bool = True
    tokenizer: str = "cl100k_base"

    @model_validator(mode="after")
    def _check_overlap(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not 200 <= self.chunk_size <= 2000:
            raise ValueError("chunk_size outside sane range (200-2000 tokens)")
        return self


class EmbeddingsConfig(BaseModel):
    provider: Literal["local", "openai", "mock"] = "local"
    model: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True
    query_prefix: str = ""
    device: str = "auto"
    cache_dir: str = ".ragpipe_cache/embeddings"


class VectorStoreConfig(BaseModel):
    provider: Literal["chroma"] = "chroma"
    path: str = "data/chroma"
    collection: str = "ragpipe_chunks"
    distance: Literal["cosine", "l2", "ip"] = "cosine"

    @property
    def store_path(self) -> Path:
        return _resolve(self.path)


class BM25Config(BaseModel):
    k1: float = 1.5
    b: float = 0.75
    index_path: str = "data/processed/bm25_index.pkl"

    @property
    def path(self) -> Path:
        return _resolve(self.index_path)


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = 5
    candidate_k: int = 30
    fusion: Literal["rrf", "weighted_sum"] = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    min_score: float = 0.0
    bm25: BM25Config = Field(default_factory=BM25Config)

    @model_validator(mode="after")
    def _check(self) -> RetrievalConfig:
        if self.top_k > self.candidate_k:
            raise ValueError("top_k cannot exceed candidate_k")
        return self


class RerankConfig(BaseModel):
    enabled: bool = True
    provider: Literal["local", "cohere", "mock", "none"] = "local"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5
    batch_size: int = 16
    device: str = "auto"
    score_threshold: float | None = None


class LLMConfig(BaseModel):
    provider: Literal["mock", "anthropic", "openai", "groq", "openrouter", "ollama"] = "mock"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: int = 60
    max_retries: int = 3

    # Overrides that make an arbitrary OpenAI-compatible endpoint (DeepSeek,
    # Together, a local vLLM server, ...) work with provider: openai (or a
    # new one-line registry entry) without any new provider code: point
    # base_url at the host and api_key_env at whatever env var holds its key.
    base_url: str | None = None
    # Reasoning models (Groq's gpt-oss-*, for one) spend output tokens on
    # reasoning before emitting any content, and those tokens are billed and
    # counted against rate limits. Measured on gpt-oss-120b: a three-word
    # answer cost 148 reasoning + 15 content tokens. "low" cut reasoning ~40%
    # with identical verdicts on a claim-check probe, which matters a lot on a
    # token-capped free tier. None omits the parameter entirely, so providers
    # that reject it are unaffected.
    reasoning_effort: str | None = None
    api_key_env: str | None = None

    # Client-side pacing so a rate-limited (e.g. free-tier) key does not just
    # 429 in a loop. None means unlimited -- paid-tier users see no change.
    # Groq's own free-tier ceilings are filled in as defaults by GroqLLM when
    # these are left unset (see providers/llm/groq.py).
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_day: int | None = None

    # OpenRouter attribution headers (HTTP-Referer / X-Title). Ignored by
    # every other provider.
    http_referer: str | None = None
    app_title: str | None = None


class PromptsConfig(BaseModel):
    dir: str = "prompts"
    answer_version: str = "v2"
    claim_check_version: str = "v1"

    @property
    def path(self) -> Path:
        return _resolve(self.dir)


class GenerationConfig(BaseModel):
    max_context_tokens: int = 6000
    include_locators: bool = True


class CitationConfig(BaseModel):
    enforce: bool = True
    # A raw cross-encoder logit; re-measure if the reranker model changes.
    # None disables the relevance gate.
    min_relevance_score: float | None = -7.0
    min_supported_ratio: float = 0.8
    verifier: Literal["lexical", "llm", "hybrid"] = "hybrid"
    lexical_threshold: float = 0.45
    require_citation_per_sentence: bool = True
    refusal_message: str = (
        "I can't answer that from the indexed documents."
    )


class EvalThresholds(BaseModel):
    faithfulness: float = 0.80
    answer_relevancy: float = 0.70
    context_precision: float = 0.65
    context_recall: float = 0.65
    refusal_accuracy: float = 0.80


class EvaluationConfig(BaseModel):
    dataset_path: str = "eval/golden_dataset.jsonl"
    results_dir: str = "eval_results"
    metrics: list[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
    )
    sample_size: int | None = None
    concurrency: int = 4
    thresholds: EvalThresholds = Field(default_factory=EvalThresholds)

    @property
    def dataset(self) -> Path:
        return _resolve(self.dataset_path)

    @property
    def results(self) -> Path:
        return _resolve(self.results_dir)


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_output: bool = Field(default=False, alias="json")
    model_config = {"populate_by_name": True}


class _YamlSource(PydanticBaseSettingsSource):
    """Lowest-priority settings source: the merged YAML layers.

    Registering YAML as a *source* rather than passing it as init kwargs is
    what makes RAGPIPE_* env vars able to override the file -- init kwargs
    outrank env in pydantic-settings, YAML sources do not.
    """

    _data: dict[str, Any] = {}

    def get_field_value(self, field, field_name):  # pragma: no cover - unused
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(type(self)._data)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGPIPE_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    citation: CitationConfig = Field(default_factory=CitationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Priority, highest first: explicit overrides, env, .env, YAML.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls),
            file_secret_settings,
        )

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def fingerprint(self) -> str:
        """Stable hash of the resolved config, recorded with eval runs."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def describe(self) -> dict[str, Any]:
        """Compact summary for logs, API /health and eval result headers."""
        return {
            "fingerprint": self.fingerprint(),
            "retrieval_mode": self.retrieval.mode,
            "top_k": self.retrieval.top_k,
            "candidate_k": self.retrieval.candidate_k,
            "fusion": self.retrieval.fusion,
            "rerank": self.rerank.model if self.rerank.enabled else "off",
            "embeddings": f"{self.embeddings.provider}:{self.embeddings.model}",
            "llm": f"{self.llm.provider}:{self.llm.model or 'default'}",
            "chunking": f"{self.chunking.chunk_size}/{self.chunking.chunk_overlap}",
            "prompt_version": self.prompts.answer_version,
            "citation_enforced": self.citation.enforce,
        }


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(
    env: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Build Settings from YAML layers plus overrides. Env vars win last."""
    data: dict[str, Any] = {}
    default_file = CONFIG_DIR / "default.yaml"
    if default_file.exists():
        data = yaml.safe_load(default_file.read_text()) or {}

    env = env or os.getenv("RAGPIPE_ENV")
    if env:
        env_file = CONFIG_DIR / f"{env}.yaml"
        if env_file.exists():
            data = _deep_merge(data, yaml.safe_load(env_file.read_text()) or {})
        else:
            raise FileNotFoundError(f"No config for RAGPIPE_ENV={env}: {env_file}")

    _YamlSource._data = data
    try:
        return Settings(**(overrides or {}))
    finally:
        _YamlSource._data = {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
