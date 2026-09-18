"""OpenRouter chat completions backend (OpenAI-compatible).

Default model: `meta-llama/llama-3.1-8b-instruct`. Reasoning: OpenRouter
fronts dozens of models behind one API, so the "right" default is whichever
is cheap enough to eval against freely and still capable enough to produce a
sane extractive/grounded answer for this pipeline's citation-enforcement
tests -- an 8B instruction-tuned Llama is the same tier of model as Groq's
own free `llama-3.1-8b-instant` fallback, so switching between the two
providers for a quick smoke test compares like with like. Swap it via
`llm.model` for anything else OpenRouter hosts; nothing else about this class
depends on the choice.

OpenRouter honours two optional attribution headers, `HTTP-Referer` and
`X-Title`, which affects how a request shows up in OpenRouter's own
dashboards. They're set from config (`llm.http_referer` / `llm.app_title`),
never hardcoded to anything personally identifying.
"""

from __future__ import annotations

from typing import Any

from .openai_compatible import OpenAICompatibleLLM

DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"
BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_REFERER = "https://github.com/ragpipe-project/ragpipe"
DEFAULT_TITLE = "ragpipe RAG eval"


class OpenRouterLLM(OpenAICompatibleLLM):
    """LLMProvider backed by OpenRouter's OpenAI-compatible chat completions API."""

    name = "openrouter"
    env_var = "OPENROUTER_API_KEY"
    base_url = BASE_URL
    default_model = DEFAULT_MODEL

    def __init__(self, cfg: Any):
        referer = getattr(cfg, "http_referer", None) or DEFAULT_REFERER
        title = getattr(cfg, "app_title", None) or DEFAULT_TITLE
        self.extra_headers = {"HTTP-Referer": referer, "X-Title": title}
        super().__init__(cfg)
