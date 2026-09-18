"""OpenAI chat completions backend.

The `openai` SDK is an optional dependency: it is imported lazily so the rest
of the pipeline (retrieval, citation enforcement, eval harness, and the
`mock` provider) keeps working in environments where it is not installed.

This is now a thin subclass of `OpenAICompatibleLLM` (see
`openai_compatible.py`), which also backs `GroqLLM` and `OpenRouterLLM` since
all three speak the same wire format. Behaviour is unchanged from before that
refactor: env `OPENAI_API_KEY`, no `base_url` override, default model
`gpt-4o-mini`.
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleLLM

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAILLM(OpenAICompatibleLLM):
    """LLMProvider backed by the OpenAI chat completions API."""

    name = "openai"
    env_var = "OPENAI_API_KEY"
    base_url = None
    default_model = DEFAULT_MODEL
