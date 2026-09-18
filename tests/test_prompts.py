from __future__ import annotations

import pytest

from ragpipe.prompts import PromptError, list_prompts, load_prompt


@pytest.fixture
def prompts_dir(settings):
    return str(settings.prompts.path)


def test_versions_are_discoverable(prompts_dir):
    available = list_prompts(prompts_dir)
    assert "v1" in available["answer"] and "v2" in available["answer"]


def test_active_version_loads(settings, prompts_dir):
    p = load_prompt("answer", settings.prompts.answer_version, prompts_dir)
    assert p.id == f"answer/{settings.prompts.answer_version}"
    assert p.system and p.user_template


def test_render_substitutes_variables(prompts_dir):
    p = load_prompt("answer", "v2", prompts_dir)
    system, user = p.render(context="[S1] (T | p. 1)\nBody.", question="What?")
    assert "Body." in user and "What?" in user
    assert "$context" not in user and "$question" not in user
    assert system


def test_missing_variable_is_an_error_not_a_blank(prompts_dir):
    """A silently empty context block makes the model answer from memory,
    which looks like a faithfulness bug rather than a prompt bug."""
    p = load_prompt("answer", "v2", prompts_dir)
    with pytest.raises(PromptError):
        p.render(context="only context")


def test_unknown_version_lists_alternatives(prompts_dir):
    with pytest.raises(PromptError) as exc:
        load_prompt("answer", "v99", prompts_dir)
    assert "v1" in str(exc.value) or "v2" in str(exc.value)


def test_v2_declares_a_refusal_sentinel(prompts_dir):
    """The pipeline detects refusals by sentinel, not by prose matching."""
    p = load_prompt("answer", "v2", prompts_dir)
    assert p.metadata.get("refusal_sentinel") == "INSUFFICIENT_CONTEXT"


def test_v2_warns_the_model_about_source_reference_markers(prompts_dir):
    p = load_prompt("answer", "v2", prompts_dir)
    assert "[S1]" in p.system
    assert "reference" in p.system.lower()
