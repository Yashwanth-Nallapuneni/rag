"""Versioned prompt loading.

Prompts live in `prompts/<name>/<version>.yaml`, not in Python string
literals, because they are a tuned part of the system's behaviour: a prompt
change can move faithfulness as much as a retrieval change can. Keeping them
as versioned files means a metric shift can be attributed to a specific
prompt version, and an older version can be re-run for comparison.

The loaded version is recorded on every Answer and in every eval result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any

import yaml


class PromptError(RuntimeError):
    pass


@dataclass
class Prompt:
    name: str
    version: str
    system: str
    user_template: str
    description: str = ""
    variables: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.name}/{self.version}"

    def render(self, **kwargs: Any) -> tuple[str, str]:
        """Return (system, user). Missing variables are an error, not an
        empty string: a silently blank context block would make the model
        answer from parametric memory and look like a faithfulness bug."""
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise PromptError(f"{self.id} missing variables: {missing}")
        try:
            user = Template(self.user_template).substitute(**kwargs)
        except KeyError as exc:
            raise PromptError(f"{self.id} references undefined variable {exc}") from exc
        return self.system.strip(), user.strip()


def _load_file(path: Path) -> Prompt:
    data = yaml.safe_load(path.read_text()) or {}
    for required in ("name", "version", "system", "user"):
        if required not in data:
            raise PromptError(f"{path} is missing required key '{required}'")
    return Prompt(
        name=data["name"],
        version=str(data["version"]),
        system=data["system"],
        user_template=data["user"],
        description=data.get("description", ""),
        variables=list(data.get("variables", [])),
        metadata={
            k: v
            for k, v in data.items()
            if k not in {"name", "version", "system", "user", "description", "variables"}
        },
    )


@lru_cache(maxsize=32)
def load_prompt(name: str, version: str, prompts_dir: str) -> Prompt:
    path = Path(prompts_dir) / name / f"{version}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (Path(prompts_dir) / name).glob("*.yaml"))
        raise PromptError(
            f"no prompt {name}/{version} at {path}; available versions: {available}"
        )
    prompt = _load_file(path)
    if prompt.version != version:
        raise PromptError(
            f"{path} declares version {prompt.version!r} but is filed as {version!r}"
        )
    return prompt


def list_prompts(prompts_dir: str | Path) -> dict[str, list[str]]:
    root = Path(prompts_dir)
    if not root.exists():
        return {}
    return {
        d.name: sorted(p.stem for p in d.glob("*.yaml"))
        for d in sorted(root.iterdir())
        if d.is_dir()
    }
