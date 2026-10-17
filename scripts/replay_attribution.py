#!/usr/bin/env python
"""Offline replay of the D15 attribution check over saved eval results.

Free: re-runs local retrieval (no LLM call), maps each answered sample's [S#]
markers to the rendered context order, and asks which answered samples the
check would now refuse. The result JSON stores no chunk ids, so the cited
documents are a RECONSTRUCTION (same config => same retrieval and context
packing); markers that do not resolve are reported.

    PYTHONPATH=src .venv/bin/python scripts/replay_attribution.py eval_results/X.json ...
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpipe.config import load_settings  # noqa: E402
from ragpipe.eval.golden import load_verified  # noqa: E402
from ragpipe.generation.attribution import DocumentIndex, check_attribution  # noqa: E402
from ragpipe.generation.context import render_context  # noqa: E402
from ragpipe.index.builder import get_store  # noqa: E402
from ragpipe.retrieval import get_retriever  # noqa: E402

MARK = re.compile(r"\[S(\d{1,3})\]")


def main(paths: list[str]) -> int:
    settings = load_settings()
    store = get_store(settings)
    index = DocumentIndex.from_chunks(store.iter_chunks())
    retriever = get_retriever(settings, store)
    pairs = {p.id: p for p in load_verified(settings.evaluation.dataset_path)}
    print(f"verified pairs: {len(pairs)}")
    triggers = [q.id for q in pairs.values() if len(index.named_documents(q.question)) == 1]
    multi = [q.id for q in pairs.values() if len(index.named_documents(q.question)) > 1]
    print(f"title detection: exactly-one={len(triggers)} multi={len(multi)}")

    cache: dict[str, object] = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        print(f"\n== {path} (prompt {data['prompt_version']})")
        n_ans = n_ref = unresolved = 0
        for s in data["samples"]:
            if s["status"] != "answered" or s["id"] not in pairs:
                continue
            n_ans += 1
            q = pairs[s["id"]].question
            if len(index.named_documents(q)) != 1:
                continue
            if s["id"] not in cache:
                hits = retriever.retrieve(q)[: settings.retrieval.top_k]
                cache[s["id"]] = render_context(
                    hits, settings.generation.max_context_tokens,
                    include_locators=settings.generation.include_locators,
                )
            used = cache[s["id"]].marker_to_chunk
            ms = sorted({int(m) for m in MARK.findall(s["answer"])})
            cited = [used[m] for m in ms if m in used]
            unresolved += sum(1 for m in ms if m not in used)
            reason = check_attribution(q, cited, index)
            if reason:
                n_ref += 1
                print(f"WOULD REFUSE {s['id']} unanswerable={s['unanswerable']}\n"
                      f"  Q: {q}\n  {reason}\n  cited={[c.chunk_id for c in cited]}")
        print(f"answered={n_ans} would_refuse={n_ref} unresolved_markers={unresolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
