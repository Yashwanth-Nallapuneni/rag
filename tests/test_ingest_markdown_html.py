from __future__ import annotations

import pytest

from ragpipe.ingest.html import HTMLParser
from ragpipe.ingest.markdown import MarkdownParser
from ragpipe.schemas import SourceType

MD = """---
title: Retrieval Notes
---

# Retrieval Notes

Intro prose about retrieval.

## Dense Retrieval

Dense text here.

### Bi-encoders

Bi-encoder detail.

```python
# not a heading
def score(q, d):
    return 1
---
```

## Sparse Retrieval

BM25 uses term frequency.

| Method | Recall |
|---|---|
| BM25 | 0.61 |
| Dense | 0.74 |

- first point
- second point

Setext Heading
==============

After setext.
"""

HTML = """<html><head><title>Widgets | ACME Corp</title>
<meta name="description" content="About widgets">
<script>var tracking = 1;</script></head>
<body>
<nav>Home Products Contact</nav>
<div class="sidebar">Subscribe to our newsletter</div>
<main>
  <h1>Widgets Overview</h1>
  <p>Widgets are small components.</p>
  <h2>Design</h2>
  <p>Design considerations matter.</p>
  <h3>Materials</h3>
  <p>Steel and aluminium.</p>
  <pre><code>def widget():
    return True</code></pre>
  <table><tr><th>Name</th><th>Weight</th></tr><tr><td>Small</td><td>1kg</td></tr></table>
  <figure><img src="x.png"><figcaption>Figure 1: A widget.</figcaption></figure>
  <h2>Usage</h2>
  <p>Install it first.</p>
</main>
<footer>Copyright ACME 2026</footer>
</body></html>"""


@pytest.fixture
def md_doc(settings, tmp_path):
    p = tmp_path / "notes.md"
    p.write_text(MD)
    return MarkdownParser(settings.ingest).parse(str(p))


@pytest.fixture
def html_doc(settings, tmp_path):
    p = tmp_path / "widgets.html"
    p.write_text(HTML)
    return HTMLParser(settings.ingest).parse(str(p))


def _by_kind(doc, kind):
    return [b for b in doc.blocks if b.kind == kind]


# --- Markdown -------------------------------------------------------------


def test_front_matter_supplies_title(md_doc):
    assert md_doc.document.title == "Retrieval Notes"
    assert md_doc.document.source_type == SourceType.MARKDOWN


def test_fenced_code_contents_are_not_parsed_as_structure(md_doc):
    """A `#` or `---` inside a fence must not shred the heading hierarchy."""
    headings = [b.text for b in _by_kind(md_doc, "heading")]
    assert "not a heading" not in " ".join(headings)
    assert "Dense Retrieval" in headings and "Sparse Retrieval" in headings


def test_nested_section_path_is_exact(md_doc):
    deep = [b for b in md_doc.blocks if b.text.startswith("Bi-encoder detail")]
    assert deep[0].section_path == ["Retrieval Notes", "Dense Retrieval", "Bi-encoders"]


def test_returning_to_shallower_level_clears_deeper(md_doc):
    bm25 = [b for b in md_doc.blocks if b.text.startswith("BM25 uses")]
    assert bm25[0].section_path == ["Retrieval Notes", "Sparse Retrieval"]


def test_setext_heading_recognised(md_doc):
    assert "Setext Heading" in [b.text for b in _by_kind(md_doc, "heading")]


def test_markdown_has_no_pages_but_keeps_a_locator(md_doc):
    body = md_doc.body_blocks()
    assert all(b.page is None for b in body)
    assert any(b.section_path for b in body), "without pages, sections are the locator"


def test_code_block_keeps_indentation(md_doc):
    code = _by_kind(md_doc, "code")
    assert code and "    return 1" in code[0].text


def test_table_rows_stay_on_separate_lines(md_doc):
    table = _by_kind(md_doc, "table")
    assert table and len(table[0].text.splitlines()) >= 3


# --- HTML -----------------------------------------------------------------


def test_chrome_is_removed(html_doc):
    text = html_doc.text
    for noise in ("tracking", "Subscribe to our newsletter", "Copyright ACME", "Home Products"):
        assert noise not in text, f"page chrome leaked into content: {noise!r}"


def test_h1_preferred_over_title_tag(html_doc):
    assert html_doc.document.title == "Widgets Overview"


def test_html_heading_stack(html_doc):
    materials = [b for b in html_doc.blocks if b.text.startswith("Steel")]
    assert materials[0].section_path == ["Widgets Overview", "Design", "Materials"]
    install = [b for b in html_doc.blocks if b.text.startswith("Install")]
    assert install[0].section_path == ["Widgets Overview", "Usage"]


def test_html_code_keeps_indentation(html_doc):
    code = _by_kind(html_doc, "code")
    assert code and "    return True" in code[0].text


def test_html_table_rows_preserved(html_doc):
    table = _by_kind(html_doc, "table")
    assert table and "Name | Weight" in table[0].text
    assert len(table[0].text.splitlines()) == 2


def test_figcaption_captured(html_doc):
    assert any("Figure 1" in b.text for b in _by_kind(html_doc, "caption"))


def test_metadata_captured(html_doc):
    assert html_doc.document.metadata.get("description") == "About widgets"


def test_supports_routing(settings):
    md = MarkdownParser(settings.ingest)
    html = HTMLParser(settings.ingest)
    assert md.supports("a.md") and not md.supports("a.html")
    assert html.supports("a.html") and html.supports("https://example.com")
    assert not html.supports("a.md")
