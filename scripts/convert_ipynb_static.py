#!/usr/bin/env python3
"""Convert Jupyter notebooks to scriptless static HTML using only stdlib.

The output is for static study reading. It preserves markdown/code/outputs where
safe, but never emits scripts, inline event handlers, active embedded content, or
javascript/data HTML links outside image data URLs.
"""
from __future__ import annotations

import html
import io
import json
import keyword
import re
import sys
import tokenize
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

SCRIPT_RE = re.compile(r"<\s*script\b[^>]*>.*?<\s*/\s*script\s*>", re.I | re.S)
EVENT_ATTR_RE = re.compile(r"\s+on[a-zA-Z]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)", re.I)

SAFE_HTML_TAGS = {
    "a", "abbr", "b", "blockquote", "br", "caption", "code", "col", "colgroup",
    "dd", "del", "details", "div", "dl", "dt", "em", "figcaption", "figure", "h1",
    "h2", "h3", "h4", "h5", "h6", "hr", "i", "img", "ins", "kbd", "li", "ol",
    "p", "pre", "s", "small", "span", "strong", "sub", "summary", "sup", "table",
    "tbody", "td", "tfoot", "th", "thead", "tr", "u", "ul",
}
VOID_TAGS = {"br", "hr", "img", "col"}
SAFE_ATTRS = {
    "a": {"href", "title"},
    "img": {"src", "alt", "title", "width", "height"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan"},
    "col": {"span"},
    "*": {"class"},
}
DROP_CONTENT_TAGS = {"script", "style", "iframe", "object", "embed", "form", "input", "button", "textarea", "select", "option", "meta", "link", "base"}


def join_source(value):
    if isinstance(value, list):
        return "".join(value)
    return value or ""


def safe_url(url: str, *, image: bool = False) -> bool:
    parsed = urlparse((url or "").strip())
    if not parsed.scheme:
        return True
    if parsed.scheme in {"http", "https"}:
        return True
    if image and parsed.scheme == "data" and url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/gif;base64,")):
        return True
    return False


class SafeHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.drop_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in DROP_CONTENT_TAGS:
            self.drop_depth += 1
            return
        if self.drop_depth:
            return
        if tag not in SAFE_HTML_TAGS:
            self.out.append(html.escape(self.get_starttag_text() or ""))
            return
        allowed = SAFE_ATTRS.get(tag, set()) | SAFE_ATTRS.get("*", set())
        clean_attrs = []
        for name, value in attrs:
            name = name.lower()
            value = value or ""
            if name.startswith("on") or name not in allowed:
                continue
            if name == "href" and not safe_url(value):
                continue
            if name == "src" and not safe_url(value, image=(tag == "img")):
                continue
            clean_attrs.append(f'{name}="{html.escape(value, quote=True)}"')
        suffix = (" " + " ".join(clean_attrs)) if clean_attrs else ""
        self.out.append(f"<{tag}{suffix}>")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in DROP_CONTENT_TAGS:
            if self.drop_depth:
                self.drop_depth -= 1
            return
        if self.drop_depth or tag not in SAFE_HTML_TAGS or tag in VOID_TAGS:
            return
        self.out.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.drop_depth:
            self.out.append(html.escape(data))

    def handle_entityref(self, name):
        if not self.drop_depth:
            self.out.append(f"&{name};")

    def handle_charref(self, name):
        if not self.drop_depth:
            self.out.append(f"&#{name};")


def sanitize_html_fragment(value: str) -> str:
    value = SCRIPT_RE.sub("", value)
    value = EVENT_ATTR_RE.sub("", value)
    parser = SafeHTMLParser()
    parser.feed(value)
    parser.close()
    safe = "".join(parser.out)
    if re.search(r"<\s*(script|iframe|object|embed|form|meta|link|base)\b", safe, re.I):
        raise RuntimeError("unsafe HTML tag survived sanitization")
    if re.search(r"\son[a-zA-Z]+\s*=", safe, re.I) or re.search(r"javascript\s*:", safe, re.I):
        raise RuntimeError("unsafe HTML attribute/protocol survived sanitization")
    return safe


def placeholder_tokens(text: str):
    tokens: list[str] = []

    def add(value: str) -> str:
        token = f"@@NBHTML{len(tokens)}@@"
        tokens.append(value)
        return token

    def markdown_image(match):
        alt = html.escape(match.group(1), quote=True)
        src = match.group(2).strip()
        if not safe_url(src, image=True):
            return html.escape(match.group(0))
        return add(f'<img src="{html.escape(src, quote=True)}" alt="{alt}">')

    def raw_img(match):
        fragment = sanitize_html_fragment(match.group(0))
        if "<img" not in fragment:
            return html.escape(match.group(0))
        return add(fragment)

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", markdown_image, text)
    text = re.sub(r"<\s*img\b[^>]*>", raw_img, text, flags=re.I)
    return text, tokens


def restore_tokens(text: str, tokens: list[str]) -> str:
    for idx, value in enumerate(tokens):
        text = text.replace(f"@@NBHTML{idx}@@", value)
    return text


def inline_markup(text: str) -> str:
    protected, tokens = placeholder_tokens(text)
    escaped = html.escape(protected)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)

    def markdown_link(match):
        label = match.group(1)
        href = html.unescape(match.group(2)).strip()
        if not safe_url(href):
            return label
        return f'<a href="{html.escape(href, quote=True)}" rel="noreferrer">{label}</a>'

    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", markdown_link, escaped)
    return restore_tokens(escaped, tokens)


def render_table(lines):
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    if len(rows) < 2:
        return "\n".join(f"<p>{inline_markup(line)}</p>" for line in lines)
    header = rows[0]
    body = rows[2:] if re.match(r"^\s*\|?\s*:?-{3,}:?", lines[1]) else rows[1:]
    out = ["<table>", "<thead><tr>"]
    out.extend(f"<th>{inline_markup(c)}</th>" for c in header)
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>")
        out.extend(f"<td>{inline_markup(c)}</td>" for c in row)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def render_markdown(src: str) -> str:
    lines = src.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    i = 0

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{inline_markup(' '.join(x.strip() for x in paragraph))}</p>")
            paragraph = []

    def flush_list():
        nonlocal list_items
        if list_items:
            out.append("<ul>" + "".join(f"<li>{inline_markup(x)}</li>" for x in list_items) + "</ul>")
            list_items = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_paragraph(); flush_list(); i += 1; continue
        if stripped.startswith("```"):
            flush_paragraph(); flush_list()
            fence = stripped[3:].strip()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            if i < len(lines):
                i += 1
            out.append(f'<pre><code data-lang="{html.escape(fence, quote=True)}">{html.escape(chr(10).join(code))}</code></pre>')
            continue
        if stripped.startswith("|") and "|" in stripped[1:]:
            flush_paragraph(); flush_list()
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i]); i += 1
            out.append(render_table(table_lines))
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_paragraph(); flush_list()
            level = len(m.group(1))
            out.append(f"<h{level}>{inline_markup(m.group(2))}</h{level}>")
            i += 1; continue
        if re.match(r"^[-*+]\s+", stripped):
            flush_paragraph()
            list_items.append(re.sub(r"^[-*+]\s+", "", stripped))
            i += 1; continue
        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            list_items.append(re.sub(r"^\d+\.\s+", "", stripped))
            i += 1; continue
        paragraph.append(line)
        i += 1
    flush_paragraph(); flush_list()
    return "\n".join(out)


PYTHON_BUILTINS = {
    "abs", "all", "any", "bool", "bytes", "callable", "dict", "dir", "enumerate",
    "float", "getattr", "hasattr", "int", "isinstance", "len", "list", "map", "max",
    "min", "next", "open", "print", "range", "repr", "reversed", "round", "set",
    "sorted", "str", "sum", "super", "tuple", "type", "zip",
}
TOKEN_CLASS = {
    tokenize.COMMENT: "c",
    tokenize.STRING: "s",
    tokenize.NUMBER: "m",
    tokenize.OP: "o",
}


def span(class_name: str, value: str) -> str:
    return f'<span class="tok-{class_name}">{html.escape(value)}</span>'


def highlight_python(source: str) -> str:
    """Return scriptless GitHub-dark-ish highlighted Python HTML."""
    line_offsets = [0]
    for line in source.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))

    def offset(position: tuple[int, int]) -> int:
        line, column = position
        if line <= 0:
            return column
        if line - 1 >= len(line_offsets):
            return len(source)
        return min(line_offsets[line - 1] + column, len(source))

    output: list[str] = []
    cursor = 0
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            token_type = token.type
            token_text = token.string
            if token_type in {tokenize.ENCODING, tokenize.ENDMARKER}:
                continue
            start = offset(token.start)
            end = offset(token.end)
            if start > cursor:
                output.append(html.escape(source[cursor:start]))
            if token_type == tokenize.NAME:
                if keyword.iskeyword(token_text):
                    output.append(span("k", token_text))
                elif token_text in PYTHON_BUILTINS:
                    output.append(span("b", token_text))
                else:
                    output.append(html.escape(token_text))
            elif token_type in TOKEN_CLASS:
                output.append(span(TOKEN_CLASS[token_type], token_text))
            else:
                output.append(html.escape(token_text))
            cursor = max(cursor, end)
        if cursor < len(source):
            output.append(html.escape(source[cursor:]))
        return "".join(output)
    except tokenize.TokenError:
        return html.escape(source)


def render_output(output) -> str:
    otype = output.get("output_type", "")
    if otype == "stream":
        text = join_source(output.get("text", ""))
        return f'<div class="nb-output nb-stream"><pre>{html.escape(text)}</pre></div>'
    if otype == "error":
        tb = "\n".join(output.get("traceback") or [output.get("ename", ""), output.get("evalue", "")])
        return f'<div class="nb-output nb-error"><pre>{html.escape(tb)}</pre></div>'
    data = output.get("data") or {}
    if "image/png" in data:
        img = join_source(data["image/png"]).replace("\n", "")
        return f'<div class="nb-output nb-image"><img alt="notebook output" src="data:image/png;base64,{html.escape(img, quote=True)}"></div>'
    if "image/jpeg" in data:
        img = join_source(data["image/jpeg"]).replace("\n", "")
        return f'<div class="nb-output nb-image"><img alt="notebook output" src="data:image/jpeg;base64,{html.escape(img, quote=True)}"></div>'
    if "text/html" in data:
        safe = sanitize_html_fragment(join_source(data["text/html"]))
        return f'<div class="nb-output nb-html">{safe}</div>'
    if "text/markdown" in data:
        return f'<div class="nb-output nb-markdown">{render_markdown(join_source(data["text/markdown"]))}</div>'
    if "text/plain" in data:
        return f'<div class="nb-output nb-plain"><pre>{html.escape(join_source(data["text/plain"]))}</pre></div>'
    return ""


def convert(ipynb: Path, out: Path, title: str | None = None):
    nb = json.loads(ipynb.read_text(encoding="utf-8"))
    title = title or ipynb.stem
    cells = []
    for idx, cell in enumerate(nb.get("cells", []), start=1):
        ctype = cell.get("cell_type")
        src = join_source(cell.get("source", ""))
        if ctype == "markdown":
            body = render_markdown(src)
        elif ctype == "code":
            exec_count = cell.get("execution_count")
            prompt = f"In [{exec_count if exec_count is not None else ' '}]:"
            outputs = "".join(render_output(o) for o in cell.get("outputs", []))
            body = f'<div class="nb-code-head">{html.escape(prompt)}</div><pre class="nb-code"><code class="language-python">{highlight_python(src)}</code></pre>{outputs}'
        else:
            body = f"<pre>{html.escape(src)}</pre>"
        cells.append(f'<section class="nb-cell nb-{html.escape(str(ctype))}" data-cell="{idx}">{body}</section>')

    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; --bg:#f8fafc; --paper:#ffffff; --ink:#17202a; --muted:#5c6b7a; --line:#d9e2ec; --code:#0f172a; --code-ink:#d9f3ff; --accent:#2563eb; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans KR',sans-serif; line-height:1.68; font-size:16px; }}
    .nb-wrap {{ max-width:1080px; margin:0 auto; padding:28px 24px 72px; }}
    .nb-title {{ margin:0 0 6px; font-size:30px; letter-spacing:-0.03em; }}
    .nb-source {{ color:var(--muted); font-size:13px; margin:0 0 24px; word-break:break-all; }}
    .nb-cell {{ background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:18px 20px; margin:16px 0; box-shadow:0 8px 22px rgba(15,23,42,.05); overflow:auto; }}
    .nb-markdown h1 {{ font-size:28px; }} .nb-markdown h2 {{ font-size:23px; border-bottom:1px solid var(--line); padding-bottom:8px; }} .nb-markdown h3 {{ font-size:19px; }}
    .nb-markdown h1,.nb-markdown h2,.nb-markdown h3,.nb-markdown h4 {{ letter-spacing:-0.025em; line-height:1.25; }}
    .nb-markdown table, .nb-output table {{ border-collapse:collapse; width:max-content; max-width:100%; margin:12px 0; }}
    th,td {{ border:1px solid var(--line); padding:8px 10px; vertical-align:top; }} th {{ background:#eef4ff; }}
    pre {{ white-space:pre-wrap; overflow:auto; }}
    .nb-code-head {{ color:var(--accent); font-weight:800; margin-bottom:8px; font-size:13px; }}
    .nb-code {{ background:var(--code); color:var(--code-ink); border-radius:12px; padding:14px 16px; font-size:13px; line-height:1.58; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace; }}
 .tok-k {{ color:#ff7b72; font-weight:650; }} .tok-b {{ color:#d2a8ff; }} .tok-s {{ color:#a5d6ff; }} .tok-m {{ color:#79c0ff; }} .tok-c {{ color:#8b949e; font-style:italic; }} .tok-o {{ color:#ff7b72; }}
    .nb-output {{ margin-top:12px; border-left:4px solid #93c5fd; background:#f1f7ff; border-radius:10px; padding:10px 12px; }}
    .nb-error {{ border-left-color:#ef4444; background:#fff1f2; }}
    .nb-image img, .nb-markdown img, .nb-html img {{ max-width:100%; height:auto; display:block; }}
    a {{ color:#1d4ed8; }}
  </style>
</head>
<body>
  <main class="nb-wrap">
    <h1 class="nb-title">{html.escape(title)}</h1>
    <p class="nb-source">Source notebook: {html.escape(str(ipynb))}</p>
    {''.join(cells)}
  </main>
</body>
</html>
"""
    document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
    if re.search(r"<\s*(script|iframe|object|embed|form)\b", document, re.I):
        raise RuntimeError(f"unsafe active markup remained in {out}")
    if re.search(r"\son[a-zA-Z]+\s*=", document, re.I) or re.search(r"javascript\s*:", document, re.I):
        raise RuntimeError(f"unsafe script/event/protocol markup remained in {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")


def main(argv):
    if len(argv) < 3:
        print("usage: convert_ipynb_static.py NOTEBOOK.ipynb OUT.html [TITLE]", file=sys.stderr)
        return 2
    convert(Path(argv[1]), Path(argv[2]), argv[3] if len(argv) > 3 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
