"""Build the HTML docs from docs/*.md.

- docs/html/index.html: every doc on one page.
- docs/html/<name>.html: one page per doc (README.md becomes overview.html; feature-sheet.md, which has no number,
  becomes feature-sheet.html and comes last).
- A doc with a hand-made page (00 → how-it-all-works.html) gets no generated page; links go to the
  hand-made one, whose section ids match the Markdown headings.

Links between docs, including #anchors, work on both. Heading ids follow GitHub's rules, so the
anchors written for GitHub work here too.

Usage: python scripts/build_docs_html.py [docs_dir] [out_dir]
Requires: pip install markdown   (or: uv run --with markdown python scripts/build_docs_html.py)
"""
import glob
import html
import os
import re
import sys

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DOCS, "html")
REPO = "https://github.com/Merit-Order/meritorder/blob/main/"
HANDMADE = {"00-how-it-all-works.md": "how-it-all-works.html"}  # written by hand, kept in sync with the doc

files = sorted(glob.glob(os.path.join(DOCS, "*.md")))
ids = {}
for f in files:
    b = os.path.basename(f)
    ids[b] = "overview" if b == "README.md" else re.sub(r"^\d+-", "", b[:-3])


def page_of(b: str) -> str:
    return HANDMADE.get(b) or f"{ids[b]}.html"


SHADOW_SVG = '''<figure class="chart"><svg viewBox="0 0 640 320" role="img" aria-labelledby="sp-t">
<title id="sp-t">Shadow price vs replica load</title>
<rect x="70" y="30" width="270" height="230" class="z1"/>
<rect x="340" y="30" width="180" height="230" class="z2"/>
<rect x="520" y="30" width="90" height="230" class="z3"/>
<text x="205" y="52" text-anchor="middle" class="t i1">Headroom: 2% of the API price</text>
<text x="430" y="52" text-anchor="middle" class="t i2">Filling up</text>
<text x="565" y="52" text-anchor="middle" class="t i3">Full</text>
<line x1="70" y1="85" x2="520" y2="85" class="ref"/>
<text x="76" y="78" class="lbl">API price of the same model</text>
<line x1="70" y1="260" x2="612" y2="260" class="axis"/>
<line x1="70" y1="260" x2="70" y2="26" class="axis"/>
<text x="340" y="303" text-anchor="middle" class="lbl">Replica load: vLLM's KV-cache use (85%+ when requests queue), or in-flight / max_concurrency</text>
<text x="30" y="145" text-anchor="middle" class="lbl" transform="rotate(-90 30 145)">Shadow price per token</text>
<text x="70" y="279" text-anchor="middle" class="tick">0%</text>
<text x="340" y="279" text-anchor="middle" class="tick">60% (knee)</text>
<text x="430" y="279" text-anchor="middle" class="tick">80%</text>
<text x="520" y="279" text-anchor="middle" class="tick">100%</text>
<path d="M70 256.5 L340 256.5 L349 256.1 L358 254.8 L367 252.6 L376 249.6 L385 245.8 L394 241.1 L403 235.5 L412 229.1 L421 221.8 L430 213.6 L439 204.6 L448 194.8 L457 184.0 L466 172.5 L475 160.0 L484 146.7 L493 132.6 L502 117.6 L511 101.7 L520 85.0" class="curve"/>
<circle cx="430" cy="213.6" r="4" class="dot"/>
<text x="424" y="207" text-anchor="end" class="tick">27% at 80%</text>
<circle cx="475" cy="160" r="4" class="dot"/>
<text x="469" y="154" text-anchor="end" class="tick">57% at 90%</text>
<circle cx="520" cy="85" r="5" class="dot"/>
<text x="565" y="110" text-anchor="middle" class="tick i3">nothing new</text>
<text x="565" y="126" text-anchor="middle" class="tick i3">while another</text>
<text x="565" y="142" text-anchor="middle" class="tick i3">route has room</text>
</svg><figcaption>A running GPU's spare capacity is priced at 2% of the API price up to 60% load (routing.shadow_knee), then rises with the square of the load to the full API price at 100%. A full replica takes nothing new while another route has room.</figcaption></figure>'''


def gh_slug(value: str, separator: str = "-") -> str:
    """GitHub's heading ids: lowercase, punctuation dropped (letters, digits, _ and - kept), spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", value.strip().lower()).replace(" ", separator)


def render(path: str, prefix: str = "") -> tuple[str, str, list[tuple[str, str]]]:
    """(title, body html, [(h2 id, h2 text)]) for one doc. ``prefix`` namespaces heading ids (combined page)."""
    src = open(path, encoding="utf-8").read()
    blocks: list[str] = []

    def grab(m):
        blocks.append(m.group(1))
        return f"\n\nMERMAIDBLOCK{len(blocks) - 1}\n\n"

    src = re.sub(r"```mermaid\n(.*?)```", grab, src, flags=re.S)
    src = re.sub(r"!\[[^\]]*\]\(img/shadow-price\.svg\)", "\n\nSHADOWSVG\n\n", src)
    title_m = re.match(r"#\s+(.+)\n", src)
    title = title_m.group(1).strip()
    src = src[title_m.end():]
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"],
                           extension_configs={"toc": {"slugify": lambda v, sep: prefix + gh_slug(v, sep)}})
    body = md.convert(src)
    for i, blk in enumerate(blocks):
        body = body.replace(f"<p>MERMAIDBLOCK{i}</p>",
                            f'<figure class="diagram"><pre class="mermaid">{html.escape(blk.strip())}</pre></figure>')
    body = body.replace("<p>SHADOWSVG</p>", SHADOW_SVG)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    h2s = [(t["id"], t["name"]) for t in md.toc_tokens if t["level"] == 2]
    return title, body, h2s


def relink(body: str, combined: bool, sid: str = "") -> str:
    """Point links between docs at the right page (or section of the combined page, where heading ids
    carry the doc's id as a prefix)."""

    def fix(m):
        href = m.group(1)
        if href.startswith("#") and combined and sid:
            return f'href="#{sid}-{html.escape(href[1:], quote=True)}"'
        if re.match(r"^[a-z]+:|^#", href):
            return m.group(0)
        doc = re.match(r"^((?:\d\d-[\w-]+|README|feature-sheet)\.md)(#.*)?$", href)
        if doc and doc.group(1) in ids:
            b, frag = doc.group(1), (doc.group(2) or "")[1:]
            if b in HANDMADE:
                target = HANDMADE[b] + (f"#{frag}" if frag else "")
            elif combined:
                target = f"#{ids[b]}-{frag}" if frag else f"#{ids[b]}"
            else:
                target = page_of(b) + (f"#{frag}" if frag else "")
        elif href.startswith("html/"):
            target = href[len("html/"):] or "index.html"
        elif href.startswith("img/"):
            target = "../" + href
        elif href.startswith("../"):
            target = REPO + href[3:]
        else:
            target = href
        return f'href="{html.escape(target, quote=True)}"'

    return re.sub(r'href="([^"]*)"', fix, body)


def shift_headings(body: str) -> str:
    """h2 → h3 and h3 → h4, keeping attributes (ids)."""
    body = re.sub(r"<(/?)h3(\s[^>]*)?>", lambda m: f"<{m.group(1)}h4{m.group(2) or ''}>", body)
    return re.sub(r"<(/?)h2(\s[^>]*)?>", lambda m: f"<{m.group(1)}h3{m.group(2) or ''}>", body)


def split_title(title: str, b: str) -> tuple[str, str]:
    num, _, name = title.partition(" — ")
    return (num, name) if name else ("", title)


tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs_template.html"), encoding="utf-8").read()


def fill(**parts) -> str:
    out = tpl
    for k, v in parts.items():
        out = out.replace("{{" + k + "}}", v)
    return out


os.makedirs(OUT, exist_ok=True)

# ---- every doc on one page
sections, nav, hero_body = [], [], ""
for f in files:
    b = os.path.basename(f)
    sid = ids[b]
    title, body, _ = render(f, prefix=f"{sid}-")
    body = relink(body, combined=True, sid=sid)
    if sid == "overview":
        hero_body = shift_headings(body)
        continue
    num, name = split_title(title, b)
    nav.append(f'<li><a href="#{sid}"><span class="n">{num}</span>{html.escape(name)}</a></li>')
    sections.append(f'<section id="{sid}"><header class="sec-head"><span class="n">{num}</span><h2>{html.escape(name)}</h2></header>'
                    f"{shift_headings(body)}</section>")
split = hero_body.split("<h3", 1)  # the overview's text before its first heading is the hero lede
lede, rest = split[0], ("<h3" + split[1] if len(split) > 1 else "")
pages_link = '<p class="also">Also one page per doc: start with <a href="overview.html">the overview</a> or <a href="how-it-all-works.html">how it all works</a>.</p>'
index = fill(TITLE="Merit Order Design Notes", EYEBROW="Merit Order · design notes", H1="A gateway that spends less on every token",
             LEDE=lede + pages_link, RAIL="Chapters",
             NAV='<li><a href="#overview"><span class="n">00</span>Overview</a></li>' + "".join(nav),
             OVERVIEW=f'<section id="overview"><header class="sec-head"><span class="n">00</span><h2>Overview</h2></header>{rest}</section>',
             SECTIONS="".join(sections))
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(index)
print("wrote", os.path.join(OUT, "index.html"), len(index), "bytes")

# ---- one page per doc
entries = []
for f in files:
    b = os.path.basename(f)
    title = open(f, encoding="utf-8").readline().lstrip("# ").strip()
    num, name = split_title(title, b)
    entries.append((b, "&nbsp;&nbsp;" if b == "README.md" else num, "Overview" if b == "README.md" else name))
entries.sort(key=lambda e: (e[0] != "README.md", e[0]))

for f in files:
    b = os.path.basename(f)
    if b in HANDMADE:
        continue
    title, body, h2s = render(f)
    body = relink(body, combined=False)
    num, name = split_title(title, b)
    if b == "README.md":
        num, name = "", "Overview"
    current = ' class="active" aria-current="page"'
    nav_items = "".join(
        f'<li><a href="{page_of(e[0])}"{current if e[0] == b else ""}>'
        f'<span class="n">{e[1]}</span>{html.escape(e[2])}</a></li>' for e in entries)
    contents = ""
    if len(h2s) >= 3:
        contents = ('<details class="contents"><summary>On this page</summary><ol>'
                    + "".join(f'<li><a href="#{i}">{html.escape(t)}</a></li>' for i, t in h2s) + "</ol></details>")
    lede = ('<p class="also"><a href="index.html">All docs on one page</a> · '
            '<a href="how-it-all-works.html">How it all works</a></p>' + contents)
    page = fill(TITLE=f"{html.escape(name)} · Merit Order docs", EYEBROW=f"Merit Order docs{' · ' + num if num else ''}",
                H1=html.escape(name), LEDE=lede, RAIL="Docs", NAV=nav_items, OVERVIEW="",
                SECTIONS=f'<section id="doc">{body}</section>')
    out = os.path.join(OUT, page_of(b))
    open(out, "w", encoding="utf-8").write(page)
    print("wrote", out, len(page), "bytes")
