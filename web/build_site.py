"""Build the GitHub Pages site: the user guide at the root, eMIPS in the
browser at ``run/``.

    python3 web/build_site.py OUT

``run/`` holds the page (``web/index.html``, ``app.js``, ``app.css``,
``bridge.py``), ``emips.zip`` (eMIPS without its Qt window, for Pyodide),
the window's examples and ``programs.json``: the examples, the guide's labs
and its other programs (assembly only: no C compiler in a browser), each
with the files its ``# eMIPS:`` line names (a lab's disk image, its files).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "guide"
PAGE = ("index.html", "app.js", "worker.js", "app.css", "bridge.py")
#: what the browser runs: eMIPS without its window (the window needs Qt)
SKIP = ("emips/gui/", "emips/image.py")


def companions(path: Path, base: Path = GUIDE) -> list[str]:
    """The files a program's ``# eMIPS:`` line names (``--disk`` images,
    ``--file`` files) and the files it ``.include``s,
    relative to the guide."""
    text = path.read_text(encoding="utf-8", errors="replace")
    names = []
    for m in re.finditer(r"#\s*eMIPS:([^\n]*)", text):
        for d in re.findall(r"--(?:disk|file)\s+(?:[\w.-]+=)?(\S+)", m.group(1)):
            for name in d.split(","):
                f = path.parent / name.split("=")[-1]
                if f.is_file():
                    names.append(f.relative_to(base).as_posix())
    sibling = path.with_suffix(".check")          # (its checks: Check on the page)
    if sibling.is_file():
        names.append(sibling.relative_to(base).as_posix())
    for name in re.findall(r'^\s*(?:[\w.$]+:\s*)*\.include\s+"([^"]+)"', text, re.M):
        f = path.parent / name                    # (its .include files: PLAN M26)
        if f.is_file() and f.relative_to(base).as_posix() not in names:
            names.append(f.relative_to(base).as_posix())
    return names


def programs(run: Path) -> list[dict]:
    sys.path.insert(0, str(ROOT))
    from emips.gui.examples import EXAMPLES     # (Qt-free: the list and its files)
    from emips.labs import catalogue
    groups = []
    examples = run / "examples"
    examples.mkdir()
    items = []
    for ex in EXAMPLES:
        shutil.copy2(ex.path, examples / ex.file)
        files = []
        for name in companions(ex.path, ex.path.parent):    # (its --file files)
            shutil.copy2(ex.path.parent / name, examples / name)
            files.append(f"run/examples/{name}")
        items.append({"title": ex.title, "path": f"run/examples/{ex.file}",
                      "keys": ex.keys, "files": files})
    groups.append({"group": "Examples", "items": items})
    from emips.gui.examples import RECIPES      # (PLAN M28: the video series' programs)
    recipes = run / "recipes"
    recipes.mkdir()
    items = []
    for r in RECIPES:
        for kind, path in r.files():
            if path.suffix != ".asm":
                continue
            shutil.copy2(path, recipes / path.name)
            for name in companions(path, path.parent):      # (its --file files)
                shutil.copy2(path.parent / name, recipes / name)
            if "ibrary" in kind:                # (included by a program: not one itself)
                continue
            items.append({"title": f"{r.number}. {r.title}" + (" (RISC-V)" if kind == "RISC-V"
                                                               else ""),
                          "path": f"run/recipes/{path.name}", "keys": "",
                          "files": [f"run/recipes/{n}" for n in companions(path, path.parent)]})
    groups.append({"group": "MIPS by Example in eMIPS", "items": items})
    labs, more = catalogue(GUIDE)
    for name, entries in (("Labs", labs), ("Programs from the guide", more)):
        items = []
        for lab in entries:
            for f in lab.files:
                if f.suffix != ".asm":
                    continue
                title = lab.title if len(lab.files) == 1 else f"{lab.title}: {f.name}"
                items.append({"title": title, "path": f.relative_to(GUIDE).as_posix(),
                              "files": companions(f)})
        groups.append({"group": name, "items": items})
    return groups


_ASM = re.compile(r'<pre><code class="language-asm">(.*?)</code></pre>', re.S)
_HEADING = re.compile(r'<h([1-4]) id="[^"]+">(.*?)</h\1>', re.S)
_PAGES = ("01-getting-started", "02-writing-programs", "03-the-gui", "04-command-line",
          "05-understanding", "06-reference")


def guide_examples(site: Path) -> list[dict]:
    """The guide's complete programs (its ``language-asm`` blocks: the ones
    the window's *Open in the editor* opens) as ``run/guide/PAGE-N.asm``, and
    an *Open in eMIPS* link above each block on the site's copy of its page.
    The block's markers (``# arch: rv32``, ``# keys: ...``) become the
    program's ``# eMIPS:`` line and the item's keys."""
    sys.path.insert(0, str(ROOT))
    import html
    import shlex
    from emips.profile import guide_flags
    folder = site / "run" / "guide"
    folder.mkdir()
    items = []
    for page in _PAGES:
        path = site / f"{page}.html"
        text = path.read_text(encoding="utf-8")
        n = iter(range(1, 1 << 30))
        out, last = [], 0
        for m in _ASM.finditer(text):
            k = next(n)
            program = html.unescape(m.group(1))
            name = f"{page}-{k}.asm"
            flags = guide_flags(program)
            if flags:
                program = program.rstrip("\n") + "\n# eMIPS: " + shlex.join(flags) + "\n"
            (folder / name).write_text(program, encoding="utf-8")
            heads = _HEADING.findall(text, 0, m.start())
            title = re.sub(r"<[^>]+>", "", html.unescape(heads[-1][1])).strip() if heads else page
            keys = re.search(r"^#\s*keys:\s?(.*)$", program, re.M)
            given = re.search(r"^#\s*input:\s?(.*)$", program, re.M)
            items.append({"title": title, "path": f"run/guide/{name}", "files": [],
                          "keys": keys.group(1) if keys else "",
                          "input": given.group(1) if given else ""})
            out.append(text[last:m.start()])
            out.append(f'<p class="open-in-emips"><a href="run/?open=run/guide/{name}">'
                       "Open in eMIPS</a> <span>runs this program in your browser</span></p>\n")
            last = m.start()
        out.append(text[last:])
        path.write_text("".join(out), encoding="utf-8")
    titles: dict[str, int] = {}                   # (the same heading twice: numbered)
    for it in items:
        titles[it["title"]] = titles.get(it["title"], 0) + 1
    seen: dict[str, int] = {}
    for it in items:
        if titles[it["title"]] > 1:
            seen[it["title"]] = seen.get(it["title"], 0) + 1
            it["title"] += f" ({seen[it['title']]})"
    return items


#: the visit counter (GoatCounter: no cookies, no personal data kept): the
#: site's code comes from the build (EMIPS_GOATCOUNTER: the Pages workflow
#: sets "emips"); a local build counts nothing
COUNTER_SCRIPT = ('<script data-goatcounter="https://{code}.goatcounter.com/count"\n'
                  '        async src="//gc.zgo.at/count.js"></script>')
COUNTER_NOTE = ('<p class="counter-note">This site counts visits with '
                '<a href="https://www.goatcounter.com">GoatCounter</a>: no cookies, no '
                'personal data.</p>')


def add_counter(site: Path, code: str) -> int:
    """The counter's script in every page of the site (and a line saying so
    at the end of each); the number of pages."""
    if not re.fullmatch(r"[a-z0-9-]{2,50}", code):
        raise SystemExit(f"bad GoatCounter code {code!r} (letters, digits and -)")
    n = 0
    for page in sorted(site.rglob("*.html")):
        text = page.read_text(encoding="utf-8")
        if "</head>" not in text or "goatcounter" in text:
            continue
        text = text.replace("</head>", COUNTER_SCRIPT.format(code=code) + "\n</head>", 1)
        text = text.replace("</body>", COUNTER_NOTE + "\n</body>", 1)
        page.write_text(text, encoding="utf-8")
        n += 1
    return n


def main(out: str, counter: str | None = None) -> None:
    out_dir = Path(out)
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(GUIDE, out_dir)
    run = out_dir / "run"
    run.mkdir()
    for name in PAGE:
        shutil.copy2(ROOT / "web" / name, run / name)
    with zipfile.ZipFile(run / "emips.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "emips").rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_dir() or "__pycache__" in rel or rel.startswith(SKIP):
                continue
            z.write(p, rel)
    groups = programs(run)
    groups.append({"group": "Examples in the guide", "items": guide_examples(out_dir)})
    (run / "programs.json").write_text(json.dumps(groups, indent=1), encoding="utf-8")
    print(f"{out_dir}: the guide, run/ ({(run / 'emips.zip').stat().st_size / 1e6:.1f} MB "
          "emips.zip)")
    counter = counter if counter is not None else os.environ.get("EMIPS_GOATCOUNTER", "")
    if counter.strip():
        print(f"visit counter: {counter.strip()}.goatcounter.com in "
              f"{add_counter(out_dir, counter.strip())} pages")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_site")
