"""Build the GitHub Pages site: the user guide at the root, eMIPS in the
browser at ``run/``.

    python3 web/build_site.py OUT

``run/`` holds the page (``web/index.html``, ``app.js``, ``app.css``,
``bridge.py``), ``emips.zip`` (eMIPS without its Qt window, for Pyodide),
the window's examples and ``programs.json``: the examples, the guide's labs
and its other programs (assembly only: no C compiler in a browser), each
with the files its ``# eMIPS:`` line names (a lab's disk image).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "guide"
PAGE = ("index.html", "app.js", "app.css", "bridge.py")
#: what the browser runs: eMIPS without its window (the window needs Qt)
SKIP = ("emips/gui/", "emips/image.py")


def companions(path: Path) -> list[str]:
    """The files a program's ``# eMIPS:`` line names (``--disk`` images),
    relative to the guide."""
    text = path.read_text(encoding="utf-8", errors="replace")
    names = []
    for m in re.finditer(r"#\s*eMIPS:([^\n]*)", text):
        for d in re.findall(r"--disk\s+(?:\w+=)?(\S+)", m.group(1)):
            for name in d.split(","):
                f = path.parent / name.split("=")[-1]
                if f.is_file():
                    names.append(f.relative_to(GUIDE).as_posix())
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
        shutil.copy2(ROOT / "emips" / "gui" / "examples" / ex.file, examples / ex.file)
        items.append({"title": ex.title, "path": f"run/examples/{ex.file}",
                      "keys": ex.keys, "files": []})
    groups.append({"group": "Examples", "items": items})
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


def main(out: str) -> None:
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
    (run / "programs.json").write_text(json.dumps(programs(run), indent=1), encoding="utf-8")
    print(f"{out_dir}: the guide, run/ ({(run / 'emips.zip').stat().st_size / 1e6:.1f} MB "
          "emips.zip)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_site")
