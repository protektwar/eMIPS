"""The guide's labs as a catalogue (File > Labs; Qt-free).

Read from the guide itself (``docs/guide/05-understanding.html``): each
``Lab N — title`` section and the lab programs (``labs/...``) it links;
the programs the guide links elsewhere come as "more programs", by the
section that uses them.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path

GUIDE_DIR = Path(__file__).resolve().parents[1] / "docs" / "guide"
PROGRAM_SUFFIXES = (".asm", ".s", ".c")


@dataclass
class Lab:
    title: str                                   # "Lab 30 — A real board"
    files: list[Path] = field(default_factory=list)


def _text(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _files(guide: Path, body: str, seen) -> list[Path]:
    """The lab programs a part of the guide names (links or code text)."""
    out: list[Path] = []
    for name in re.findall(r"labs/([\w.-]+\.(?:asm|c))\b", body):
        p = (guide / "labs" / name).resolve()
        if p.is_file() and p not in out and p not in seen:
            out.append(p)
    return out


def catalogue(guide: Path = GUIDE_DIR) -> tuple[list[Lab], list[Lab]]:
    """(the labs, more programs by section); empty without the guide."""
    page = guide / "05-understanding.html"
    if not page.is_file():
        return [], []
    text = page.read_text(encoding="utf-8")
    heads = list(re.finditer(r"<h([23]) id=\"([^\"]+)\">(.*?)</h\1>", text, re.S))
    labs: list[Lab] = []
    more: list[Lab] = []
    seen: set[Path] = set()
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[h.end():end]
        files = _files(guide, body, [])
        if not files:
            continue
        title = _text(h.group(3))
        if h.group(2).startswith("lab-"):
            labs.append(Lab(title, files))
            seen.update(files)
    for other in sorted(guide.glob("0*.html")):  # programs used outside the labs
        text = other.read_text(encoding="utf-8")
        heads = list(re.finditer(r"<h([23]) id=\"([^\"]+)\">(.*?)</h\1>", text, re.S))
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            if h.group(2).startswith("lab-"):
                continue
            files = _files(guide, text[h.end():end], seen)
            if files:
                more.append(Lab(_text(h.group(3)), files))
                seen.update(files)
    rest = [p.resolve() for p in sorted((guide / "labs").glob("*"))
            if p.suffix in PROGRAM_SUFFIXES and p.resolve() not in seen]
    if rest:
        more.append(Lab("Other programs", rest))
    return labs, more
