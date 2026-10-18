#!/usr/bin/env python3
"""Create or update the GUI translation file for a language.

    python3 tools/extract_gui_strings.py ro   # -> emips/gui/translations/emips_ro.ts

Collects every ``tr("...")`` / ``N_("...")`` literal in ``emips/gui/*.py`` and writes a Qt
Linguist ``.ts`` file (context ``eMIPS``).  Existing translations are kept;
strings no longer used are marked ``vanished``; new ones are ``unfinished``.
Edit the result with Qt Linguist or a text editor — the GUI reads ``.ts``
directly (see ``emips/gui/i18n.py``).
"""

from __future__ import annotations

import ast
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUI = ROOT / "emips" / "gui"


def collect() -> dict[str, list[tuple[str, int]]]:
    """``{string: [(file, line), ...]}`` in source order."""
    found: dict[str, list[tuple[str, int]]] = {}
    for path in sorted(GUI.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("tr", "N_") and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                found.setdefault(node.args[0].value, []).append(
                    (path.relative_to(ROOT).as_posix(), node.lineno))
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    code = argv[0]
    out = GUI / "translations" / f"emips_{code}.ts"
    old: dict[str, ET.Element] = {}
    if out.is_file():
        for msg in ET.parse(out).getroot().iter("message"):
            old[msg.findtext("source") or ""] = msg.find("translation")
    strings = collect()
    ts = ET.Element("TS", version="2.1", language=code)
    ctx = ET.SubElement(ts, "context")
    ET.SubElement(ctx, "name").text = "eMIPS"
    done = 0
    for src, locs in sorted(strings.items(), key=lambda kv: kv[1][0]):
        msg = ET.SubElement(ctx, "message")
        for f, line in locs:
            ET.SubElement(msg, "location", filename=f"../../../{f}", line=str(line))
        ET.SubElement(msg, "source").text = src
        prev = old.get(src)
        t = ET.SubElement(msg, "translation")
        if prev is not None and (prev.text or "").strip():
            t.text = prev.text
            done += 1
            if prev.get("type") == "unfinished":
                t.set("type", "unfinished")
        else:
            t.set("type", "unfinished")
    for src, prev in old.items():
        if src not in strings and prev is not None and (prev.text or "").strip():
            msg = ET.SubElement(ctx, "message")
            ET.SubElement(msg, "source").text = src
            t = ET.SubElement(msg, "translation", type="vanished")
            t.text = prev.text
    ET.indent(ts)
    out.parent.mkdir(exist_ok=True)
    out.write_text('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE TS>\n'
                   + ET.tostring(ts, encoding="unicode") + "\n", encoding="utf-8")
    print(f"{out.relative_to(ROOT)}: {len(strings)} strings, {done} translated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
