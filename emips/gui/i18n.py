"""Interface translations.  English is the default and the source
language; every user-visible string of the GUI goes through :func:`tr`.

A translation is a Qt Linguist ``.ts`` file in ``emips/gui/translations``
(``emips_<code>.ts``, e.g. ``emips_ro.ts``), read directly — no ``lrelease``
step and no binding-specific tools.  Create or update one with::

    python3 tools/extract_gui_strings.py ro

then fill in the translations (Qt Linguist or any text editor).  The
language is chosen in Configure > Language and applies at the next start;
Qt's own texts (dialog buttons) come from Qt's ``qtbase_<code>.qm`` when the
Qt installation has it.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .qt import QtCore

DIR = Path(__file__).with_name("translations")
_table: dict[str, str] = {}
_language = "en"


def tr(text: str) -> str:
    """``text`` in the current language (itself when untranslated)."""
    return _table.get(text) or text


def N_(text: str) -> str:
    """Mark a string for translation where it is defined (a table); translate
    it with :func:`tr` where it is shown."""
    return text


def language() -> str:
    return _language


def available() -> dict[str, str]:
    """``{code: native name}`` of English and every translation found."""
    out = {"en": "English"}
    for f in sorted(DIR.glob("emips_*.ts")):
        code = f.stem.split("_", 1)[1]
        name = QtCore.QLocale(code).nativeLanguageName() or code
        out[code] = name[:1].upper() + name[1:]
    return out


def read_ts(path: Path) -> dict[str, str]:
    """``{source: translation}`` of a ``.ts`` file (finished and unfinished
    non-empty translations)."""
    out = {}
    for msg in ET.parse(path).getroot().iter("message"):
        src, tra = msg.findtext("source"), msg.find("translation")
        if src and tra is not None and (tra.text or "").strip() \
                and tra.get("type") not in ("obsolete", "vanished"):
            out[src] = tra.text
    return out


def install(code: str, app=None) -> bool:
    """Switch to language ``code`` (call before building the windows).
    Returns False when there is no such translation (English is used)."""
    global _table, _language
    _table, _language = {}, "en"
    if code in ("", "en"):
        return True
    path = DIR / f"emips_{code}.ts"
    if not path.is_file():
        return False
    _table, _language = read_ts(path), code
    if app is not None:                       # Qt's own strings (buttons ...)
        qt = QtCore.QTranslator(app)
        where = QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.TranslationsPath)
        if qt.load(f"qtbase_{code}", where):
            app.installTranslator(qt)
            app._emips_qt_translator = qt     # keep it alive
    return True
