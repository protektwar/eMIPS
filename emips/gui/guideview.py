"""Help > User Guide: the HTML guide (``docs/guide/*.html``) in a window of
its own.

The pages are the same files a web browser shows; Qt's rich text engine
reads a subset of HTML / CSS, so each page is adapted before it is shown
(:func:`adapt`): the theme's colours instead of ``guide.css``, the page
list as one line, every heading also a named anchor (``scrollToAnchor``),
each lab answer (``<details>``) folded behind a link, and an *Open in the
editor* link above every complete program.  Links to other pages, to
their sections and back / forward work as in a browser; a link to a lab
program (``labs/*.asm``, ``.c``, ``.elf``) opens it in the editor, a web
link or anything else opens outside (``QDesktopServices``); *Open in a
web browser* shows the page there with the guide's own stylesheet."""

from __future__ import annotations

import html
import re
from pathlib import Path

from . import theme
from .i18n import tr
from .qt import QtCore, QtGui, QtWidgets, Signal

Qt = QtCore.Qt

#: the guide in a source checkout (the repository's docs/guide)
GUIDE_DIR = Path(__file__).resolve().parents[2] / "docs" / "guide"
#: files a link opens in the editor
PROGRAM_SUFFIXES = (".asm", ".s", ".c", ".elf", ".hex")

_DETAILS = re.compile(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", re.S)
_ASM = re.compile(r'<pre><code class="language-asm">(.*?)</code></pre>', re.S)
_NAV = re.compile(r'<header>\s*<nav class="guide-nav".*?</nav>\s*</header>', re.S)


def guide_dir() -> Path | None:
    return GUIDE_DIR if (GUIDE_DIR / "index.html").is_file() else None


def programs(page_html: str) -> list[str]:
    """The complete programs of a page (its ``language-asm`` blocks)."""
    return [html.unescape(m.group(1)) for m in _ASM.finditer(page_html)]


def adapt(page_html: str, open_answers: set[int]) -> str:
    """The page as Qt's rich text shows it (see the module docstring)."""
    text = re.sub(r'<link rel="stylesheet"[^>]*>', "", page_html)
    text = re.sub(r'<a class="skip"[^>]*>.*?</a>', "", text)

    def nav(m: re.Match) -> str:
        items = re.findall(r'<a href="([^"]+)"( aria-current="page")?>(.*?)</a>', m.group(0))
        parts = [f"<b>{label}</b>" if cur else f'<a href="{href}">{label}</a>'
                 for href, cur, label in items]
        return '<p class="nav">' + " &nbsp;·&nbsp; ".join(parts) + "</p><hr>"
    text = _NAV.sub(nav, text)
    text = re.sub(r'<(h[1-6]) id="([^"]+)">', r'<\1 id="\2"><a name="\2"></a>', text)
    count = iter(range(1 << 30))

    def details(m: re.Match) -> str:
        n = next(count)
        summary = re.sub(r"<[^>]+>", "", m.group(1))
        if n in open_answers:
            return (f'<p class="summary"><a href="emips:answer/{n}">▾ {summary}</a></p>'
                    f'<div class="answer">{m.group(2)}</div>')
        return f'<p class="summary"><a href="emips:answer/{n}">▸ {summary}</a></p>'
    text = _DETAILS.sub(details, text)
    code = iter(range(1 << 30))
    text = _ASM.sub(lambda m: (f'<p class="run"><a href="emips:code/{next(code)}">'
                               f'{html.escape(tr("Open in the editor"))}</a></p>'
                               + m.group(0)), text)
    return text


def stylesheet() -> str:
    """CSS for Qt's rich text from the current theme."""
    c = theme.colors()
    pal = QtWidgets.QApplication.palette()
    link = pal.color(QtGui.QPalette.ColorRole.Link).name()
    base = pal.color(QtGui.QPalette.ColorRole.AlternateBase).name()
    return (f"a {{ color: {link}; }}"
            f"pre {{ background-color: {base}; font-family: monospace; white-space: pre-wrap; }}"
            f"code {{ font-family: monospace; background-color: {base}; }}"
            f"th {{ background-color: {base}; }}"
            "table { border-collapse: collapse; }"
            f"th, td {{ border: 1px solid {c.idle_line}; padding: 3px; }}"
            f"p.nav {{ color: {c.muted_text}; }}"
            f"p.summary {{ font-weight: bold; }}"
            f"div.answer {{ margin-left: 16px; }}"
            "p.run { margin-bottom: 0px; }"
            "img { margin: 4px; }")


class GuideView(QtWidgets.QWidget):
    """The guide window: a page, back / forward / contents, find, and
    *Open in a web browser*."""

    #: the text of a guide program to put in the editor
    open_code = Signal(str)
    #: a lab file to open in the editor
    open_path = Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(tr("eMIPS User Guide"))
        self.root = guide_dir()
        self.page: Path | None = None
        self.raw = ""
        self.open_answers: set[int] = set()
        #: (page, scroll position) to go back / forward to
        self.history: list[tuple[Path, int]] = []
        self.ahead: list[tuple[Path, int]] = []
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        style = self.style()
        SP = QtWidgets.QStyle.StandardPixmap
        self.back = QtWidgets.QToolButton()
        self.back.setIcon(style.standardIcon(SP.SP_ArrowBack))
        self.back.setToolTip(tr("Back (Alt+Left)"))
        self.back.setShortcut(QtGui.QKeySequence("Alt+Left"))
        self.back.clicked.connect(self.go_back)
        self.forward = QtWidgets.QToolButton()
        self.forward.setIcon(style.standardIcon(SP.SP_ArrowForward))
        self.forward.setToolTip(tr("Forward (Alt+Right)"))
        self.forward.setShortcut(QtGui.QKeySequence("Alt+Right"))
        self.forward.clicked.connect(self.go_forward)
        self.home = QtWidgets.QPushButton(tr("Contents"))
        self.home.clicked.connect(lambda: self.show_page("index.html"))
        self.find = QtWidgets.QLineEdit()
        self.find.setPlaceholderText(tr("Find in this page (Enter: next)"))
        self.find.setClearButtonEnabled(True)
        self.find.returnPressed.connect(self.find_next)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Find, self,
                        activated=lambda: (self.find.setFocus(), self.find.selectAll()))
        self.external = QtWidgets.QPushButton(tr("Open in a web browser"))
        self.external.clicked.connect(self.open_external)
        for w in (self.back, self.forward, self.home):
            bar.addWidget(w)
        bar.addWidget(self.find, 1)
        bar.addWidget(self.external)
        lay.addLayout(bar)
        self.view = QtWidgets.QTextBrowser()
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self.follow)
        lay.addWidget(self.view, 1)
        self.resize(900, 760)
        if self.root is None:
            self.view.setHtml("<h2>" + html.escape(tr("The user guide is not installed"))
                              + "</h2><p>" + html.escape(tr(
                                  "eMIPS looks for docs/guide/index.html next to its "
                                  "package (a source checkout). The guide is also in the "
                                  "eMIPS repository.")) + "</p>")
            for w in (self.back, self.forward, self.home, self.find, self.external):
                w.setEnabled(False)
        else:
            self.show_page("index.html")

    # -- pages ---------------------------------------------------------------

    def show_page(self, name: str, fragment: str = "", *, remember: bool = True) -> bool:
        """Show ``name`` (a file of the guide, relative to the current page)
        at ``fragment``."""
        if self.root is None:
            return False
        base = self.page.parent if self.page is not None else self.root
        path = (base / name).resolve() if name else self.page
        if path is None or not path.is_file():
            return False
        if remember and self.page is not None:
            self._remember()
        if path != self.page:
            self.page = path
            self.raw = path.read_text(encoding="utf-8")
            self.open_answers = set()
            self._render()
        if fragment:
            self.view.scrollToAnchor(fragment)
        else:
            self.view.verticalScrollBar().setValue(0)
        self._sync()
        return True

    def _render(self, keep_scroll: bool = False) -> None:
        bar = self.view.verticalScrollBar()
        pos = bar.value()
        doc = self.view.document()
        doc.setDefaultStyleSheet(stylesheet())
        doc.setBaseUrl(QtCore.QUrl.fromLocalFile(str(self.page)))
        self.view.setSearchPaths([str(self.page.parent)])
        self.view.setHtml(adapt(self.raw, self.open_answers))
        title = re.search(r"<title>(.*?)</title>", self.raw, re.S)
        if title:
            self.setWindowTitle(html.unescape(title.group(1)))
        if keep_scroll:
            bar.setValue(pos)

    def restyle(self) -> None:
        """The theme changed: draw the page again."""
        if self.page is not None:
            self._render(keep_scroll=True)

    def _sync(self) -> None:
        self.back.setEnabled(bool(self.history))
        self.forward.setEnabled(bool(self.ahead))

    def _here(self) -> tuple[Path, int]:
        return self.page, self.view.verticalScrollBar().value()

    def _remember(self) -> None:
        self.history.append(self._here())
        self.ahead.clear()

    def _go(self, page: Path, pos: int) -> None:
        self.show_page(str(page), remember=False)
        self.view.verticalScrollBar().setValue(pos)

    def go_back(self) -> None:
        if self.history:
            self.ahead.append(self._here())
            self._go(*self.history.pop())

    def go_forward(self) -> None:
        if self.ahead:
            self.history.append(self._here())
            self._go(*self.ahead.pop())

    # -- links ---------------------------------------------------------------

    def follow(self, url: QtCore.QUrl) -> None:
        text = url.toString()
        if url.scheme() == "emips":
            kind, _, n = url.path().partition("/")
            if kind == "answer":
                self.open_answers ^= {int(n)}
                self._render(keep_scroll=True)
            elif kind == "code":
                self.open_code.emit(programs(self.raw)[int(n)])
            return
        if url.scheme() in ("http", "https", "mailto"):
            QtGui.QDesktopServices.openUrl(url)
            return
        path, _, fragment = text.partition("#")
        if url.isLocalFile():
            path = url.toLocalFile()
        if not path:                            # a section of this page
            self._remember()
            self.view.scrollToAnchor(fragment)
            self._sync()
            return
        target = (self.page.parent / path).resolve()
        if target.suffix == ".html":
            self.show_page(str(target), fragment)
        elif target.suffix.lower() in PROGRAM_SUFFIXES and target.is_file():
            self.open_path.emit(str(target))
        else:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(target)))

    def find_next(self) -> None:
        text = self.find.text()
        if not text:
            return
        if not self.view.find(text):            # wrap around
            self.view.moveCursor(QtGui.QTextCursor.MoveOperation.Start)
            self.view.find(text)

    def open_external(self) -> None:
        if self.page is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.page)))
