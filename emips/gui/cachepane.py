"""The cache pane (PLAN M9 phase 5): one cache's sets × ways, the line
the last access used (green: hit, red: miss = filled), how its address
was cut into tag | index | offset, and the statistics with the 3 C's.

Everything comes from :mod:`emips.cacheview` (side-effect free reads of
the arrays and counters) and :meth:`Session.cache_accesses` (the recorded
accesses of the last step)."""

from __future__ import annotations

import html

from .. import cacheview as cv
from ..session import Session
from . import theme
from .i18n import tr
from .qt import QtCore, QtWidgets
from .widgets import _set, mono_font

Qt = QtCore.Qt
T = theme.colors


class CacheView(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._session: Session | None = None
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.which = QtWidgets.QComboBox()
        self.which.setToolTip(tr("Which cache to show"))
        self.all_sets = QtWidgets.QCheckBox(tr("all sets"))
        self.all_sets.setToolTip(tr(
            "List every set (caches of up to {n} sets); otherwise only the sets "
            "holding valid lines and the one last used").format(n=cv.ALL_SETS_LIMIT))
        self.reset = QtWidgets.QPushButton(tr("Reset statistics"))
        bar.addWidget(self.which)
        bar.addWidget(self.all_sets)
        bar.addStretch(1)
        bar.addWidget(self.reset)
        lay.addLayout(bar)
        self.address = QtWidgets.QLabel()
        self.address.setTextFormat(Qt.TextFormat.RichText)
        self.address.setWordWrap(True)
        lay.addWidget(self.address)
        split = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.setFont(mono_font())
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.stats = QtWidgets.QPlainTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setFont(mono_font())
        split.addWidget(self.table)
        split.addWidget(self.stats)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)
        self.which.currentIndexChanged.connect(lambda _: self._again())
        self.all_sets.toggled.connect(lambda _: self._again())
        self.reset.clicked.connect(self._reset_stats)

    def _again(self) -> None:
        if self._session is not None:
            self.refresh(self._session)

    def _reset_stats(self) -> None:
        s = self._session
        if s is not None and s.machine is not None and s.machine.caches is not None:
            s.machine.caches.reset_stats()
            self.refresh(s)

    def _caches(self, s: Session) -> list:
        m = s.machine
        return [] if m is None or m.caches is None else m.caches.all

    def current(self, s: Session):
        """The cache picked in the combo box (None: no caches)."""
        caches = self._caches(s)
        names = [c.name for c in caches]
        if names != [self.which.itemText(i) for i in range(self.which.count())]:
            keep = self.which.currentText()
            self.which.blockSignals(True)
            self.which.clear()
            self.which.addItems(names)
            if keep in names:
                self.which.setCurrentIndex(names.index(keep))
            elif "D-cache" in names:
                self.which.setCurrentIndex(names.index("D-cache"))
            self.which.blockSignals(False)
        i = self.which.currentIndex()
        return caches[i] if 0 <= i < len(caches) else None

    def refresh(self, s: Session) -> None:
        self._session = s
        c = self.current(s)
        if c is None:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.address.setText(tr("No caches. Turn them on in Configure > Settings > "
                                    "Caches (the program restarts)."))
            self.stats.setPlainText("")
            return
        acc = cv.last_access(s.cache_accesses(), c)
        self.address.setText(self._address_html(c, acc))
        focus = acc.set if acc is not None else None
        rows = cv.pane_sets(c, all_sets=self.all_sets.isChecked(), focus=focus)
        self.all_sets.setEnabled(c.sets <= cv.ALL_SETS_LIMIT)
        self.table.setColumnCount(c.ways)
        self.table.setHorizontalHeaderLabels([tr("way {n}").format(n=w) for w in range(c.ways)])
        self.table.setRowCount(len(rows))
        self.table.setVerticalHeaderLabels([tr("set {n}").format(n=r) for r in rows])
        big = s.machine.big_endian
        colors = T()
        for r, sidx in enumerate(rows):
            for wv in cv.set_view(c, sidx):
                bg = colors.valid_fill if wv.valid else None
                bold = False
                if acc is not None and acc.set == sidx and acc.way == wv.way:
                    bg = colors.hit_fill if acc.hit else colors.miss_fill
                    bold = True
                _set(self.table, r, wv.way, cv.cell_text(c, wv), bg, bold)
                self.table.item(r, wv.way).setToolTip(cv.cell_tip(c, sidx, wv, big))
        if focus is not None and focus in rows:
            self.table.scrollToItem(self.table.item(rows.index(focus), acc.way or 0))
        text = cv.stats_lines(c)
        if c.kind == "D" and s.machine.caches.wbb is not None:
            text += cv.wbb_lines(s.machine.caches.wbb)
        self.stats.setPlainText("\n".join(text))

    @staticmethod
    def _address_html(c, acc) -> str:
        e = html.escape
        head = e(f"{c.name}: {c.geometry.describe()}")
        if acc is None:
            legend = (tr("cells: tag (hex), a = age (0 = most recently used), M / E / S = "
                         "Modified / Exclusive / Shared (MESI), L = locked") if c.mesi else
                      tr("cells: tag (hex), a = age (0 = most recently used), D = dirty, "
                         "L = locked"))
            return (f"<b>{head}</b><br>" + e(tr("not used in the last step")) +
                    "<br><small>" + e(legend) + "</small>")
        sp = cv.split(c, acc.va, acc.pa)
        colors = T()
        cells = []
        for i, (name, msb, lsb, value, src) in enumerate(sp.fields()):
            if msb < lsb:
                continue
            addr = sp.pa if src == "PA" else sp.va
            bits = format((addr >> lsb) & ((1 << (msb - lsb + 1)) - 1), f"0{msb - lsb + 1}b")
            cells.append(
                f"<td bgcolor='{colors.fields[i]}' style='padding:2px 6px'>"
                f"<span style='color:{colors.highlight_text}'><b>{name}</b> "
                f"{src}[{msb}:{lsb}]<br><tt>{bits}</tt><br>0x{value:x} ({value})</span></td>")
        return (f"<b>{head}</b><br>{e(acc.describe())}"
                f"<table cellspacing=2><tr>{''.join(cells)}</tr></table>")
