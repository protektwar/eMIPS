"""The Checks pane (PLAN M21 phase 2): the program's checks (emips.check)
run and their results — every run (the machine it ran on, what it took),
every check passed or failed with what the run had.

Each run is a fresh Session built by the window (its settings, the
editor's program, the run's own ``eMIPS:`` flags): the program being
debugged is left as it is.  The runs go a slice at a time from a timer,
so the window stays live and Stop works.  A check of the program's own
lines: double-click it to go to its line."""

from __future__ import annotations

from typing import Callable

from .. import check as K
from . import theme
from .i18n import tr
from .qt import QtCore, QtGui, QtWidgets, Signal
from .widgets import mono_font

Qt = QtCore.Qt
T = theme.colors

#: steps per timer tick (a few tens of milliseconds)
SLICE = 2000
#: Qt.ItemDataRole.UserRole: an item's (passed, line)
ROLE = Qt.ItemDataRole.UserRole


class CheckView(QtWidgets.QWidget):
    """Check / Stop, where the checks come from, the results."""

    #: Check was pressed (the window collects the checks and calls start)
    check_requested = Signal()
    #: Checks File… / Program's Checks
    choose_requested = Signal()
    clear_requested = Signal()
    #: a check of the program's own lines: its line in the editor (1-based)
    line_chosen = Signal(int)
    #: the checks are over (passed, total)
    finished = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.check_button = QtWidgets.QPushButton(tr("Check"))
        self.check_button.setToolTip(tr("Run the program as its checks say (F8): its "
                                        "'# check:' lines, the checks file named like it, "
                                        "or the one chosen here"))
        self.check_button.clicked.connect(self.check_requested.emit)
        self.stop_button = QtWidgets.QPushButton(tr("Stop"))
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.source = QtWidgets.QLabel()
        self.source.setWordWrap(True)
        self.choose_button = QtWidgets.QPushButton(tr("Checks File…"))
        self.choose_button.setToolTip(tr("Check against a checks file (a teacher's): it "
                                         "replaces the program's own checks"))
        self.choose_button.clicked.connect(self.choose_requested.emit)
        self.clear_button = QtWidgets.QPushButton(tr("Program's Checks"))
        self.clear_button.setToolTip(tr("Forget the chosen checks file: the program's own "
                                        "checks again"))
        self.clear_button.clicked.connect(self.clear_requested.emit)
        for w in (self.check_button, self.stop_button):
            row.addWidget(w)
        row.addWidget(self.source, 1)
        row.addWidget(self.choose_button)
        row.addWidget(self.clear_button)
        lay.addLayout(row)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels([tr("result"), tr("check"), tr("the run had")])
        self.tree.setRootIsDecorated(True)
        self.tree.setFont(mono_font())
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        lay.addWidget(self.tree, 1)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self._tick)
        self._queue: list[K.Run] = []
        self._make: Callable[[K.Run], object] | None = None
        self._limit: int | None = None
        self._lines = False                     # the checks are the program's own lines
        self._ex: K.Execution | None = None
        self._item: QtWidgets.QTreeWidgetItem | None = None
        self._number = 0
        self.passed = self.total = 0
        self.results: list[K.RunResult] = []
        self.set_source(None)

    # -- the source --------------------------------------------------------------------

    def set_source(self, file: str | None, chosen: bool = False) -> None:
        """Say where the checks come from (``file``: a checks file;
        ``chosen``: picked here, not found next to the program)."""
        if file is None:
            self.source.setText(tr("the program's own '# check:' lines"))
        else:
            self.source.setText((tr("checks file {name} (chosen)") if chosen
                                 else tr("checks file {name}")).format(name=file))
        self.clear_button.setEnabled(chosen)

    @property
    def running(self) -> bool:
        return self.timer.isActive()

    # -- running ---------------------------------------------------------------------------

    def start(self, runs: list[K.Run], make: Callable[[K.Run], object], *,
              lines: bool, limit: int | None = None) -> None:
        """Run ``runs``: ``make(run)`` builds each one's loaded Session (a
        ValueError: that run cannot be made); ``lines``: the checks are the
        program's own lines (double-click goes to them); ``limit``: the step
        limit of a run that names none (default: check.default_limit)."""
        self.stop()
        self.tree.clear()
        self.results = []
        self.passed = self.total = 0
        self._queue = list(runs)
        self._make, self._limit, self._lines = make, limit, lines
        self._number = 0
        self.check_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._next()

    def show_problem(self, text: str, line: int | None = None) -> None:
        """No run: the checks cannot be read (``line``: the program's line)."""
        self.stop()
        self.tree.clear()
        item = QtWidgets.QTreeWidgetItem([tr("error"), text, ""])
        item.setData(0, ROLE, (False, line))
        self.tree.addTopLevelItem(item)
        self._paint(item)
        self.summary.setText(text)
        self.tree.resizeColumnToContents(0)

    def stop(self) -> None:
        """Stop the run in progress (it is shown as stopped)."""
        if self.timer.isActive():
            self.timer.stop()
            if self._item is not None:
                self._item.setText(0, tr("stopped"))
                self._item.setText(2, tr("after {n} steps").format(
                    n=self._ex.done if self._ex is not None else 0))
            self.summary.setText(tr("stopped — {p} of {t} checks passed so far").format(
                p=self.passed, t=self.total))
        self._queue = []
        self._ex = None
        self.check_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _next(self) -> None:
        if not self._queue:
            self._done()
            return
        run = self._queue.pop(0)
        self._number += 1
        name = tr("run {n}").format(n=self._number) + (f": {run.name}" if run.name else "")
        self._item = QtWidgets.QTreeWidgetItem([tr("running"), name, ""])
        self.tree.addTopLevelItem(self._item)
        try:
            session = self._make(run)
        except ValueError as exc:
            self._item.setText(0, tr("FAIL"))
            self._item.setText(2, str(exc))
            self._item.setData(0, ROLE, (False, (run.flags_line or run.line)
                                         if self._lines else None))
            self._paint(self._item)
            self.total += 1 + len(run.checks)
            self._next()
            return
        self._ex = K.Execution(session, run,
                               limit=self._limit or K.default_limit(session))
        self.timer.start()

    def _tick(self) -> None:
        ex = self._ex
        if ex is None:
            self.timer.stop()
            return
        if not ex.advance(SLICE):
            self._item.setText(2, tr("{n} steps…").format(n=ex.done))
            return
        self.timer.stop()
        self._show(ex.result())
        self._ex = None
        self._next()

    def _show(self, res: K.RunResult) -> None:
        item = self._item
        self.results.append(res)
        item.setText(2, res.summary)
        if res.output:
            out = QtWidgets.QTreeWidgetItem(["", tr("output"), K.quote(res.output)])
            out.setToolTip(2, res.output)
            item.addChild(out)
        passed = total = 0
        if res.run.after is None:
            total += 1
            passed += res.finished
            done = QtWidgets.QTreeWidgetItem([
                tr("PASS") if res.finished else tr("FAIL"), tr("the program finishes"),
                "" if res.finished else res.why])
            done.setData(0, ROLE, (res.finished, None))
            item.addChild(done)
            self._paint(done)
        for o in res.outcomes:
            total += 1
            passed += o.passed
            text = o.check.text + (f"  ({o.check.note})" if o.check.note else "")
            child = QtWidgets.QTreeWidgetItem([tr("PASS") if o.passed else tr("FAIL"), text,
                                               "" if o.passed else o.found])
            child.setData(0, ROLE, (o.passed, o.check.line if self._lines else None))
            child.setToolTip(1, (tr("line {n} — double-click to go there").format(
                n=o.check.line) if self._lines else ""))
            item.addChild(child)
            self._paint(child)
        ok = passed == total
        item.setText(0, tr("PASS") if ok else tr("FAIL"))
        item.setData(0, ROLE, (ok, None))
        self._paint(item)
        item.setExpanded(True)
        self.passed += passed
        self.total += total
        for c in range(3):
            self.tree.resizeColumnToContents(c)

    def _done(self) -> None:
        self.check_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._item = None
        if self.total and self.passed == self.total:
            self.summary.setText(tr("all {t} checks passed").format(t=self.total))
        else:
            self.summary.setText(tr("{p} of {t} checks passed").format(
                p=self.passed, t=self.total))
        self.finished.emit(self.passed, self.total)

    # -- looks ------------------------------------------------------------------------------

    def _paint(self, item: QtWidgets.QTreeWidgetItem) -> None:
        data = item.data(0, ROLE)
        if data is None:
            return
        c = T()
        fill = QtGui.QColor(c.hit_fill if data[0] else c.miss_fill)
        item.setBackground(0, QtGui.QBrush(fill))
        item.setForeground(0, QtGui.QBrush(QtGui.QColor(c.highlight_text)))

    def restyle(self) -> None:
        """Repaint the results in the current theme's colours."""
        def walk(item):
            self._paint(item)
            for i in range(item.childCount()):
                walk(item.child(i))
        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))

    def _double_clicked(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        data = item.data(0, ROLE)
        if data is not None and data[1]:
            self.line_chosen.emit(int(data[1]))
