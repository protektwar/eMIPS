"""The System pane (PLAN M11 phase 5): a multi-core system at a glance —
the cores (power state, position, mode, interrupt pins; click one to show
it everywhere), the Coherence Manager (the domain, what it did, the MESI
copies of the line the last access used), the GIC (the enabled sources,
every core's local interrupts and pins / EIC request) and the CPS
registers (GCR, CPC, GIC).

Everything comes from :mod:`emips.cpsview` (side-effect free)."""

from __future__ import annotations

from .. import cpsview as cv
from ..session import Session
from . import theme
from .i18n import tr
from .qt import QtCore, QtWidgets, Signal
from .widgets import _set, mono_font

Qt = QtCore.Qt
T = theme.colors


def _table(headers: list[str]) -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setFont(mono_font())
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(20)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _fill(t: QtWidgets.QTableWidget, rows: list[list[str]], fills=None) -> None:
    t.setRowCount(len(rows))
    for r, cells in enumerate(rows):
        bg = fills[r] if fills else None
        for c, text in enumerate(cells):
            _set(t, r, c, text, bg)
    t.resizeColumnsToContents()


class SystemView(QtWidgets.QTabWidget):
    """Cores / Coherence / GIC / Registers of a CPS (``--cores N``)."""

    core_chosen = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._session: Session | None = None
        # cores
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self.cores = _table([tr("core"), tr("state"), tr("pc"), tr("line"), tr("mode"),
                             tr("interrupt pins")])
        self.cores.setToolTip(tr("Click a core to show it in every pane"))
        self.cores.cellClicked.connect(lambda r, _c: self.core_chosen.emit(r))
        lay.addWidget(self.cores, 1)
        self.addTab(w, tr("Cores"))
        # coherence
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.domain = _table([tr("core"), tr("GCR_COHERENCE"), tr("mode"),
                              tr("hears interventions from")])
        lay.addWidget(self.domain, 1)
        self.counts = QtWidgets.QLabel()
        self.counts.setWordWrap(True)
        lay.addWidget(self.counts)
        self.line = QtWidgets.QLabel()
        self.line.setWordWrap(True)
        lay.addWidget(self.line)
        self.copies = _table([tr("core"), tr("MESI state")])
        lay.addWidget(self.copies, 1)
        self.addTab(w, tr("Coherence"))
        # GIC
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.gic_head = QtWidgets.QLabel()
        lay.addWidget(self.gic_head)
        self.sources = _table([tr("source"), tr("trigger"), tr("state"), tr("goes to"),
                               tr("cores")])
        lay.addWidget(self.sources, 2)
        self.locals = _table([tr("core"), tr("local interrupt"), tr("state"),
                              tr("enabled"), tr("goes to")])
        self.locals.setToolTip(tr("Each core's local interrupts (GIC_VL_PEND / MASK and "
                                  "their maps); the core's timer, performance counters "
                                  "and software interrupts when routable"))
        lay.addWidget(self.locals, 2)
        self.gic_cores = QtWidgets.QPlainTextEdit()
        self.gic_cores.setReadOnly(True)
        self.gic_cores.setFont(mono_font())
        lay.addWidget(self.gic_cores, 1)
        self.addTab(w, tr("GIC"))
        # registers
        self.regs = _table([tr("block"), tr("register"), tr("offset"), tr("value"),
                            tr("meaning")])
        self.addTab(self.regs, tr("Registers"))

    def refresh(self, s: Session) -> None:
        self._session = s
        if s.machine is None or s.cps is None:
            self.note.setText(tr("One core: no Coherent Processing System. Choose 2 to 6 "
                                 "cores in Configure > Settings > Processor."))
            for t in (self.cores, self.domain, self.copies, self.sources, self.locals,
                      self.regs):
                t.setRowCount(0)
            self.counts.setText("")
            self.line.setText("")
            self.gic_head.setText("")
            self.gic_cores.setPlainText("")
            return
        colors = T()
        cps = s.cps
        rows = cv.core_rows(s)
        self.note.setText(tr("{n} cores; every running core steps together, core 0 "
                             "first. The highlighted one is shown in the other panes.")
                          .format(n=len(rows)))
        d = s.hex_digits("pc")
        _fill(self.cores, [[str(r.cpu), r.state, f"0x{r.pc:0{d}x}" if r.pc is not None else "-",
                            str(r.line) if r.line else "-", r.mode, r.pins] for r in rows],
              [colors.valid_fill if r.selected else None for r in rows])
        dom = cv.domain_rows(cps)
        _fill(self.domain, [[str(d.cpu), f"0x{d.coherence:02x}",
                             tr("coherent") if d.coherent else tr("not coherent"),
                             ", ".join(map(str, d.hears)) or "-"] for d in dom])
        self.counts.setText(", ".join(f"{k} {v}" for k, v in cps.cm.counts.items())
                            + " — " + cps.cm.describe_error())
        last = cv.last_line(s) if s.machine.caches is not None and \
            s.machine.caches.d is not None else None
        if last is None:
            self.line.setText(tr("No data-cache access in the last step (of core {n})")
                              .format(n=s.cpu))
            self.copies.setRowCount(0)
        else:
            pa, how = last
            self.line.setText(tr("Line 0x{pa:08x} ({how}), in every core's data cache:")
                              .format(pa=pa, how=how))
            copies = cv.copies_rows(s, pa)
            fill = {"Modified": colors.miss_fill, "Exclusive": colors.hit_fill,
                    "Shared": colors.valid_fill}
            _fill(self.copies, [[str(j), st] for j, st in copies] or [["-", tr("no copy")]],
                  [fill.get(st) for _, st in copies] or None)
        gic = cps.gic
        if not gic.enabled:
            self.gic_head.setText(tr("GIC disabled (GCR_GIC_BASE.GIC_EN = 0)"))
            self.sources.setRowCount(0)
            self.locals.setRowCount(0)
            self.gic_cores.setPlainText("")
        else:
            self.gic_head.setText(tr("GIC at 0x{base:08x}: {n} sources, counter {c}")
                                  .format(base=gic.base, n=gic.n, c=gic.counter))
            src = cv.gic_rows(gic)
            _fill(self.sources, [[str(r.src), r.kind, tr("PENDING") if r.pending else tr("idle"),
                                  r.target, r.vps] for r in src],
                  [colors.hit_fill if r.pending else None for r in src])
            loc = cv.local_rows(gic)
            _fill(self.locals, [[str(r.cpu), r.name,
                                 tr("PENDING") if r.pending else tr("idle"),
                                 tr("yes") if r.enabled else tr("no"), r.target]
                                for r in loc],
                  [colors.hit_fill if r.pending and r.enabled else None for r in loc])
            self.gic_cores.setPlainText("\n".join(cv.gic_core_lines(gic)))
        regs = cv.register_rows(s)
        _fill(self.regs, [[r.block, r.name, f"+0x{r.offset:04x}", f"0x{r.value:08x}",
                           r.meaning] for r in regs])
