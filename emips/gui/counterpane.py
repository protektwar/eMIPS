"""The performance counters pane (PLAN M10 phase 5): the two counters of
CP0 register 25 — the event each one counts (the 24K's lists, Table 6.46),
the modes it counts in (U / S / K / EXL), the overflow interrupt, the
count — and the ratio of the two.

Setting a counter here does what software does with MTC0
(:meth:`Session.configure_counter`) and is kept across Reset / Step Back;
a program that writes the registers itself overrides it.  The pane shows
the registers as they are (it re-reads them every refresh)."""

from __future__ import annotations

from .. import perf
from ..session import Session
from .i18n import tr
from .qt import QtCore, QtWidgets
from .widgets import mono_font

Qt = QtCore.Qt
MODES = (("u", perf.CTL_U, "U"), ("s", perf.CTL_S, "S"), ("k", perf.CTL_K, "K"),
         ("x", perf.CTL_EXL, "EXL"))


class CounterView(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._session: Session | None = None
        self._loading = False
        lay = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        for c, head in enumerate((tr("counter"), tr("event"), tr("counts in"), tr("interrupt"),
                                  tr("count"))):
            label = QtWidgets.QLabel(f"<b>{head}</b>")
            grid.addWidget(label, 0, c)
        self.events: list[QtWidgets.QComboBox] = []
        self.modes: list[dict[str, QtWidgets.QCheckBox]] = []
        self.ie: list[QtWidgets.QCheckBox] = []
        self.counts: list[QtWidgets.QLabel] = []
        for i in range(2):
            grid.addWidget(QtWidgets.QLabel(tr("PerfCnt{n}").format(n=i)), i + 1, 0)
            box = QtWidgets.QComboBox()
            box.addItem(tr("(stopped)"), None)
            for n, ev in sorted(perf.EVENTS[i].items()):
                if not ev.key:
                    continue
                tag = {"the 24K pipeline model": tr(" [24K model]"),
                       "not modelled": tr(" [not modelled]")}.get(ev.source, "")
                box.addItem(f"{n}: {ev.name}{tag}", n)
            box.setToolTip(tr("What this counter counts (the 24K's events, Table 6.46)"))
            box.currentIndexChanged.connect(lambda _=0, i=i: self._apply(i))
            grid.addWidget(box, i + 1, 1)
            self.events.append(box)
            row = QtWidgets.QHBoxLayout()
            checks = {}
            for key, _, name in MODES:
                cb = QtWidgets.QCheckBox(name)
                cb.setToolTip({"u": tr("user mode"), "s": tr("supervisor mode"),
                               "k": tr("kernel mode (EXL = ERL = 0)"),
                               "x": tr("exception level (EXL = 1, ERL = 0)")}[key])
                cb.toggled.connect(lambda _=False, i=i: self._apply(i))
                row.addWidget(cb)
                checks[key] = cb
            w = QtWidgets.QWidget()
            w.setLayout(row)
            grid.addWidget(w, i + 1, 2)
            self.modes.append(checks)
            ie = QtWidgets.QCheckBox(tr("on overflow"))
            ie.setToolTip(tr("IE: bit 31 of the count set raises the performance counter "
                             "interrupt (Cause.PCI, IP7)"))
            ie.toggled.connect(lambda _=False, i=i: self._apply(i))
            grid.addWidget(ie, i + 1, 3)
            self.ie.append(ie)
            count = QtWidgets.QLabel("0")
            count.setFont(mono_font())
            count.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(count, i + 1, 4)
            self.counts.append(count)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)
        self.ratio = QtWidgets.QLabel()
        self.ratio.setWordWrap(True)
        lay.addWidget(self.ratio)
        note = QtWidgets.QLabel(tr("Setting a counter here is what software does with mtc0 "
                                   "$25 (select 0-3); the setting is kept across Reset and "
                                   "Step Back. Events marked [24K model] count in the 24K "
                                   "pipeline model only."))
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addStretch(1)

    def _apply(self, i: int) -> None:
        s = self._session
        if self._loading or s is None or s.machine is None:
            return
        event = self.events[i].currentData()
        modes = "".join(k for k, cb in self.modes[i].items() if cb.isChecked())
        if event is not None and not modes:
            modes = "uk"                         # a new event: user + kernel by default
        s.configure_counter(i, event, modes, self.ie[i].isChecked())
        self.refresh(s)

    def refresh(self, s: Session) -> None:
        self._session = s
        m = s.machine
        if m is not None and m.rv:              # a RISC-V hart: its counter CSRs
            for i in range(2):
                self.events[i].setEnabled(False)
                self.counts[i].setText("-")
            c = m.csrs
            if m.p8700 is not None:             # the P8700's four (MD01502 11.1)
                self.ratio.setText(tr(
                    "RISC-V: the counters are CSRs — cycle {cy:,}, instret {ir:,}.  The MIPS "
                    "P8700's mhpmcounter3-6 (mhpmevent3-6: csrw, or the debugger's perf "
                    "N EVENT):").format(cy=c.read(0xB00), ir=c.read(0xB02))
                    + "\n" + "\n".join(m.p8700.counters.lines()))
                return
            if c.hpm is not None:               # Zihpm: the SBI PMU events (emips.rvhpm)
                self.ratio.setText(tr(
                    "RISC-V: the counters are CSRs (Zicntr / Zihpm, priv. 3.1.10-3.1.11) — "
                    "cycle {cy:,}, instret {ir:,}, time {tm:,}.  mhpmcounter3-31 count the "
                    "SBI PMU event written to mhpmevent3-31 (csrw, or the debugger's perf N "
                    "EVENT; perf events lists them):").format(
                        cy=c.read(0xB00), ir=c.read(0xB02), tm=m.mtime)
                    + "\n" + "\n".join(c.hpm.lines()))
                return
            self.ratio.setText(tr(
                "RISC-V: the counters are CSRs (Zicntr, priv. 3.1.10) — cycle {cy:,}, instret "
                "{ir:,}, time {tm:,}; this hart has no mhpmcounter3-31 (Zihpm).  The MIPS "
                "performance counters above belong to MIPS cores.").format(
                    cy=c.read(0xB00), ir=c.read(0xB02), tm=m.mtime))
            return
        self._loading = True
        try:
            for i in range(2):
                enabled = m is not None
                self.events[i].setEnabled(enabled)
                if m is None:
                    self.counts[i].setText("-")
                    continue
                ctl = m.cp0_read(*perf.PERFCTL[i])
                on = bool(ctl & perf.CTL_MODES)
                event = (ctl >> 5) & 0x7F
                idx = self.events[i].findData(event) if on else 0
                self.events[i].setCurrentIndex(max(idx, 0))
                for key, bit, _ in MODES:
                    self.modes[i][key].setChecked(bool(ctl & bit))
                self.modes[i]["s"].setEnabled(m.mmu.supervisor)
                self.ie[i].setChecked(bool(ctl & perf.CTL_IE))
                self.counts[i].setText(f"{m.cp0_read(*perf.PERFCNT[i]):,}")
        finally:
            self._loading = False
        self.ratio.setText(self._ratio(m))

    @staticmethod
    def _ratio(m) -> str:
        if m is None:
            return ""
        rows = perf.rows(m)
        on = [(i, cnt, ev) for i, ctl, cnt, ev in rows if ctl & perf.CTL_MODES and ev]
        text = ""
        if len(on) == 2 and on[0][1]:
            (_, a, ea), (_, b, eb) = on
            text = tr("{b} / {a} = {r:.3f}").format(b=eb.name, a=ea.name, r=b / a)
            if {ea.key, eb.key} == {"cycles", "instructions"}:
                cyc = a if ea.key == "cycles" else b
                ins = b if ea.key == "cycles" else a
                if ins:
                    text += "  —  " + tr("CPI {c:.2f}").format(c=cyc / ins)
        if m.perf.pci:
            text += ("  " if text else "") + tr("Cause.PCI: a counter interrupt is pending")
        return text
