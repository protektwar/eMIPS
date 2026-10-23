"""The Trace pane (PLAN M14 phase 4): MIPS Trace as a trace probe sees
it — the settings it scans into the selected core's TCB (TCBCONTROLA /
B / C: trace on, what besides the PC, cycle accuracy, Trace-To / -From,
all branches, inhibit overflow), the state of the TCB, and the program
flow regenerated from the trace words (:meth:`Session.trace_lines`).

The settings are TAP scans (:meth:`Session.trace_start`), recorded like
the probe's other events: kept across Reset and Step Back.  A program
that takes over with TraceControl.TS = 1 overrides them."""

from __future__ import annotations

from .. import pdtrace as PDT
from ..session import Session
from .i18n import tr
from .qt import QtWidgets
from .widgets import mono_font


class TraceView(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._session: Session | None = None
        self._loading = False
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.on = QtWidgets.QCheckBox(tr("Trace"))
        self.on.setToolTip(tr("TCBCONTROLA.On and TCBCONTROLB.EN: trace every mode into "
                              "the on-chip trace memory (8 KB)"))
        self.options: dict[str, QtWidgets.QCheckBox] = {}
        for key, text, tip in (
                ("addr", tr("Addresses"), tr("load and store addresses (TLA / TSA)")),
                ("data", tr("Data"), tr("load and store data (TD)")),
                ("ca", tr("Cycle-accurate"), tr("CA: a TF1 for every cycle without a "
                                                "completed instruction")),
                ("from", tr("Trace-From"), tr("stop when the trace memory is full "
                                              "(default Trace-To: keep the newest)")),
                ("tb", tr("All branches"), tr("TB: the PC of every branch, also the ones "
                                              "the program predicts")),
                ("io", tr("No overflow"), tr("IO: stall the core instead of losing "
                                             "trace when the FIFO is full"))):
            cb = QtWidgets.QCheckBox(text)
            cb.setToolTip(tip)
            cb.toggled.connect(self._apply)
            self.options[key] = cb
        self.on.toggled.connect(self._apply)
        row.addWidget(self.on)
        for cb in self.options.values():
            row.addWidget(cb)
        row.addStretch(1)
        lay.addLayout(row)
        self.state = QtWidgets.QLabel()
        self.state.setWordWrap(True)
        lay.addWidget(self.state)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(mono_font())
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        lay.addWidget(self.text, 1)

    def _apply(self, _checked: bool = False) -> None:
        s = self._session
        if self._loading or s is None or s.machine is None:
            return
        o = {k: cb.isChecked() for k, cb in self.options.items()}
        if self.on.isChecked():
            mode = PDT.MODE_PC | (PDT.MODE_LA | PDT.MODE_SA if o["addr"] else 0) \
                | (PDT.MODE_LD | PDT.MODE_SD if o["data"] else 0)
            s.trace_start(mode=mode, ca=o["ca"], trace_from=o["from"], tb=o["tb"],
                          io=o["io"])
        elif s.machine.ejtag.tap.tcb.enabled or s.machine.pdt.on:
            s.trace_stop()
        self.refresh(s)

    def refresh(self, s: Session) -> None:
        self._session = s
        m = s.machine
        self._loading = True
        try:
            for cb in [self.on, *self.options.values()]:
                cb.setEnabled(m is not None)
            if m is None or m.rv:               # (MIPS Trace: a MIPS core's)
                for cb in [self.on, *self.options.values()]:
                    cb.setEnabled(False)
                self.state.setText("" if m is None else tr(
                    "MIPS Trace (PDtrace and the TCB, MD00439) belongs to MIPS cores: a "
                    "RISC-V hart has none"))
                self.text.setPlainText("")
                return
            tcb = m.ejtag.tap.tcb
            st = m.pdt.settings()
            on = st["on"] and tcb.enabled
            self.on.setChecked(on)
            if not on:                          # the settings for the next start
                return
            self.options["addr"].setChecked(bool(st["mode"] & PDT.MODE_LA))
            self.options["data"].setChecked(bool(st["mode"] & PDT.MODE_LD))
            self.options["ca"].setChecked(tcb.ca)
            self.options["from"].setChecked(bool(tcb.control_b >> PDT.TB_TM_SHIFT & 1))
            self.options["tb"].setChecked(st["tb"])
            self.options["io"].setChecked(st["io"])
        finally:
            self._loading = False
            self._show(s)

    def _show(self, s: Session) -> None:
        if s.machine is None or s.machine.rv:
            return
        self.state.setText("\n".join(s.trace_state()[:2]))
        if not self.isVisible():
            return                              # regeneration only for a shown pane
        lines = s.trace_lines(200)
        self.text.setPlainText("\n".join(line.strip() for line in lines)
                               if lines else tr("Nothing traced yet: tick Trace, then run."))
        bar = self.text.verticalScrollBar()
        bar.setValue(bar.maximum())
