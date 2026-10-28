"""The Trace pane (PLAN M14 phase 4): MIPS Trace as a trace probe sees
it — the settings it scans into the selected core's TCB (TCBCONTROLA /
B / C: trace on, what besides the PC, cycle accuracy, Trace-To / -From,
all branches, inhibit overflow), the state of the TCB, and the program
flow regenerated from the trace words (:meth:`Session.trace_lines`).

The settings are TAP scans (:meth:`Session.trace_start`), recorded like
the probe's other events: kept across Reset and Step Back.  A program
that takes over with TraceControl.TS = 1 overrides them.

On a RISC-V hart (LF11) the same pane drives the RISC-V Trace Control
Interface (emips.rvtrace) as a debugger does through System Bus Access:
the protocol (E-Trace / N-Trace), the optimizations a user picks most
(implicit return, branch prediction and the jump target cache for
E-Trace, HTM and repeated history for N-Trace, sequential jumps,
timestamps), the components' state, and the regenerated flow — or the
packets / messages the encoder sent, with why."""

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
        # the RISC-V controls (LF11)
        self.protocol = QtWidgets.QComboBox()
        self.protocol.addItem(tr("E-Trace"), "e")
        self.protocol.addItem(tr("N-Trace"), "n")
        self.protocol.setToolTip(tr("trTeFormat: Efficient Trace for RISC-V 2.0 or the "
                                    "Nexus-based N-Trace 1.0"))
        self.protocol.currentIndexChanged.connect(self._apply)
        row.addWidget(self.protocol)
        self.rv_options: dict[str, QtWidgets.QCheckBox] = {}
        for key, text, tip in (
                ("return", tr("Implicit return"), tr("trTeInstEnImplicitReturn: a return "
                                                     "to the predicted address sends nothing")),
                ("predict", tr("Branch prediction"), tr("E-Trace: correctly predicted "
                                                        "branches as a count")),
                ("cache", tr("Jump cache"), tr("E-Trace: uninferable targets as a jump "
                                               "target cache index")),
                ("htm", tr("HTM"), tr("N-Trace: branch history (one bit a branch; "
                                      "off: BTM, a message a taken branch)")),
                ("repeat", tr("Repeated history"), tr("N-Trace: repeated HIST records and "
                                                      "branch messages counted")),
                ("sijump", tr("Sequential jumps"), tr("LUI / AUIPC + JALR: inferable")),
                ("time", tr("Timestamps"), tr("trTsEnable: the hart's cycles"))):
            cb = QtWidgets.QCheckBox(text)
            cb.setToolTip(tip)
            cb.toggled.connect(self._apply)
            self.rv_options[key] = cb
            row.addWidget(cb)
        self.rv_options["htm"].setChecked(True)
        self.packets = QtWidgets.QCheckBox(tr("Packets"))
        self.packets.setToolTip(tr("show the packets / messages the encoder sent instead "
                                   "of the regenerated flow"))
        self.packets.toggled.connect(lambda _c: self._session and self._show(self._session))
        row.addWidget(self.packets)
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

    def _rv(self) -> bool:
        s = self._session
        return s is not None and s.machine is not None and s.machine.rv

    def _apply(self, _checked: bool = False) -> None:
        s = self._session
        if self._loading or s is None or s.machine is None:
            return
        if self._rv():
            self._apply_rv(s)
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

    def _apply_rv(self, s: Session) -> None:
        if s.rvtrace is None:
            return
        o = {k: cb.isChecked() for k, cb in self.rv_options.items()}
        proto = self.protocol.currentData()
        if self.on.isChecked():
            opts = tuple(k for k in ("return", "sijump") if o[k])
            if proto == "e":
                opts += tuple(k for k in ("predict", "cache") if o[k])
            elif o["repeat"]:
                opts += ("repeat",)
            s.rv_trace_start(protocol=proto, options=opts, htm=o["htm"],
                             timestamps=o["time"])
        elif any(e.enabled for e in s.rvtrace.encoders):
            s.trace_stop()
        self.refresh(s)

    def refresh(self, s: Session) -> None:
        self._session = s
        m = s.machine
        self._loading = True
        rv = m is not None and m.rv
        try:
            for cb in self.options.values():
                cb.setVisible(not rv)
            for w in [self.protocol, self.packets, *self.rv_options.values()]:
                w.setVisible(rv)
            for cb in [self.on, *self.options.values(), *self.rv_options.values()]:
                cb.setEnabled(m is not None)
            if m is None:
                self.state.setText("")
                self.text.setPlainText("")
                return
            if rv:
                ts = s.rvtrace
                if ts is None:
                    self.on.setEnabled(False)
                    self.state.setText(tr("The P8700 traces through its Hybrid Debug "
                                          "(not modelled here)"))
                    self.text.setPlainText("")
                    return
                e = ts.encoders[s.cpu]
                self.on.setChecked(e.enabled)
                if e.enabled:
                    self.protocol.setCurrentIndex(e.format)
                    f = e.features
                    self.rv_options["return"].setChecked(bool(f & 8))
                    self.rv_options["predict"].setChecked(bool(f & 0x10))
                    self.rv_options["cache"].setChecked(bool(f & 0x20))
                    self.rv_options["repeat"].setChecked(bool(f & 0x100))
                    self.rv_options["sijump"].setChecked(bool(f & 4))
                    self.rv_options["htm"].setChecked((e.control >> 4 & 7) != 3)
                    self.rv_options["time"].setChecked(e.ts.enabled)
                n = self.protocol.currentData() == "n"
                for k in ("predict", "cache"):
                    self.rv_options[k].setEnabled(not n)
                for k in ("htm", "repeat"):
                    self.rv_options[k].setEnabled(n)
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
        if s.machine is None or (s.machine.rv and s.rvtrace is None):
            return
        self.state.setText("\n".join(s.trace_state()[:3 if s.machine.rv else 2]))
        if not self.isVisible():
            return                              # regeneration only for a shown pane
        if s.machine.rv and self.packets.isChecked():
            from ..sessionrv import trace_packet_lines
            lines = trace_packet_lines(s, 200)
        else:
            lines = s.trace_lines(200)
        self.text.setPlainText("\n".join(line.strip() for line in lines)
                               if lines else tr("Nothing traced yet: tick Trace, then run."))
        bar = self.text.verticalScrollBar()
        bar.setValue(bar.maximum())
