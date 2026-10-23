"""The MT pane (PLAN M19 phase 2): the MT Module at a glance — the VPEs,
every thread context (its state, PC, flags, group, what it waits for;
click one for its registers and to show its VPE everywhere), the policy
manager with its PM_sys_avail input, the YIELD qualifier inputs, and the
ITC cells.

Everything comes from :mod:`emips.mtview` (side-effect free); the inputs
go through the Session (recorded: Step Back replays them)."""

from __future__ import annotations

from .. import mtview as mv
from ..session import Session
from . import theme
from .i18n import tr
from .qt import QtCore, QtWidgets, Signal
from .systempane import _fill, _table
from .widgets import mono_font

Qt = QtCore.Qt
T = theme.colors


class MTView(QtWidgets.QTabWidget):
    """Threads / ITC of an MT processor (``--mt VPES:TCS``)."""

    #: a VP to show in every pane (a TC's row was clicked)
    vp_chosen = Signal(int)
    #: the YIELD qualifier inputs / PM_sys_avail were set (the window steps)
    inputs_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._session: Session | None = None
        self._tc_key: int | None = None
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.head = QtWidgets.QLabel()
        self.head.setWordWrap(True)
        self.head.setFont(mono_font())
        lay.addWidget(self.head)
        self.vpes = _table([tr("VP"), tr("core"), tr("VPE"), tr("VPA"), tr("MVP"), tr("TE"),
                            tr("EXCPT"), tr("YQMask"), tr("runs only")])
        self.vpes.setToolTip(tr("VPEConf0.VPA / MVP, VPEControl.TE / EXCPT, YQMask; a VPE "
                                "with EXL / ERL, in Debug Mode or with TE = 0 runs only its "
                                "XTC (MD00378 6.5, 6.6)"))
        lay.addWidget(self.vpes, 1)
        self.tcs = _table([tr("TC"), tr("core"), tr("VP"), tr("state"), tr("pc"), tr("line"),
                           tr("flags"), tr("group"), tr("issued"), tr("waits for")])
        self.tcs.setToolTip(tr("Every thread context; > marks the one that issues next. "
                               "Click one for its registers (and to show its VPE in the "
                               "other panes)"))
        self.tcs.cellClicked.connect(self._tc_clicked)
        lay.addWidget(self.tcs, 2)
        self.tc_regs = QtWidgets.QPlainTextEdit()
        self.tc_regs.setReadOnly(True)
        self.tc_regs.setFont(mono_font())
        lay.addWidget(self.tc_regs, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(tr("YIELD qualifier inputs")))
        self.yq = QtWidgets.QLineEdit("0x0000")
        self.yq.setMaximumWidth(90)
        self.yq.setToolTip(tr("The 16 yield_qualifier pins a blocked YIELD waits for "
                              "(hexadecimal)"))
        row.addWidget(self.yq)
        self.yq_set = QtWidgets.QPushButton(tr("Set"))
        self.yq_set.clicked.connect(self._set_yq)
        row.addWidget(self.yq_set)
        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel(tr("PM_sys_avail")))
        self.sys = QtWidgets.QSpinBox()
        self.sys.setRange(0, 7)
        self.sys.setValue(7)
        self.sys.setToolTip(tr("The WRR2 policy manager's system-buffer input: free "
                               "entries (0-7) its throttles compare with their thresholds"))
        self.sys.valueChanged.connect(self._set_sys)
        row.addWidget(self.sys)
        row.addStretch(1)
        lay.addLayout(row)
        self.addTab(w, tr("Threads"))
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.itc_head = QtWidgets.QLabel()
        self.itc_head.setWordWrap(True)
        lay.addWidget(self.itc_head)
        self.cells = _table([tr("cell"), tr("kind"), tr("address"), tr("E/F/T"),
                             tr("data (oldest first)")])
        self.cells.setToolTip(tr("The ITC cells (MD00534 ch. 12): a load from an empty "
                                 "cell / a store to a full one through the synchronized "
                                 "view blocks the thread"))
        lay.addWidget(self.cells, 1)
        self.addTab(w, tr("ITC"))

    # -- the inputs ------------------------------------------------------------------

    def _set_yq(self) -> None:
        s = self._session
        if s is None or s.mt is None:
            return
        try:
            value = int(self.yq.text().strip(), 0)
        except ValueError:
            self.yq.setText("0x0000")
            return
        s.yield_inputs(value)
        self.inputs_changed.emit()

    def _set_sys(self, value: int) -> None:
        s = self._session
        if s is None or s.mt is None or self._filling:
            return
        s.pm_input(value)
        self.inputs_changed.emit()

    _filling = False

    def _tc_clicked(self, row: int, _col: int) -> None:
        s = self._session
        if s is None:
            return
        rows = mv.tc_rows(s)
        if not 0 <= row < len(rows):
            return
        self._tc_key = rows[row].key
        self.vp_chosen.emit(rows[row].vp)
        self._show_tc(s)

    def _show_tc(self, s: Session) -> None:
        got = mv.tc_of(s, self._tc_key) if self._tc_key is not None else None
        if got is None:
            self.tc_regs.setPlainText("")
            return
        p, tc, vp = got
        if s.cpu != vp:
            self.tc_regs.setPlainText("")
            return
        self.tc_regs.setPlainText("\n".join(s.tc_lines(tc.n)))

    # -- refresh ----------------------------------------------------------------------

    def refresh(self, s: Session) -> None:
        self._session = s
        if s.machine is None or s.mt is None:
            self.head.setText(tr("The MT Module is a MIPS ASE (MD00378): a RISC-V hart has "
                                 "none.") if s.machine is not None and s.machine.rv else
                              tr("No MT Module. Choose one in Configure > Settings > "
                                 "Processor (MT Module), or use --mt VPES:TCS."))
            for t in (self.vpes, self.tcs, self.cells):
                t.setRowCount(0)
            self.tc_regs.setPlainText("")
            self.itc_head.setText("")
            return
        colors = T()
        self.head.setText("\n".join(mv.header(s)))
        vpes = mv.vpe_rows(s)
        _fill(self.vpes, [[str(r.vp), "-" if r.core is None else str(r.core), str(r.vpe),
                           str(int(r.vpa)), str(int(r.mvp)), str(int(r.te)), str(r.excpt),
                           f"0x{r.yqmask:04x}",
                           f"TC{r.xtc} ({r.exclusive})" if r.exclusive else "-"]
                          for r in vpes],
              [colors.valid_fill if r.vp == s.cpu else None for r in vpes])
        tcs = mv.tc_rows(s)
        fill = {"running": colors.hit_fill, "halted": None, "free": None}
        d = s.hex_digits("pc")
        _fill(self.tcs, [[(">" if r.next else " ") + f"TC{r.tc}",
                          "-" if r.core is None else str(r.core), str(r.vp), r.state,
                          f"0x{r.pc:0{d}x}", str(r.line) if r.line else "-", r.flags or "-",
                          "-" if r.group is None else str(r.group), str(r.issued),
                          r.waits or "-"] for r in tcs],
              [fill.get(r.state, colors.miss_fill) for r in tcs])
        self._show_tc(s)
        procs = mv.processors(s)
        if procs:
            p = procs[0][2]
            self._filling = True
            self.sys.setValue(p.sys_avail)
            self._filling = False
            if not self.yq.hasFocus():
                self.yq.setText(f"0x{p.yq_ext:04x}")
        self.itc_head.setText(mv.itc_header(s))
        cells = mv.cell_rows(s)
        _fill(self.cells, [[str(r.cell), r.kind,
                            f"0x{r.address:08x}" if r.address is not None else "-",
                            r.flags, r.data] for r in cells])
