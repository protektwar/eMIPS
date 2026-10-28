"""The eMIPS main window (PLAN M8).

The source editor is the central widget; every other pane is a dock that
can be moved, tabbed with others or detached (Datapath, Pipeline,
Disassembly, Instruction, Registers, Console, Memory, Messages).  View >
Reset layout restores the default arrangement.

Menus: File (new, open .asm / Intel HEX, recent, examples, save, export,
quit), Edit (undo ... find / replace, go to line, comment, indent,
breakpoints), Run (assemble, run, step, step back, run to cursor, reset,
mode, forwarding), View (panes, register names, zoom, layout),
Tools (instruction reference, encoder / decoder, number converter — panes
too, hidden until opened), Configure (settings, release, byte order,
theme, language), Help.  The toolbar keeps the
run controls, the mode, forwarding and the speed.

Window layout, recent files and options are remembered between runs
(``QSettings``, INI format, per user).  Starts with an empty editor;
``python3 -m emips gui prog.asm`` (or ``prog.hex``) opens that file.
"""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

from . import help as helptext
from . import i18n
from .cachepane import CacheView
from .counterpane import CounterView
from .tracepane import TraceView
from .boardpane import BoardView
from .guideview import GuideView
from .partdialog import PartDialog
from .sourceview import SourceView
from .systempane import SystemView
from .mtpane import MTView
from .checkpane import CheckView
from .datapath import DatapathView
from .core8700 import Core8700View
from .datapath24k import Datapath24KView
from .examples import EXAMPLES, Example
from .i18n import tr
from .. import check
from ..options import MAX_CORES, Options, arch_title, gdb_program
from .qt import QtCore, QtGui, QtWidgets
from ..session import Session
from .settings import SettingsDialog
from .tlbpane import TLBView
from .toolviews import ConverterView, EncoderView, ReferenceView
from . import theme
from .widgets import (CodeEditor, ConsoleView, DisassemblyView, EditorLegend, FindBar,
                      InstrPanel, MemoryView, PipelineChart, RegisterView)

Qt = QtCore.Qt

#: run speed: (instructions or cycles per timer tick, tick in ms)
SPEEDS = [(1, 600), (1, 250), (1, 80), (10, 30), (200, 10), (5000, 1)]
RELEASES = ["R1", "R2", "R3", "R5", "R6"]
#: the Machine toolbar's / Configure menu's (architecture, release) choices
_ARCH_RELEASES = [(a, r) for a in ("mips32", "mips64") for r in RELEASES] + [
    ("nanomips", "R6"), ("rv32", "RV"), ("rv64", "RV")]      # (RISC-V: no release, PLAN M25)


def _release_action(arch: str, release: str) -> str:
    """The Configure menu action of an (architecture, release)."""
    if arch in ("rv32", "rv64"):
        return f"release_{arch}"
    return {"mips32": "release", "mips64": "release64",
            "nanomips": "release_nano"}[arch] + f"_{release}"


def _arch_key(arch: str, release: str) -> str:
    """The Machine toolbar's item of an architecture and release ("rv32:RV":
    RISC-V has no releases)."""
    return f"{arch}:RV" if arch in ("rv32", "rv64") else f"{arch}:{release}"


def _arch_text(arch: str, release: str) -> str:
    return (f"RISC-V {arch_title(arch)}" if arch in ("rv32", "rv64")
            else f"{arch_title(arch)} {release}")
HEX_SUFFIXES = (".hex", ".ihex")
TOOL_DOCKS = ("reference", "encoder", "converter")
#: File > New C Program
C_TEMPLATE = """/* A C program for eMIPS: Build (Ctrl+B) compiles it for the selected
 * MIPS32 release; Step (F10) runs one instruction, Step Line (F11) one C
 * line; the Disassembly pane shows the code the compiler made. */
#include "emips.h"

int square(int x)
{
    return x * x;
}

int main(void)
{
    int sum = 0;
    for (int i = 1; i <= 4; i++)
        sum += square(i);
    print_int(sum);
    print_char('\\n');
    return 0;
}
"""
#: events telling that the palette may have changed (ThemeChange is not
#: exposed by every binding / Qt version)
_PALETTE_EVENTS = tuple(getattr(QtCore.QEvent.Type, n) for n in (
    "PaletteChange", "ApplicationPaletteChange", "ThemeChange")
    if hasattr(QtCore.QEvent.Type, n))
MAX_RECENT = 8
DEFAULT_FONT = 10


def default_settings() -> QtCore.QSettings:
    """Per-user INI file (``~/.config/eMIPS/eMIPS.ini`` on Linux)."""
    return QtCore.QSettings(QtCore.QSettings.Format.IniFormat,
                            QtCore.QSettings.Scope.UserScope, "eMIPS", "eMIPS")


def dark_palette() -> QtGui.QPalette:
    p = QtGui.QPalette()
    R, G = QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup
    for role, color in ((R.Window, "#2b2b2b"), (R.WindowText, "#f0f0f0"),
                        (R.Base, "#1e1e1e"), (R.AlternateBase, "#262626"),
                        (R.ToolTipBase, "#2b2b2b"), (R.ToolTipText, "#f0f0f0"),
                        (R.Text, "#f0f0f0"), (R.Button, "#3a3a3a"),
                        (R.ButtonText, "#f0f0f0"), (R.BrightText, "#ff6b6b"),
                        (R.Link, "#6aa9ff"), (R.Highlight, "#2f65ca"),
                        (R.HighlightedText, "#ffffff"), (R.PlaceholderText, "#9a9a9a")):
        p.setColor(role, QtGui.QColor(color))
    for role in (R.Text, R.ButtonText, R.WindowText):
        p.setColor(G.Disabled, role, QtGui.QColor("#7f7f7f"))
    p.setColor(G.Disabled, R.Light, QtGui.QColor(0, 0, 0, 0))   # no "etched" shadow
    return p


def _alive(obj) -> bool:
    """Is the Qt object behind this Python wrapper still there?"""
    try:
        obj.objectName()
    except RuntimeError:                    # "wrapped C/C++ object ... deleted"
        return False
    return True


class DockBar(QtWidgets.QWidget):
    """The bar inside an undocked pane: a Dock button (and a double-click
    on the bar) puts the pane back into the main window; dragging the bar
    still lets Qt drop it into a dock area."""

    def __init__(self, dock: QtWidgets.QDockWidget) -> None:
        super().__init__(dock)
        self.dock = dock
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 1, 4, 1)
        lay.addStretch(1)
        self.button = QtWidgets.QToolButton()
        self.button.setText(tr("Dock"))
        self.button.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_TitleBarNormalButton))
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setAutoRaise(True)
        self.button.setToolTip(tr("Put this pane back into the main window "
                                  "(or double-click this bar)"))
        self.button.clicked.connect(lambda: dock.setFloating(False))
        lay.addWidget(self.button)

    def mouseDoubleClickEvent(self, event) -> None:
        self.dock.setFloating(False)


def _has_checks(p: Path) -> bool:
    """A lab program with checks: its own lines, or NAME.check next to it."""
    if p.with_suffix(".check").is_file():
        return True
    try:
        return check.has_checks(p.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return False


def _mode_index(s, mode: str | None = None) -> int:
    """The mode box's item: instruction, 5-stage, 24K, P8700."""
    if (mode or s.mode) != "pipeline":
        return 0
    return {"24k": 2, "p8700": 3}.get(s.options.model, 1)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path: str | None = None,
                 settings: QtCore.QSettings | None = None) -> None:
        super().__init__()
        self.settings = settings if settings is not None else default_settings()
        self.session = Session()
        self.path: Path | None = None
        #: Intel HEX text when a HEX image is loaded (the editor is read-only)
        #: a read-only binary: Intel HEX text, or an ELF executable's bytes
        self.hex_text: str | bytes | None = None
        #: the editor's language: "asm" or "c" (Build compiles it)
        self.lang = "asm"
        #: Run > Checks File…: the checks file chosen for this program (None:
        #: its own '# check:' lines, or NAME.check next to it; emips.check)
        self.checks_file: Path | None = None
        self.doc_name = ""
        #: the editor text of the last successful assembly (auto-assemble)
        self._assembled_text: str | None = None
        self._dialogs: list[QtWidgets.QWidget] = []
        self.guide: GuideView | None = None     # Help > User Guide, made when first asked
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        #: Run > GDB Server (emips.gdbstub.GdbServer), polled by gdb_timer
        self.gdb = None
        self.gdb_timer = QtCore.QTimer(self)
        self.gdb_timer.setInterval(20)
        self.gdb_timer.timeout.connect(self._gdb_tick)
        self._gdb_ticks = 0
        self.resize(1500, 950)
        self._build_widgets()
        self._build_docks()
        self.docks["trace"].visibilityChanged.connect(      # regenerates only when shown
            lambda shown: shown and self.trace.refresh(self.session))
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_status()
        self._restored = self._load_settings()
        self._update_title()
        if path:
            self.open_file(path)
        else:
            self.refresh()

    # ================================================================ build

    def _build_widgets(self) -> None:
        self.editor = CodeEditor()
        self.editor.breakpoint_clicked.connect(self.toggle_breakpoint_line)
        self.editor.document().modificationChanged.connect(self.setWindowModified)
        self.findbar = FindBar(self.editor)
        self.legend = EditorLegend()
        left = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(left)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(self.editor)
        box.addWidget(self.findbar)
        box.addWidget(self.legend)
        self.setCentralWidget(left)
        self.datapath = DatapathView()
        self.datapath.part_clicked.connect(self.show_part)
        self.part_dialog: PartDialog | None = None
        self.datapath24k = Datapath24KView()
        self.datapath24k.part_clicked.connect(self.show_part)
        #: the Datapath pane: the 5-stage datapath, or the 24K's pipeline
        self.datapath_pane = QtWidgets.QStackedWidget()
        self.datapath_pane.addWidget(self.datapath)
        self.datapath_pane.addWidget(self.datapath24k)
        self.core8700 = Core8700View()
        self.datapath_pane.addWidget(self.core8700)
        self.counters = CounterView()
        self.trace = TraceView()
        self.board = BoardView()
        self.board.nmi_requested.connect(self.board_nmi)
        self.board.save_disk_requested.connect(self.save_disk)
        self.system = SystemView()
        self.system.core_chosen.connect(lambda cpu: self.core.setCurrentIndex(cpu))
        self.mtview = MTView()
        self.mtview.vp_chosen.connect(lambda cpu: self.core.setCurrentIndex(cpu))
        self.mtview.inputs_changed.connect(self.refresh)
        self.checks = CheckView()
        self.checks.check_requested.connect(self.run_checks)
        self.checks.choose_requested.connect(self.choose_checks_file)
        self.checks.clear_requested.connect(self.clear_checks_file)
        self.checks.line_chosen.connect(self._check_line)
        self.checks.finished.connect(self._checks_finished)
        self.chart = PipelineChart()
        self.disasm = DisassemblyView()
        self.details = InstrPanel()
        self.registers = RegisterView()
        self.registers.dint_requested.connect(self.debug_interrupt)
        self.registers.rnmi_requested.connect(self.request_rnmi)
        self.registers.probe_changed.connect(self.set_probe)
        self.console = ConsoleView()
        self.console.submit_line.connect(self.give_input)
        self.console.submit_keys.connect(self.type_keys)
        self.memory = MemoryView()
        self.caches = CacheView()
        self.tlb = TLBView()
        self.source = SourceView()
        self.source.breakpoints_changed.connect(self.refresh)
        self.messages = QtWidgets.QPlainTextEdit()
        self.messages.setReadOnly(True)
        self.reference = ReferenceView()
        self.reference.insert_requested.connect(self.insert_syntax)
        self.encoder = EncoderView()
        self.converter = ConverterView()

    def _build_docks(self) -> None:
        self.setDockOptions(QtWidgets.QMainWindow.DockOption.AnimatedDocks
                            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
                            | QtWidgets.QMainWindow.DockOption.AllowNestedDocks)
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas,
                            QtWidgets.QTabWidget.TabPosition.North)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        F = QtWidgets.QDockWidget.DockWidgetFeature
        self.docks: dict[str, QtWidgets.QDockWidget] = {}
        for key, title, widget in (
                ("datapath", tr("Datapath"), self.datapath_pane),
                ("pipeline", tr("Pipeline"), self.chart),
                ("disassembly", tr("Disassembly"), self.disasm),
                ("instruction", tr("Instruction"), self.details),
                ("source", tr("Source"), self.source),
                ("registers", tr("Registers"), self.registers),
                ("console", tr("Console"), self.console),
                ("memory", tr("Memory"), self.memory),
                ("caches", tr("Caches"), self.caches),
                ("tlb", tr("TLB"), self.tlb),
                ("counters", tr("Counters"), self.counters),
                ("trace", tr("Trace"), self.trace),
                ("board", tr("Board"), self.board),
                ("system", tr("System"), self.system),
                ("mt", tr("MT"), self.mtview),
                ("checks", tr("Checks"), self.checks),
                ("messages", tr("Messages"), self.messages),
                ("reference", tr("Instruction Reference"), self.reference),
                ("encoder", tr("Encoder / Decoder"), self.encoder),
                ("converter", tr("Number Converter"), self.converter)):
            d = QtWidgets.QDockWidget(title, self)
            d.setObjectName(f"dock_{key}")
            d.setWidget(widget)
            d.setFeatures(F.DockWidgetMovable | F.DockWidgetFloatable | F.DockWidgetClosable)
            d.topLevelChanged.connect(self._dock_floated)
            self.docks[key] = d
        self._arrange_docks()

    def _arrange_docks(self) -> None:
        """The default layout: views and registers on the right of the
        editor, console / memory / messages along the bottom."""
        d = self.docks
        right, bottom = Qt.DockWidgetArea.RightDockWidgetArea, Qt.DockWidgetArea.BottomDockWidgetArea
        for dock in d.values():
            dock.setFloating(False)
            self.removeDockWidget(dock)
        self.addDockWidget(right, d["datapath"])
        self.splitDockWidget(d["datapath"], d["registers"], Qt.Orientation.Horizontal)
        for k in ("pipeline", "disassembly", "instruction", "source"):
            self.tabifyDockWidget(d["datapath"], d[k])
        self.addDockWidget(bottom, d["console"])
        for k in ("memory", "caches", "tlb", "counters", "trace", "board", "system", "mt",
                  "checks", "messages"):
            self.tabifyDockWidget(d["console"], d[k])
        for k in TOOL_DOCKS:                    # Tools: next to the views, hidden
            self.tabifyDockWidget(d["datapath"], d[k])
        for key, dock in d.items():
            dock.setVisible(key not in TOOL_DOCKS)
        d["datapath"].raise_()
        d["console"].raise_()

    def _default_sizes(self) -> None:
        d = self.docks
        w = self.width()
        self.resizeDocks([d["datapath"], d["registers"]], [int(w * 0.55), int(w * 0.2)],
                         Qt.Orientation.Horizontal)
        self.resizeDocks([d["console"]], [int(self.height() * 0.27)], Qt.Orientation.Vertical)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._restored:
            self._restored = True
            QtCore.QTimer.singleShot(0, self._default_sizes)

    def reset_layout(self) -> None:
        self._arrange_docks()
        QtCore.QTimer.singleShot(0, self._default_sizes)

    # -- undocked panes are windows ------------------------------------------------

    #: the decorations of an undocked pane: the platform's title bar with
    #: minimize / maximize / close and a resizable frame
    FLOATING_FLAGS = (Qt.WindowType.Window | Qt.WindowType.CustomizeWindowHint
                      | Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowSystemMenuHint
                      | Qt.WindowType.WindowMinMaxButtonsHint
                      | Qt.WindowType.WindowCloseButtonHint)

    def _dock_floated(self, floating: bool) -> None:
        dock = self.sender()
        if not isinstance(dock, QtWidgets.QDockWidget):
            return
        if floating:
            self._make_window(dock)
        else:                                   # docked again: Qt's own title bar
            dock.setTitleBarWidget(None)

    def _make_window(self, dock: QtWidgets.QDockWidget) -> None:
        """Give an undocked pane real window decorations (Qt draws its own
        frameless title on X11) and a bar with a Dock button.  Waits for
        the end of a drag, so a pane can still be dragged into another
        dock area."""
        if not _alive(self) or not _alive(dock) or not dock.isFloating():
            return
        if QtWidgets.QApplication.mouseButtons() & Qt.MouseButton.LeftButton:
            QtCore.QTimer.singleShot(150, lambda: self._make_window(dock))
            return
        if dock.windowFlags() & Qt.WindowType.WindowMinMaxButtonsHint == \
                Qt.WindowType.WindowMinMaxButtonsHint:
            return                              # already a window
        geo = dock.geometry()
        dock.setTitleBarWidget(DockBar(dock))
        dock.setWindowFlags(self.FLOATING_FLAGS)
        dock.setGeometry(geo)
        dock.show()

    # -- actions -------------------------------------------------------------

    def _action(self, key: str, text: str, slot=None, shortcut=None, tip: str = "",
                checkable: bool = False) -> QtGui.QAction:
        a = QtGui.QAction(text, self)
        if isinstance(shortcut, QtGui.QKeySequence.StandardKey):
            a.setShortcuts(shortcut)            # every platform binding
        elif shortcut:
            keys = shortcut if isinstance(shortcut, (list, tuple)) else [shortcut]
            a.setShortcuts([QtGui.QKeySequence(k) for k in keys])
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        a.setCheckable(checkable)
        if slot is not None:
            (a.toggled if checkable else a.triggered).connect(lambda *_: slot())
        self.addAction(a)                       # shortcuts work with the menu hidden
        self.actions[key] = a
        return a

    def _build_actions(self) -> None:
        self.actions: dict[str, QtGui.QAction] = {}
        A = self._action
        Std = QtGui.QKeySequence.StandardKey
        # File
        A("new", tr("&New"), self.new_file, Std.New)
        A("new_c", tr("New &C Program"), self.new_c_file, "Ctrl+Shift+N",
          tr("A C program: Build compiles it for the selected release (needs clang or "
             "a MIPS GCC)"))
        A("open", tr("&Open…"), self.open_dialog, Std.Open,
          tr("Open an assembly source (.asm, .s) or an Intel HEX image (.hex)"))
        A("open_hex", tr("Open Intel &HEX…"), self.open_hex_dialog, None,
          tr("Load machine code from an Intel HEX file and show its disassembly"))
        A("save", tr("&Save"), self.save, Std.Save)
        A("save_as", tr("Save &As…"), self.save_as, Std.SaveAs)
        A("export_hex", tr("Intel HEX…"), self.export_hex)
        A("export_listing", tr("Listing…"), self.export_listing)
        A("quit", tr("&Quit"), self.close, Std.Quit)
        # Edit
        for key, text, op, std in (("undo", tr("&Undo"), "undo", Std.Undo),
                                   ("redo", tr("&Redo"), "redo", Std.Redo),
                                   ("cut", tr("Cu&t"), "cut", Std.Cut),
                                   ("copy", tr("&Copy"), "copy", Std.Copy),
                                   ("paste", tr("&Paste"), "paste", Std.Paste),
                                   ("select_all", tr("Select &All"), "selectAll",
                                    Std.SelectAll)):
            A(key, text, lambda op=op: self._edit(op), std)
        A("find", tr("&Find…"), lambda: self.findbar.open(False), Std.Find)
        A("replace", tr("&Replace…"), lambda: self.findbar.open(True), "Ctrl+H")
        A("find_next", tr("Find &Next"), self.findbar.find_next, "F3")
        A("find_previous", tr("Find Pre&vious"), self.findbar.find_previous, "Shift+F3")
        A("goto", tr("&Go to Line…"), self.go_to_line, "Ctrl+G")
        A("comment", tr("Toggle &Comment"), self.editor.toggle_comment, "Ctrl+/",
          tr("Comment / uncomment the selected lines with #"))
        A("indent", tr("&Indent"), self.editor.indent, "Ctrl+]")
        A("unindent", tr("U&nindent"), self.editor.unindent, "Ctrl+[")
        A("breakpoint", tr("Toggle &Breakpoint"), self.toggle_breakpoint_here, "F9")
        A("clear_breakpoints", tr("Clear All Breakpoints"), self.clear_breakpoints)
        # Run
        A("assemble", tr("&Assemble"), self.assemble, "Ctrl+B",
          tr("Assemble the source (or reload the HEX image) and reset"))
        A("run", tr("&Run ▶▶"), self.run, "F5", tr("Run until a breakpoint, the end, "
                                                  "an error or an input request"))
        A("pause", tr("&Pause ❚❚"), self.pause, "F6")
        A("step", tr("&Step ▶"), self.step, "F10", tr("One instruction / one clock cycle"))
        A("step_back", tr("Step &Back ◀"), self.step_back, "Shift+F10", tr("Undo one step"))
        A("step_line", tr("Step &Line"), self.step_line, "F11",
          tr("Run to the next source line (a C line of a compiled program), into calls"))
        A("next_line", tr("&Next Line"), self.next_line, "Ctrl+F11",
          tr("Run to the next source line; a call runs as one step"))
        A("finish", tr("&Finish"), self.finish, "Shift+F11",
          tr("Run until the current function returns (the call stack)"))
        A("run_to_cursor", tr("Run to &Cursor"), self.run_to_cursor, "Ctrl+F10",
          tr("Run until the line the cursor is on"))
        A("reset", tr("Rese&t"), self.reset, "Ctrl+R", tr("Restart the program"))
        A("check", tr("&Check"), self.run_checks, "F8",
          tr("Run the program as its checks say — its '# check:' lines, the checks file "
             "named like it or the one chosen — and show which pass (the Checks pane)"))
        A("checks_file", tr("Checks &File…"), self.choose_checks_file, None,
          tr("Check the program against a checks file (a teacher's): it replaces the "
             "program's own checks"))
        A("dint", tr("&Debug Interrupt (DINT)"), self.debug_interrupt, None,
          tr("Request an EJTAG debug interrupt: the core enters Debug Mode at the debug "
             "vector before its next instruction"))
        A("gdb_server", tr("&GDB Server"), self.toggle_gdb_server, None,
          tr("Let GDB debug the program here: in a terminal, gdb-multiarch -ex 'target "
             "remote :1234' (nanoMIPS: nanomips-elf-gdb). The views follow what GDB does"),
          checkable=True)
        self.mode_group = QtGui.QActionGroup(self)
        for i, (key, text) in enumerate((("mode_instruction", tr("&Instruction")),
                                         ("mode_pipeline", tr("5-stage &pipeline")),
                                         ("mode_24k", tr("MIPS &24K pipeline")),
                                         ("mode_p8700", tr("MIPS P&8700 pipeline")))):
            a = A(key, text, checkable=True)
            a.triggered.connect(lambda _=False, i=i: self.mode.setCurrentIndex(i))
            self.mode_group.addAction(a)
        A("forwarding", tr("&Forwarding"), None, None,
          tr("Forwarding (bypassing) paths in the pipeline"), checkable=True)
        self.actions["forwarding"].toggled.connect(lambda on: self.fwd.setChecked(on))
        # View
        self.abi_group = QtGui.QActionGroup(self)
        for key, text, on in (("abi", tr("ABI names ($t0)"), True),
                              ("numeric", tr("Numbers ($8)"), False)):
            a = A(f"names_{key}", text, checkable=True)
            a.triggered.connect(lambda _=False, on=on: self.set_abi(on))
            self.abi_group.addAction(a)
        self.theme_group = QtGui.QActionGroup(self)
        for key, text in (("system", tr("System")), ("light", tr("Light")),
                          ("dark", tr("Dark"))):
            a = A(f"theme_{key}", text, checkable=True)
            a.triggered.connect(lambda _=False, key=key: self.set_theme(key))
            self.theme_group.addAction(a)
        A("zoom_in", tr("Zoom &In"), lambda: self.zoom(+1), ["Ctrl++", "Ctrl+="])
        A("zoom_out", tr("Zoom &Out"), lambda: self.zoom(-1), "Ctrl+-")
        A("zoom_reset", tr("&Reset Zoom"), lambda: self.zoom(0), "Ctrl+0")
        A("reset_layout", tr("Reset &Layout"), self.reset_layout)
        A("part_numbers", tr("Datapath Part &Numbers"), self._part_numbers, None,
          tr("Number the datapath's parts as the guide does (click a part: what "
             "happens inside)"), checkable=True)
        # Configure
        A("settings", tr("&Settings…"), self.open_settings, "Ctrl+,",
          tr("Processor, assembler (strict mode), boot state, exceptions, console, run"))
        self.release_group = QtGui.QActionGroup(self)
        for arch, r in _ARCH_RELEASES:   # (MIPS64: PLAN M22; nanoMIPS: M23, R6 only;
            #                                RISC-V: M25)
            a = A(_release_action(arch, r), _arch_text(arch, r)
                  + (" (Release 6)" if r == "R6" else ""), checkable=True)
            a.triggered.connect(lambda _=False, v=f"{arch}:{r}":
                                self._machine_choice("release", v))
            self.release_group.addAction(a)
        self.endian_group = QtGui.QActionGroup(self)
        for e, text in (("big", tr("Big-endian")), ("little", tr("Little-endian"))):
            a = A(f"endian_{e}", text, checkable=True)
            a.triggered.connect(lambda _=False, e=e: self.set_endian(e))
            self.endian_group.addAction(a)
        self.lang_group = QtGui.QActionGroup(self)
        for code, name in i18n.available().items():
            a = A(f"lang_{code}", name, checkable=True)
            a.triggered.connect(lambda _=False, code=code: self.set_language(code))
            self.lang_group.addAction(a)
        # Help
        A("help_guide", tr("&User Guide"), self.show_guide, Std.HelpContents)
        A("help_features", tr("&What's in eMIPS"),
          lambda: self.show_guide("features.html"),
          None, tr("Every feature: where it is in the window, its command, the guide "
                   "section and the labs"))
        A("help_quick", tr("&Quick Start"), lambda: self.show_help("quick"))
        A("help_keys", tr("&Keyboard Shortcuts"), lambda: self.show_help("keys"))
        A("help_syscalls", tr("&System Calls"), lambda: self.show_help("syscalls"))
        A("help_registers", tr("&Registers"), lambda: self.show_help("registers"))
        A("about", tr("&About eMIPS"), lambda: self.show_help("about"))
        A("about_qt", tr("About &Qt"), QtWidgets.QApplication.aboutQt)
        ed = self.editor
        ed.undoAvailable.connect(self.actions["undo"].setEnabled)
        ed.redoAvailable.connect(self.actions["redo"].setEnabled)
        self.actions["undo"].setEnabled(False)
        self.actions["redo"].setEnabled(False)

    def _build_menus(self) -> None:
        mb, a = self.menuBar(), self.actions
        m = mb.addMenu(tr("&File"))
        m.addActions([a["new"], a["new_c"], a["open"], a["open_hex"]])
        self.recent_menu = m.addMenu(tr("Open &Recent"))
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        ex = m.addMenu(tr("E&xamples"))
        for e in EXAMPLES:
            act = ex.addAction(tr(e.title))
            act.setToolTip(e.file)
            act.triggered.connect(lambda _=False, e=e: self.load_example(e))
        self.labs_menu = m.addMenu(tr("&Labs"))
        self._fill_labs(self.labs_menu)
        m.addSeparator()
        m.addActions([a["save"], a["save_as"]])
        exp = m.addMenu(tr("&Export"))
        exp.addActions([a["export_hex"], a["export_listing"]])
        m.addSeparator()
        m.addAction(a["quit"])

        m = mb.addMenu(tr("&Edit"))
        m.addActions([a["undo"], a["redo"]])
        m.addSeparator()
        m.addActions([a["cut"], a["copy"], a["paste"], a["select_all"]])
        m.addSeparator()
        m.addActions([a["find"], a["replace"], a["find_next"], a["find_previous"], a["goto"]])
        m.addSeparator()
        m.addActions([a["comment"], a["indent"], a["unindent"]])
        m.addSeparator()
        m.addActions([a["breakpoint"], a["clear_breakpoints"]])

        m = mb.addMenu(tr("&Run"))
        m.addActions([a["assemble"], a["run"], a["pause"], a["step"], a["step_back"],
                      a["step_line"], a["next_line"], a["finish"],
                      a["run_to_cursor"], a["reset"]])
        m.addSeparator()
        m.addActions([a["check"], a["checks_file"]])
        m.addSeparator()
        m.addAction(a["dint"])
        m.addAction(a["gdb_server"])
        m.addSeparator()
        mode = m.addMenu(tr("&Mode"))
        mode.addActions(self.mode_group.actions())
        m.addAction(a["forwarding"])

        m = mb.addMenu(tr("&View"))
        for key, dock in self.docks.items():
            if key not in TOOL_DOCKS:
                m.addAction(dock.toggleViewAction())
        m.addSeparator()
        names = m.addMenu(tr("Register &Names"))
        names.addActions(self.abi_group.actions())
        m.addSeparator()
        m.addActions([a["zoom_in"], a["zoom_out"], a["zoom_reset"]])
        m.addSeparator()
        m.addAction(a["part_numbers"])
        m.addAction(a["reset_layout"])

        m = mb.addMenu(tr("&Tools"))
        for key, keys in (("reference", "Ctrl+Shift+I"), ("encoder", "Ctrl+Shift+E"),
                          ("converter", "Ctrl+Shift+N")):
            act = self.docks[key].toggleViewAction()
            act.setShortcut(QtGui.QKeySequence(keys))
            act.triggered.connect(lambda on, key=key: on and self.show_tool(key))
            self.actions[f"tool_{key}"] = act
            m.addAction(act)

        m = mb.addMenu(tr("&Configure"))
        m.addAction(a["settings"])
        m.addSeparator()
        rel = m.addMenu(tr("&Architecture and Release"))
        for i, act in enumerate(self.release_group.actions()):
            if i == len(RELEASES):
                rel.addSeparator()              # MIPS32 | MIPS64
            rel.addAction(act)
        end = m.addMenu(tr("&Byte Order"))
        end.addActions(self.endian_group.actions())
        m.addSeparator()
        theme = m.addMenu(tr("&Theme"))
        theme.addActions(self.theme_group.actions())
        lang = m.addMenu(tr("&Language"))
        lang.addActions(self.lang_group.actions())

        m = mb.addMenu(tr("&Help"))
        m.addActions([a["help_guide"], a["help_features"], a["help_quick"], a["help_keys"],
                      a["help_syscalls"], a["help_registers"]])
        m.addSeparator()
        m.addActions([a["about"], a["about_qt"]])

    def _build_toolbar(self) -> None:
        tb = self.addToolBar(tr("Run controls"))
        tb.setObjectName("toolbar_run")
        a = self.actions
        tb.addActions([a["assemble"], a["reset"]])
        tb.addSeparator()
        tb.addActions([a["step_back"], a["step"], a["step_line"], a["run"], a["pause"]])
        tb.addSeparator()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems([tr("Instruction"), tr("5-stage pipeline"), tr("24K pipeline"),
                            tr("P8700 pipeline")])
        self.mode.setToolTip(tr("Instruction: one instruction per step. 5-stage pipeline: "
                                "one clock cycle per step, the textbook pipeline. 24K "
                                "pipeline: the MIPS 24K's 8 stages, branch prediction."))
        self.mode.currentIndexChanged.connect(self._mode_changed)
        tb.addWidget(self.mode)
        self.fwd = QtWidgets.QCheckBox(tr("forwarding"))
        self.fwd.setChecked(True)
        self.fwd.toggled.connect(self._mode_changed)
        tb.addWidget(self.fwd)
        self.core = QtWidgets.QComboBox()
        self.core.setToolTip(tr("The core the views show (Settings > Processor > Cores); "
                                "every running core steps together"))
        self.core.currentIndexChanged.connect(self._core_changed)
        self.core_action = tb.addWidget(self.core)
        self.core_action.setVisible(False)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" " + tr("speed") + " "))
        self._build_machine_bar()
        self.speed = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(0, len(SPEEDS) - 1)
        self.speed.setValue(2)
        self.speed.setFixedWidth(120)
        self.speed.valueChanged.connect(self._speed_changed)
        tb.addWidget(self.speed)

    def _sync_machine_bar(self, mode_index: int) -> None:
        """The toolbar's mode and machine choices as the session has them;
        values that do not go with the rest disabled, the reason as a tooltip."""
        from ..compat import forbidden, program_problem
        s = self.session
        self._syncing = True
        try:
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(mode_index)
            why = forbidden("model", "24k", s.options, s.release)   # (a microMIPS core)
            self.mode.model().item(2).setEnabled(why is None or mode_index == 2)
            self.mode.setItemData(2, why or "", Qt.ItemDataRole.ToolTipRole)
            why = forbidden("model", "p8700", s.options, s.release)  # (the P8700's own)
            self.mode.model().item(3).setEnabled(why is None or mode_index == 3)
            self.mode.setItemData(3, why or "", Qt.ItemDataRole.ToolTipRole)
            self.mode.blockSignals(False)
            self.fwd.setEnabled(mode_index < 2)     # the 24K / P8700 always bypass
            self.fwd.setToolTip(tr("Forwarding paths of the 5-stage pipeline") if mode_index < 2
                                else tr("This pipeline always has its bypasses"))
            from .. import cores
            core = cores.get(s.options.core)
            self.m_core.setCurrentIndex(max(0, self.m_core.findData(s.options.core)))
            self.m_release.setCurrentIndex(max(0, self.m_release.findData(
                _arch_key(s.options.arch, s.release))))
            for i in range(self.m_release.count()):
                arch, rel = self.m_release.itemData(i).split(":")
                if rel == "RV":                 # (RISC-V: the session's release stays)
                    rel = s.release
                why = forbidden("release", rel, replace(s.options, arch=arch), s.release)
                self.m_release.model().item(i).setEnabled(why is None)
                self.m_release.setItemData(i, why or "", Qt.ItemDataRole.ToolTipRole)
            self.m_board.setCurrentIndex(max(0, self.m_board.findData(
                s.options.board.split(":")[0])))
            base = s.options.mmu.split("+")[0].split(":")[0]
            self.m_mmu.setCurrentIndex(max(0, self.m_mmu.findData(base)))
            for i in range(self.m_mmu.count()):
                kind = self.m_mmu.itemData(i)
                why = forbidden("mmu", kind, s.options, s.release) if core is None else \
                    None if any(m.split(":")[0].split("+")[0] == kind for m in core.mmu) else \
                    cores.conflict(replace(s.options, mmu=kind), s.release, "mmu")
                self.m_mmu.model().item(i).setEnabled(why is None)
                self.m_mmu.setItemData(i, why or "", Qt.ItemDataRole.ToolTipRole)
            on = s.options.icache != "none" or s.options.dcache != "none"
            other = replace(s.options, icache="none" if on else "32K",
                            dcache="none" if on else "32K")
            why = (cores.conflict(other, s.release, "icache")
                   or cores.conflict(other, s.release, "dcache"))
            self.m_caches.setEnabled(why is None)
            self.m_caches.setToolTip(why or tr("The 24K's L1 caches: 32 KB instruction and "
                                               "data, 4-way (other shapes, L2 / L3: "
                                               "Settings > Caches)"))
            why = forbidden("cores", 2, s.options, s.release)
            allowed = core.cores if core is not None else range(1, MAX_CORES + 1)
            self.m_cores.setMinimum(min(allowed))
            self.m_cores.setMaximum(1 if why and core is None else max(allowed))
            self.m_cores.setToolTip(why or tr("More than one core: a coherent multi-core "
                                              "system"))
            self.m_cores.setValue(s.options.cores)
            self.m_caches.setChecked(s.options.icache != "none" or s.options.dcache != "none")
            kind = "c" if s.c_file is not None else s.kind
            item = self.m_board.model().item(self.m_board.findData("malta"))
            test = replace(s.options, board="malta")
            problem = program_problem(test, kind) if s.program is not None else None
            item.setEnabled(problem is None)
            self.m_board.setItemData(self.m_board.findData("malta"), problem or "",
                                     Qt.ItemDataRole.ToolTipRole)
        finally:
            self._syncing = False

    def _build_machine_bar(self) -> None:
        """The machine, one click away: release, board, MMU, cores, caches
        (what Settings > Processor / Caches set; only what goes together,
        emips.compat)."""
        self.addToolBarBreak()                  # its own row: always in view
        mb = self.addToolBar(tr("Machine"))
        mb.setObjectName("toolbar_machine")
        from ..cores import CORES
        self.m_core = QtWidgets.QComboBox()
        self.m_core.addItem(tr("Generic core"), "")
        for core in CORES.values():
            self.m_core.addItem(core.title, core.name)
        self.m_core.setToolTip(tr("A real MIPS core: the release, compressed ISA, FPU, MMU, "
                                  "caches and pipeline it is built with (the others are "
                                  "greyed out); Generic core: any setting"))
        self.m_release = QtWidgets.QComboBox()
        for arch, r in _ARCH_RELEASES:
            self.m_release.addItem(_arch_text(arch, r), f"{arch}:{r}")
        self.m_release.setToolTip(tr("The architecture and release: MIPS32, MIPS64 (64-bit "
                                     "registers, the doubleword instructions), nanoMIPS "
                                     "(the I7200's 16 / 32 / 48-bit instructions) or "
                                     "RISC-V (RV32 / RV64: the extensions and privilege "
                                     "modes in Settings > Processor); its instructions "
                                     "and its control registers"))
        self.m_board = QtWidgets.QComboBox()
        self.m_board.addItem(tr("Flat memory"), "flat")
        self.m_board.addItem(tr("Malta board"), "malta")
        self.m_board.setToolTip(tr("Flat: every address is RAM (the MARS layout). Malta: a "
                                   "real board — its memory map, bus errors, devices"))
        self.m_mmu = QtWidgets.QComboBox()
        for spec, text in (("fixed", tr("Fixed mapping")), ("bat", tr("BAT")),
                           ("tlb", tr("TLB (24K)")), ("ftlb", tr("VTLB + FTLB")),
                           ("mpu", tr("MPU (I7200)"))):
            self.m_mmu.addItem(text, spec)
        self.m_mmu.setToolTip(tr("The MMU (more choices in Settings > Processor)"))
        self.m_cores = QtWidgets.QSpinBox()
        self.m_cores.setRange(1, MAX_CORES)
        self.m_cores.setSuffix(" " + tr("core(s)"))
        self.m_cores.setToolTip(tr("More than one core: a coherent multi-core system"))
        self.m_caches = QtWidgets.QCheckBox(tr("caches"))
        self.m_caches.setToolTip(tr("The 24K's L1 caches: 32 KB instruction and data, 4-way "
                                    "(other shapes, L2 / L3: Settings > Caches)"))
        for w in (self.m_core, self.m_release, self.m_board, self.m_mmu, self.m_cores,
                  self.m_caches):
            mb.addWidget(w)
        self.m_core.currentIndexChanged.connect(
            lambda _i: self._machine_choice("core", self.m_core.currentData()))
        self.m_release.currentIndexChanged.connect(
            lambda _i: self._machine_choice("release", self.m_release.currentData()))
        self.m_board.currentIndexChanged.connect(
            lambda _i: self._machine_choice("board", self.m_board.currentData()))
        self.m_mmu.currentIndexChanged.connect(
            lambda _i: self._machine_choice("mmu", self.m_mmu.currentData()))
        self.m_cores.valueChanged.connect(lambda v: self._machine_choice("cores", v))
        self.m_caches.toggled.connect(lambda on: self._machine_choice("caches", on))

    def _machine_choice(self, field: str, value) -> None:
        """A choice on the Machine toolbar: what depends on it adjusts
        (and says so); a choice the loaded program cannot run with is
        refused."""
        from .. import cores
        from ..compat import adjust, program_problem
        if getattr(self, "_syncing", False):
            return
        s = self.session
        release = s.release
        o = replace(s.options, latency=dict(s.options.latency),
                    mem_timing=dict(s.options.mem_timing))
        if field == "release":                  # "mips64:R5": the architecture too
            if not value:
                return
            o.arch, _, release = value.rpartition(":") if ":" in value else (o.arch, "", value)
            if release == "RV":                 # (RISC-V: no release; the session's stays)
                release = s.release
            if s.options.arch == "nanomips" and o.arch != "nanomips" and \
                    getattr(cores.get(o.core), "arch", "nanomips") == "nanomips":
                o.fpu = True                    # (leaving nanoMIPS: the FPU it took;
                #                                  a nanoMIPS core preset goes too)
        core = cores.get(o.core)
        if field == "core":
            o, core_release = cores.apply(o, value)
            release = core_release or release
            core = cores.get(o.core)
            if core is not None:
                self._log(tr("core: {lines}").format(lines="\n  ".join(cores.describe(core))))
        elif field == "board":
            o.board = value
        elif field == "mmu":                    # the core's own spelling of that MMU
            o.mmu = next((m for m in core.mmu if m.split(":")[0].split("+")[0] == value),
                         value) if core is not None else value
        elif field == "cores":
            o.cores = int(value)
        elif field == "caches":                 # a core: its own shapes
            if core is not None and value:
                o.icache, o.dcache = core.build["icache"], core.build["dcache"]
            else:
                o.icache = o.dcache = "32K" if value else "none"
        kind = "c" if s.c_file is not None else s.kind
        problem = program_problem(o, kind, s.program) if s.program is not None else None
        if problem:
            self.status.showMessage(problem, 10000)
            self._log(problem)
            self._sync_actions()                # back to what it was
            return
        o, notes = adjust(o, release)
        for note in notes:
            self._log(tr("adjusted: {note}").format(note=note))
        if notes:
            self.status.showMessage(tr("adjusted: {note}").format(note=notes[0]), 8000)
        self.apply_settings(release, "little" if o.arch in ("rv32", "rv64") else s.endian,
                            s.forwarding, o.validated())

    def _build_status(self) -> None:
        self.status = self.statusBar()
        self.state_label = QtWidgets.QLabel()
        self.config_label = QtWidgets.QLabel()
        self.gdb_label = QtWidgets.QLabel()
        self.gdb_label.hide()
        self.status.addPermanentWidget(self.gdb_label)
        self.status.addPermanentWidget(self.state_label)
        self.status.addPermanentWidget(self.config_label)

    # ============================================================= settings

    def _get(self, key: str, default: str) -> str:
        v = self.settings.value(key)
        return default if v is None else str(v)

    def _put(self, key: str, value) -> None:
        self.settings.setValue(key, value)

    def _load_settings(self) -> bool:
        """Apply the remembered options; True if a window layout was
        restored."""
        s = self.session
        rel = self._get("release", "R5")
        s.release = rel if rel in RELEASES else "R5"
        s.endian = "little" if self._get("endian", "big") == "little" else "big"
        s.abi = self._get("abi", "1") == "1"
        s.mode = "pipeline" if self._get("mode", "instruction") == "pipeline" else "instruction"
        s.forwarding = self._get("forwarding", "1") == "1"
        s.options = Options.from_strings(
            {k: v for k in Options().to_strings()
             if (v := self.settings.value(f"options/{k}")) is not None})
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(_mode_index(s))
        self.mode.blockSignals(False)
        self.fwd.blockSignals(True)
        self.fwd.setChecked(s.forwarding)
        self.fwd.blockSignals(False)
        try:
            speed = int(self._get("speed", "2"))
        except ValueError:
            speed = 2
        self.speed.blockSignals(True)
        self.speed.setValue(max(0, min(speed, len(SPEEDS) - 1)))
        self.speed.blockSignals(False)
        self.actions["part_numbers"].setChecked(self._get("part_numbers", "0") == "1")
        try:
            size = int(self._get("font_size", str(DEFAULT_FONT)))
        except ValueError:
            size = DEFAULT_FONT
        self.editor.set_point_size(size)
        self.legend.set_mode(s.mode)
        name = self._get("theme", "system")
        self.set_theme(name if name in ("system", "light", "dark") else "system",
                       save=False)
        self._sync_actions()
        restored = False
        geo, state = self.settings.value("geometry"), self.settings.value("window_state")
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            restored = self.restoreState(state, 1)
            for dock in self.docks.values():     # panes left undocked last time
                if dock.isFloating():
                    self._make_window(dock)
        return restored

    def _sync_actions(self) -> None:
        s, a = self.session, self.actions
        mode_index = _mode_index(s)
        a[("mode_instruction", "mode_pipeline", "mode_24k", "mode_p8700")[mode_index]] \
            .setChecked(True)
        self._sync_machine_bar(mode_index)
        a["forwarding"].blockSignals(True)
        a["forwarding"].setChecked(s.forwarding)
        a["forwarding"].blockSignals(False)
        a[_release_action(s.options.arch, s.release)].setChecked(True)
        c = getattr(self, "lang", "asm") == "c"
        a["assemble"].setText(tr("&Build") if c else tr("&Assemble"))
        a["assemble"].setToolTip(tr("Compile and link the C program") if c else
                                 tr("Assemble the program"))
        a[f"endian_{s.endian}"].setChecked(True)
        for e in ("big", "little"):             # (RISC-V: little-endian only)
            a[f"endian_{e}"].setEnabled(not s.rv)
        a["names_abi" if s.abi else "names_numeric"].setChecked(True)
        lang = self._get("language", "en")
        a.get(f"lang_{lang}", a["lang_en"]).setChecked(True)
        self.reference.set_context(s.release, s.abi, s.options.pseudo, s.options.arch)
        self.encoder.set_context(s.release, s.abi, s.options.pseudo, s.options.arch)
        self.converter.set_nan2008(s.release == "R6")
        from ..cores import get as get_core
        core = get_core(s.options.core)
        what = (f"{arch_title(s.options.arch)} {s.options.rv_extensions}" if s.rv
                else f"{arch_title(s.options.arch)} {s.release}")
        self.config_label.setText(f" {core.title + ' · ' if core else ''}{what} · "
                                  + (tr("big-endian") if s.endian == "big"
                                     else tr("little-endian")) + " ")

    def closeEvent(self, event) -> None:
        if not self._confirm_discard(tr("Quit")):
            event.ignore()
            return
        self.pause()
        self._stop_gdb()
        self._put("geometry", self.saveGeometry())
        self._put("window_state", self.saveState(1))
        self.settings.sync()
        for d in self._dialogs:
            if _alive(d):
                d.close()
        if self.guide is not None:
            self.guide.close()
        event.accept()

    # ================================================================= files

    def _update_title(self) -> None:
        name = self.doc_name or (tr("untitled") if self.editor.toPlainText() else "")
        self.setWindowTitle(f"eMIPS — {name}[*]" if name else "eMIPS[*]")
        hexmode = self.hex_text is not None
        self.editor.setReadOnly(hexmode)
        for k in ("save", "comment", "indent", "unindent", "replace"):
            self.actions[k].setEnabled(not hexmode)

    def _confirm_discard(self, title: str) -> bool:
        """Ask before unsaved edits are lost: Save / Discard / Cancel."""
        doc = self.editor.document()
        if self.hex_text is not None or not doc.isModified() \
                or not self.editor.toPlainText().strip():
            return True
        B = QtWidgets.QMessageBox.StandardButton
        answer = QtWidgets.QMessageBox.question(
            self, title, tr("The program has unsaved changes. Save them first?"),
            B.Save | B.Discard | B.Cancel, B.Save)
        if answer == B.Save:
            return self.save()
        return answer == B.Discard

    def _set_text(self, text: str, name: str, path: Path | None, hex_text: str | None,
                  lang: str = "asm") -> None:
        self.pause()
        self.lang = lang
        self.session.breakpoints.clear()        # a new document: its own breakpoints
        self.hex_text = hex_text
        self.path = path
        self.doc_name = name
        self.checks_file = None                 # (chosen for the old document)
        self._show_check_source()
        self.findbar.hide()
        self.editor.setReadOnly(False)
        self.editor.setPlainText(text)
        self.editor.document().setModified(False)
        self._update_title()

    def new_c_file(self) -> None:
        """File > New C Program: a small C program to start from."""
        if not self._confirm_discard(tr("New")):
            return
        self._set_text(C_TEMPLATE, "", None, None, lang="c")
        self.session.unload()
        self.editor.highlight({})
        self._sync_actions()
        self.refresh()

    def new_file(self) -> None:
        if not self._confirm_discard(tr("New")):
            return
        self._set_text("", "", None, None)
        self.session.unload()
        self.editor.highlight({})
        self.refresh()

    def _start_dir(self) -> str:
        return self._get("last_dir", str(Path.home()))

    def open_dialog(self) -> None:
        if not self._confirm_discard(tr("Open")):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Open program"), self._start_dir(),
            tr("Programs") + " (*.asm *.s *.c *.hex *.ihex *.elf);;" + tr("Assembly")
            + " (*.asm *.s);;" + "C (*.c);;" + tr("Assembly")
            + " (*.asm *.s);;" + "Intel HEX (*.hex *.ihex);;" + tr("ELF executables")
            + " (*.elf);;" + tr("All files") + " (*)")
        if path:
            self.open_file(path)

    def open_hex_dialog(self) -> None:
        if not self._confirm_discard(tr("Open")):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Open Intel HEX"), self._start_dir(),
            "Intel HEX (*.hex *.ihex);;" + tr("All files") + " (*)")
        if path:
            self.open_file(path, as_hex=True)

    def open_file(self, path: str, as_hex: bool | None = None, *, header: bool = True) -> bool:
        """Open an assembly or C source, or an Intel HEX image (by suffix
        unless ``as_hex`` says); a source's ``# eMIPS:`` line sets the
        machine up (``header``)."""
        p = Path(path)
        try:
            raw = p.read_bytes()
            text = raw.decode("utf-8", errors="replace")
        except OSError as exc:
            self._fail(tr("cannot open {path}: {error}").format(
                path=path, error=exc.strerror or exc))
            self._forget_recent(str(p))
            self.refresh()
            return False
        self._put("last_dir", str(p.resolve().parent))
        if raw[:4] == b"\x7fELF":
            return self._open_elf(raw, p)
        if as_hex or (as_hex is None and p.suffix.lower() in HEX_SUFFIXES):
            return self._open_hex(text, p)
        self._set_text(text, p.name, p, None, lang="c" if p.suffix.lower() == ".c" else "asm")
        if header:
            from ..profile import read
            try:
                prof = read(text, p.resolve().parent)
            except ValueError as exc:
                self._log(str(exc))
                prof = None
            if prof is not None:
                self.apply_profile(prof)
        self._add_recent(p)
        self.assemble()
        return True

    def _open_hex(self, text: str, p: Path) -> bool:
        s = self.session
        if not s.load_hex(text):
            self._fail(s.errors[0])
            return False
        self._set_text(s.listing, p.name, p, text)
        self._add_recent(p)
        self._log(tr("loaded {name}: {n} instruction words, entry 0x{entry:08x}").format(
            name=p.name, n=len(s.program.source_map), entry=s.program.entry))
        self.refresh()
        return True

    def _open_elf(self, data: bytes, p: Path) -> bool:
        s = self.session
        if not s.load_elf(data, str(p)):
            self._fail(s.errors[0])
            return False
        self._set_text(s.listing, p.name, p, data)
        self._add_recent(p)
        self._sync_actions()
        self.docks["source"].show()
        self.docks["source"].raise_()
        self._log(tr("loaded {name}: an ELF executable, {arch} {release}, {endian}-endian, "
                     "{n} instruction words, entry 0x{entry:08x}").format(
            name=p.name, arch=arch_title(s.options.arch), release=s.release,
            endian=s.endian, n=len(s.program.source_map), entry=s.program.entry))
        self.refresh()
        return True

    def _fill_labs(self, menu: QtWidgets.QMenu) -> None:
        """File > Labs: the guide's labs and the other programs it uses
        (emips.labs), each opened with the settings it is made for."""
        from ..labs import catalogue
        labs, more = catalogue()
        if not labs and not more:
            act = menu.addAction(tr("(the guide's labs are not installed)"))
            act.setEnabled(False)
            return
        def mark(p: Path) -> str:               # a lab with checks: Run → Check
            return tr(" (checks)") if _has_checks(p) else ""
        for lab in labs:
            if len(lab.files) == 1:
                act = menu.addAction(lab.title + mark(lab.files[0]))
                act.setToolTip(lab.files[0].name)
                act.triggered.connect(lambda _=False, p=lab.files[0]: self.open_lab(p))
            else:
                sub_ = menu.addMenu(lab.title + (tr(" (checks)") if any(
                    _has_checks(p) for p in lab.files) else ""))
                for p in lab.files:
                    sub_.addAction(p.name + mark(p)).triggered.connect(
                        lambda _=False, p=p: self.open_lab(p))
        if more:
            menu.addSeparator()
            other = menu.addMenu(tr("Programs from the guide"))
            for group in more:
                sub_ = other.addMenu(group.title)
                for p in group.files:
                    sub_.addAction(p.name + mark(p)).triggered.connect(
                        lambda _=False, p=p: self.open_lab(p))

    def _show_feature_panes(self) -> None:
        """The panes of what the settings switch on come up (step 4 of the
        GUI plan: a feature you turn on is where you can see it)."""
        o = self.session.options
        wanted = []
        if o.icache != "none" or o.dcache != "none" or o.l2 != "none" or o.write_buffer:
            wanted.append("caches")
        if o.mmu.split("+")[0].split(":")[0] in ("tlb", "ftlb", "bat"):
            wanted.append("tlb")
        if o.cores > 1:
            wanted.append("system")
        if self.session.mt is not None:
            wanted.append("mt")
        if o.board.split(":")[0] == "malta":
            wanted.append("board")
        for key in wanted:
            self.docks[key].show()
        if wanted:
            self.docks[wanted[-1]].raise_()
        if self.session.mode == "pipeline":
            self.docks["datapath"].raise_()

    def load_example(self, ex: Example | None = None) -> None:
        """Put an example in the editor (asks first about unsaved edits),
        set the machine up as its ``# eMIPS:`` line says (the defaults
        without one) and assemble it."""
        ex = ex or EXAMPLES[0]
        if not self._confirm_discard(tr("Examples")):
            return
        self._set_text(ex.source(), tr("example: {file}").format(file=ex.file), None, None)
        self.apply_profile(ex.profile)
        self.assemble()

    def open_lab(self, path: Path) -> None:
        """File > Labs: open a lab program with the settings it is made
        for (its ``# eMIPS:`` line; the defaults without one)."""
        from ..profile import Profile, read
        if not self._confirm_discard(tr("Labs")):
            return
        try:
            prof = read(path.read_text(encoding="utf-8"), path.resolve().parent) \
                or Profile(args=[])
        except (OSError, ValueError) as exc:
            self._fail(str(exc))
            return
        self.apply_profile(prof)
        self.open_file(str(path), header=False)

    def apply_profile(self, prof) -> bool:
        """Switch to the settings a program's ``# eMIPS:`` line asks for
        (emips.profile): the release, byte order, mode and the machine
        options it names, the others of those at their defaults; what
        goes together is kept so (emips.compat).  Returns True when
        something changed."""
        from ..compat import adjust
        s = self.session
        if not prof.differs(s.options, s.release, s.endian, s.mode, s.forwarding):
            return False
        options, _notes = adjust(prof.options_for(s.options), prof.release)
        s.release, s.endian, s.forwarding, s.options = (prof.release, prof.endian,
                                                        prof.forwarding, options)
        self._put("release", s.release)
        self._put("endian", s.endian)
        self._put("forwarding", "1" if s.forwarding else "0")
        for k, v in options.to_strings().items():
            self._put(f"options/{k}", v)
        self.fwd.blockSignals(True)
        self.fwd.setChecked(s.forwarding)
        self.fwd.blockSignals(False)
        if (s.mode == "pipeline") != (prof.mode == "pipeline"):
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(0 if prof.mode != "pipeline" else
                                      _mode_index(s, "pipeline"))
            self.mode.blockSignals(False)
            s.set_mode(prof.mode, forwarding=s.forwarding)
            self._put("mode", s.mode)
            self.legend.set_mode(s.mode)
        self._sync_actions()
        text = tr("settings for this program: {flags}").format(flags=prof.describe())
        self._log(text)
        self.status.showMessage(text, 8000)
        self._show_feature_panes()
        return True

    def save(self) -> bool:
        if self.hex_text is not None:
            self.status.showMessage(tr("an Intel HEX image or ELF executable is "
                                       "read-only: use File > Export"), 6000)
            return False
        if self.path is None:
            return self.save_as()
        try:
            self.path.write_text(self.editor.toPlainText(), encoding="utf-8")
        except OSError as exc:
            self._fail(tr("cannot save {path}: {error}").format(
                path=self.path, error=exc.strerror or exc))
            return False
        self.editor.document().setModified(False)
        self.doc_name = self.path.name
        self._update_title()
        self._add_recent(self.path)
        self.status.showMessage(tr("saved {path}").format(path=self.path), 3000)
        return True

    def save_as(self) -> bool:
        if self.hex_text is not None:
            return self.save()
        c = self.lang == "c"
        start = str(self.path) if self.path else str(Path(self._start_dir()) /
                                                     ("program.c" if c else "program.asm"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("Save program"), start,
            "C (*.c)" if c else tr("Assembly") + " (*.asm *.s)")
        if not path:
            return False
        self.path = Path(path)
        self._put("last_dir", str(self.path.resolve().parent))
        return self.save()

    def _export(self, title: str, suffix: str, filt: str, content) -> None:
        if self.session.program is None:
            self.status.showMessage(tr("nothing to export — assemble a program first"), 6000)
            return
        base = self.path.with_suffix(suffix) if self.path else \
            Path(self._start_dir()) / f"program{suffix}"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, str(base), filt)
        if not path:
            return
        try:
            Path(path).write_text(content(), encoding="utf-8")
        except OSError as exc:
            self._fail(tr("cannot save {path}: {error}").format(
                path=path, error=exc.strerror or exc))
            return
        self.status.showMessage(tr("saved {path}").format(path=path), 4000)

    def export_hex(self) -> None:
        self._export(tr("Export Intel HEX"), ".hex", "Intel HEX (*.hex)",
                     lambda: self.session.program.to_ihex())

    def listing_text(self) -> str:
        s = self.session
        lines = [f"# {self.doc_name or 'program'} — {arch_title(s.options.arch)} {s.release}, "
                 f"{s.endian}-endian"]
        src = s.program.source_map
        for a, w, text, line in s.disassembly():
            source = src[a].text.strip() if s.kind == "asm" else ""
            lines.append(f"{a:08x}  {s.program.encoding(a, w)}  {text:<32}"
                         + (f"  # {line}: {source}" if source else ""))
        return "\n".join(lines) + "\n"

    def export_listing(self) -> None:
        self._export(tr("Export listing"), ".lst", tr("Listing") + " (*.lst *.txt)",
                     self.listing_text)

    # -- recent files ------------------------------------------------------------

    def recent_files(self) -> list[str]:
        return [p for p in self._get("recent", "").split("\n") if p]

    def _add_recent(self, p: Path) -> None:
        path = str(p.resolve())
        files = [path] + [f for f in self.recent_files() if f != path]
        self._put("recent", "\n".join(files[:MAX_RECENT]))

    def _forget_recent(self, path: str) -> None:
        full = str(Path(path).resolve())
        self._put("recent", "\n".join(f for f in self.recent_files() if f != full))

    def _fill_recent(self) -> None:
        m = self.recent_menu
        m.clear()
        files = self.recent_files()
        for i, f in enumerate(files, 1):
            act = m.addAction(f"&{i}  {Path(f).name}")
            act.setToolTip(f)
            act.setStatusTip(f)
            act.triggered.connect(lambda _=False, f=f: self.open_recent(f))
        if files:
            m.addSeparator()
            m.addAction(tr("Clear Recent Files"), lambda: self._put("recent", ""))
        else:
            m.addAction(tr("(none)")).setEnabled(False)

    def open_recent(self, path: str) -> None:
        if self._confirm_discard(tr("Open")):
            self.open_file(path)

    # ================================================================== edit

    def _edit(self, op: str) -> None:
        """Undo / copy / paste ... act on the focused text widget (editor,
        console input, memory address ...)."""
        w = QtWidgets.QApplication.focusWidget() or self.editor
        if not hasattr(w, op):
            w = self.editor
        getattr(w, op)()

    def show_tool(self, key: str) -> None:
        dock = self.docks[key]
        dock.show()
        dock.raise_()
        widget = {"reference": self.reference.search, "encoder": self.encoder.input,
                  "converter": self.converter.input}[key]
        widget.setFocus()

    def insert_syntax(self, text: str) -> None:
        """Instruction reference > Insert: the syntax at the cursor."""
        if self.editor.isReadOnly():
            self.status.showMessage(tr("the editor is read-only (Intel HEX)"), 4000)
            return
        self.editor.insertPlainText(text)
        self.editor.setFocus()

    def go_to_line(self) -> None:
        line, ok = QtWidgets.QInputDialog.getInt(
            self, tr("Go to Line"), tr("Line (1 - {n}):").format(n=self.editor.blockCount()),
            self.editor.current_line(), 1, max(1, self.editor.blockCount()))
        if ok:
            self.editor.go_to_line(line)

    def toggle_breakpoint_here(self) -> None:
        self.toggle_breakpoint_line(self.editor.current_line())

    def clear_breakpoints(self) -> None:
        self.session.breakpoints.clear()
        self.refresh()
        self._log(tr("all breakpoints cleared"))

    # ============================================================== options

    def set_release(self, release: str, reload: bool = True) -> None:
        if release not in RELEASES:
            return
        changed = release != self.session.release
        self.session.release = release
        self._put("release", release)
        self._sync_actions()
        if changed and reload:
            self.assemble(quiet_if_empty=True)

    def set_endian(self, endian: str) -> None:
        changed = endian != self.session.endian
        self.session.endian = endian
        self._put("endian", endian)
        self._sync_actions()
        if changed:
            self.assemble(quiet_if_empty=True)

    def set_abi(self, on: bool) -> None:
        self.session.abi = on
        self._put("abi", "1" if on else "0")
        self._sync_actions()
        if self.hex_text is not None:           # the listing shows register names
            self.assemble(quiet_if_empty=True)
        else:
            self.refresh()

    def set_theme(self, name: str, save: bool = True) -> None:
        """Configure > Theme: Qt's widgets get the palette (System: the
        desktop's, followed live; Light / Dark: Fusion), the views drawn
        here get the matching :mod:`theme` colours and are repainted."""
        app = QtWidgets.QApplication.instance()
        if not hasattr(app, "_emips_system_style"):
            app._emips_system_style = app.style().name()
        self.theme_name = name
        if name == "system":
            app.setStyle(app._emips_system_style)
            app.setPalette(QtGui.QPalette())    # nothing pinned: the platform's
        else:
            app.setStyle("Fusion")
            app.setPalette(dark_palette() if name == "dark" else app.style().standardPalette())
        self.actions[f"theme_{name}"].setChecked(True)
        if save:
            self._put("theme", name)
        self._follow_palette(force=True)

    def _follow_palette(self, force: bool = False) -> None:
        dark = self.theme_name == "dark" or (
            self.theme_name == "system" and theme.palette_is_dark(
                QtWidgets.QApplication.instance().palette()))
        if theme.set_dark(dark) or force:
            self.restyle()

    def changeEvent(self, event) -> None:
        """System theme: the desktop switched light / dark while running."""
        super().changeEvent(event)
        if event.type() in _PALETTE_EVENTS and getattr(self, "theme_name", None) == "system":
            self._follow_palette()

    def restyle(self) -> None:
        """Repaint every view drawn here in the current theme colours."""
        self.datapath.restyle()
        self.datapath24k.restyle()
        self.legend.set_mode(self.session.mode)
        self.console.restyle()
        self.checks.restyle()
        for view in (self.reference, self.encoder, self.converter):
            view.update_view()
        self.editor.gutter.update()
        if self.guide is not None:
            self.guide.restyle()
        self.refresh()

    def zoom(self, step: int) -> None:
        f = self.editor.font().pointSize()
        size = DEFAULT_FONT if step == 0 else f + step
        self.editor.set_point_size(size)
        self._put("font_size", str(self.editor.font().pointSize()))

    def open_settings(self) -> SettingsDialog:
        s = self.session
        dlg = SettingsDialog(s.release, s.endian, s.forwarding, s.options, self)
        dlg.applied.connect(self.apply_settings)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.open()
        return dlg

    def apply_settings(self, release: str, endian: str, forwarding: bool,
                       options: Options) -> None:
        """Store the Settings-dialog values; re-assemble when the program
        would assemble differently, else reset the machine."""
        s = self.session
        old = s.options
        reassemble = (release != s.release or endian != s.endian
                      or options.assembler_differs(old))
        machine = (options != old or forwarding != s.forwarding)
        s.release, s.endian, s.forwarding, s.options = release, endian, forwarding, options
        self._put("release", release)
        self._put("endian", endian)
        self._put("forwarding", "1" if forwarding else "0")
        for k, v in options.to_strings().items():
            self._put(f"options/{k}", v)
        self.fwd.blockSignals(True)
        self.fwd.setChecked(forwarding)
        self.fwd.blockSignals(False)
        self._sync_actions()
        self._show_feature_panes()
        if reassemble:
            self.assemble(quiet_if_empty=True)
        elif machine and s.program is not None:
            self.reset()
            self.status.showMessage(tr("settings applied — the program was restarted"), 5000)

    def set_language(self, code: str) -> None:
        self._put("language", code)
        self._sync_actions()
        QtWidgets.QMessageBox.information(
            self, tr("Language"), tr("The language changes the next time eMIPS starts."))

    # ================================================================== help

    def shortcuts_html(self) -> str:
        rows = []
        for a in self.actions.values():
            keys = ", ".join(k.toString(QtGui.QKeySequence.SequenceFormat.NativeText)
                             for k in a.shortcuts())
            if keys:
                rows.append(f"<tr><td><b>{keys}</b></td>"
                            f"<td>{a.text().replace('&', '')}</td></tr>")
        return (f"<h2>{tr('Keyboard shortcuts')}</h2><table cellpadding=3>"
                + "".join(rows) + "</table><p>"
                + tr("Click a line number in the editor to set or clear a breakpoint.")
                + "</p>")

    def board_nmi(self) -> None:
        """The Board pane's ON/NMI button."""
        self.session.board_nmi()
        self.refresh()

    def save_disk(self, slot: str, path: str | None = None) -> None:
        """The Board pane's Save Disk: the disk as the program left it, to
        a file (its image file by default)."""
        s = self.session
        if path is None:
            source = s.disk_sources().get(slot, "")
            start = "" if source.lower().startswith("blank:") else source
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, tr("Save Disk"), start, tr("Disk images (*.img);;All files (*)"))
            if not path:
                return
        try:
            self._log(s.save_disk(slot, path))
        except (OSError, ValueError) as exc:
            self._fail(str(exc))

    def show_guide(self, page: str = "", fragment: str = "") -> GuideView:
        """Help > User Guide: the HTML guide in its own window (one,
        raised again when asked again); ``page`` / ``fragment``: where."""
        if self.guide is None:
            self.guide = GuideView(self)
            self.guide.open_code.connect(self.open_guide_program)
            self.guide.open_path.connect(self.open_guide_file)
        if page:
            self.guide.show_page(page, fragment)
        self.guide.show()
        self.guide.raise_()
        self.guide.activateWindow()
        return self.guide

    def open_guide_program(self, text: str) -> None:
        """A program of the guide into the editor (its ``# release:``,
        ``# arch:``, ``# core:`` ... lines set up the machine:
        emips.profile.guide_profile), assembled."""
        if not self._confirm_discard(tr("User Guide")):
            return
        self._set_text(text, tr("guide example"), None, None)
        from ..profile import guide_profile
        try:
            prof = guide_profile(text)
        except ValueError:
            prof = None
        if prof is not None:
            self.apply_profile(prof)
        self.assemble()
        self.raise_()
        self.activateWindow()

    def open_guide_file(self, path: str) -> None:
        """A lab program the guide links to, opened in the editor."""
        if self._confirm_discard(tr("User Guide")):
            self.open_file(path)
            self.raise_()
            self.activateWindow()

    def show_help(self, page: str) -> QtWidgets.QDialog:
        title, body = {
            "quick": (tr("Quick Start"), helptext.quick_start),
            "keys": (tr("Keyboard Shortcuts"), self.shortcuts_html),
            "syscalls": (tr("System Calls"), helptext.syscalls),
            "registers": (tr("Registers"), helptext.registers),
            "about": (tr("About eMIPS"), helptext.about),
        }[page]
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        lay = QtWidgets.QVBoxLayout(dlg)
        view = QtWidgets.QTextBrowser()
        view.setOpenExternalLinks(True)
        view.setHtml(body())
        lay.addWidget(view)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.close)
        lay.addWidget(buttons)
        dlg.resize(620, 520)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # forget the closed (deleted) ones now: no slot on `destroyed`, which
        # Qt may emit while a garbage-collected window is half torn down
        self._dialogs = [d for d in self._dialogs if _alive(d)] + [dlg]
        dlg.show()
        return dlg

    # ================================================================== run

    def assemble(self, quiet_if_empty: bool = False) -> None:
        """Assemble the editor text — or reload the Intel HEX image — and
        reset."""
        self.pause()
        s = self.session
        keep = set(s.breakpoints)
        if self.hex_text is not None:
            ok = self._load_program(s)
            if ok:
                self.editor.setReadOnly(False)
                self.editor.setPlainText(s.listing)
                self.editor.document().setModified(False)
                self.editor.setReadOnly(True)
        elif not self.editor.toPlainText().strip():
            self._assembled_text = None
            if not quiet_if_empty:
                self.status.showMessage(tr("nothing to assemble — type a program, open a "
                                           "file or pick an example"), 6000)
            return
        elif self.lang == "c":
            self.status.showMessage(tr("building the C program…"))
            QtWidgets.QApplication.processEvents()
            ok = self._load_program(s)
            self.status.clearMessage()
            self._assembled_text = self.editor.toPlainText() if ok else None
        else:
            ok = self._load_program(s)
            self._assembled_text = self.editor.toPlainText() if ok else None
        if ok:
            code = set(s.program.source_map)
            s.breakpoints = {a for a in keep if a in code}
            if self.hex_text is None and self.lang == "c":
                self._log(tr("built for {arch} {release}, {endian}-endian: {n} instruction "
                             "words (the Disassembly pane shows the code)").format(
                    arch=arch_title(s.options.arch), release=s.options.rv_extensions if s.rv else s.release, endian=s.endian,
                    n=len(code)))
                self._log(s.c_build)
                for key in ("disassembly", "source"):
                    self.docks[key].show()
                self.docks["disassembly"].raise_()
            elif self.hex_text is None:
                self._log(tr("assembled for {arch} {release}, {endian}-endian: "
                             "{n} instruction words").format(
                    arch=arch_title(s.options.arch), release=s.options.rv_extensions if s.rv else s.release, endian=s.endian,
                    n=len(code)))
                for text in s.assembler_messages():     # (LF12: .print, .warning ...)
                    self._log(text)
            for note in s.disk_notes:           # a disk image that is not there
                self._log(tr("disk: {note}").format(note=note))
        else:
            self._fail(s.errors[0], "\n".join(s.errors))
            self.refresh()                      # nothing runnable: empty views
            line = _error_line(s.errors[0])
            if line:
                self.editor.highlight({line: theme.colors().error_line})
                self.editor.show_line(line)
            self._sync_actions()
            return
        self._sync_actions()
        self.refresh()

    def reset(self) -> None:
        self.pause()
        if self.session.program is not None:
            self.session.reset()
        self.refresh()

    def _mode_changed(self, *_) -> None:
        self.pause()
        s = self.session
        i = self.mode.currentIndex()
        model = {1: "5-stage", 2: "24k", 3: "p8700"}.get(i, s.options.model)
        if model != s.options.model:            # 5-stage / 24K / P8700: the pipeline model
            s.options = replace(s.options, model=model)
            for k, v in s.options.to_strings().items():
                self._put(f"options/{k}", v)
        s.set_mode("pipeline" if i else "instruction",
                   forwarding=self.fwd.isChecked())
        if i and s.mode != "pipeline":          # (the MT Module on MIPS64)
            for note in s.program_notes:
                self._log(tr("adjusted: {note}").format(note=note))
            if s.program_notes:
                self.status.showMessage(s.program_notes[-1], 10000)
        self._put("mode", s.mode)
        self._put("forwarding", "1" if s.forwarding else "0")
        self.legend.set_mode(s.mode)
        self._sync_actions()
        self.docks["datapath"].raise_()
        self.refresh()

    def show_part(self, key: str) -> None:
        """A datapath part was clicked: what it is and does now."""
        if self.part_dialog is None:
            self.part_dialog = PartDialog(self)
        self.part_dialog.show_part(self.session, key)

    def _part_numbers(self) -> None:
        on = self.actions["part_numbers"].isChecked()
        self.datapath.show_numbers(on)
        self.datapath24k.show_numbers(on)
        self._put("part_numbers", "1" if on else "0")

    def _core_changed(self, index: int) -> None:
        s = self.session
        if 0 <= index < len(s.machines) and index != s.cpu:
            s.select(index)
            self.refresh()

    def _sync_cores(self) -> None:
        """The core selector: one entry per core with its power state,
        shown only with more than one core."""
        s = self.session
        rows = s.cores() if s.cps is not None or len(s.machines) > 1 else []
        self.core_action.setVisible(bool(rows))
        self.core.blockSignals(True)
        if self.core.count() != len(rows):
            self.core.clear()
            self.core.addItems(["" for _ in rows])
        for r in rows:
            state = tr("exited") if r["halted"] else tr("waiting") if r["waiting"] \
                else tr(r["state"])
            name = s.cpu_name(r["cpu"])
            self.core.setItemText(r["cpu"], (tr("core {n}").format(n=r["cpu"])
                                             if name.startswith("core") else name)
                                  + f" — {state}")
        if rows:
            self.core.setCurrentIndex(s.cpu)
        self.core.blockSignals(False)

    def _load_program(self, s: Session) -> bool:
        """Load the document into ``s``: the HEX / ELF image, the C program
        (built), or the editor's assembly (Assemble; every run of the
        checks)."""
        if self.hex_text is not None:
            return (s.load_elf(self.hex_text, str(self.path or "")) if isinstance(
                self.hex_text, bytes) else s.load_hex(self.hex_text))
        if self.lang == "c":
            return s.load_c(self.editor.toPlainText(),
                            self.path.name if self.path else "program.c",
                            optimize=f"-O{s.options.c_optimize}")
        return s.load_source(self.editor.toPlainText(),
                             str(self.path) if self.path else None)

    # -- checks (PLAN M21 phase 2; emips.check, gui/checkpane.py) ---------------------------

    def _show_check_source(self) -> None:
        """The Checks pane says where the checks come from now."""
        sibling = self.path.with_suffix(".check") if self.path is not None else None
        if self.checks_file is not None:
            self.checks.set_source(self.checks_file.name, chosen=True)
        elif sibling is not None and sibling.is_file() and not (
                self.hex_text is None and check.has_checks(self.editor.toPlainText())):
            self.checks.set_source(sibling.name)
        else:
            self.checks.set_source(None)

    def run_checks(self) -> None:
        """Run > Check (F8): the program as its checks say, each run in a
        fresh session (the one being debugged is left alone)."""
        if self.checks.running:
            return
        dock = self.docks["checks"]
        dock.show()
        dock.raise_()
        self._show_check_source()
        text = self.editor.toPlainText() if self.hex_text is None else None
        try:
            found = check.checks_of(text, self.path, self.checks_file)
        except OSError as exc:
            self.checks.show_problem(tr("cannot read {file}: {why}").format(
                file=exc.filename, why=exc.strerror))
            return
        except check.CheckError as exc:
            where = Path(exc.source).name if exc.source else tr("the program")
            self.checks.show_problem(f"{where}: {exc}", None if exc.source else exc.line)
            return
        if not found.runs:
            self.checks.show_problem(tr("no checks: add '# check: …' lines to the program, "
                                        "save a NAME.check next to it, or choose a checks "
                                        "file (Run → Checks File…)"))
            return
        s = self.session

        def make(run: check.Run) -> Session:
            release, endian, mode, fwd, options = check.machine_for(
                run, s.release, s.endian, s.mode, s.forwarding, s.options)
            new = Session(release=release, endian=endian, mode=mode, forwarding=fwd,
                          abi=s.abi, options=options)
            if not self._load_program(new):
                raise ValueError(new.errors[0] if new.errors else tr("cannot load the program"))
            return new
        self.checks.start(found.runs, make, lines=found.file is None)

    def _checks_finished(self, passed: int, total: int) -> None:
        self.status.showMessage(
            tr("checks: all {t} passed").format(t=total) if passed == total and total
            else tr("checks: {p} of {t} passed").format(p=passed, t=total), 8000)

    def choose_checks_file(self) -> None:
        """Run > Checks File…: check against a checks file from now on."""
        start = str(self.path.parent) if self.path is not None else self._start_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Checks file"), start,
            tr("Checks") + " (*.check);;" + tr("All files") + " (*)")
        if path:
            self.checks_file = Path(path)
            self.run_checks()

    def clear_checks_file(self) -> None:
        self.checks_file = None
        self._show_check_source()

    def _check_line(self, line: int) -> None:
        """A check of the program's own lines was double-clicked."""
        self.editor.go_to_line(line)
        self.editor.setFocus()

    def _ensure_assembled(self) -> bool:
        """Auto-assemble (Settings > Assembler): when the source changed
        since the last assembly, assemble it again before running.  False
        if that failed."""
        if self.hex_text is not None or not self.session.options.auto_assemble:
            return True
        text = self.editor.toPlainText()
        if not text.strip() or text == self._assembled_text:
            return True
        self.assemble()
        if self.session.program is None or self._assembled_text != text:
            return False
        self.status.showMessage(tr("the source changed — assembled again"), 4000)
        return True

    def step(self) -> None:
        if not self._ensure_assembled():
            return
        self._report(self.session.step())
        self.refresh()

    def step_line(self) -> None:
        self._line_step(lambda s: s.step_line(over=False))

    def next_line(self) -> None:
        self._line_step(lambda s: s.step_line(over=True))

    # ============================================================ GDB

    #: the run controls GDB owns while it is connected
    GDB_LOCKED = ("assemble", "run", "pause", "step", "step_back", "step_line",
                  "next_line", "finish", "run_to_cursor", "reset")

    def toggle_gdb_server(self) -> None:
        """Run > GDB Server: wait for GDB on the port (Settings key
        ``gdb_port``, 1234); while it is connected GDB runs the program
        and the views follow (emips.gdbstub.GdbServer)."""
        on = self.actions["gdb_server"].isChecked()
        if on == (self.gdb is not None):
            return
        if not on:
            self._stop_gdb()
            self._log(tr("GDB server stopped"))
            return
        if not self._ensure_assembled() or self.session.program is None:
            self.actions["gdb_server"].setChecked(False)
            return
        from ..gdbstub import GdbServer, exec_file_for
        self.pause()
        s = self.session
        port = int(self._get("gdb_port", "1234"))
        path = str(self.path) if self.path is not None else None
        try:
            self.gdb = GdbServer(s, port, exec_file=lambda: exec_file_for(s, path))
        except OSError as exc:
            self.actions["gdb_server"].setChecked(False)
            self._fail(tr("GDB server: cannot listen on port {port} ({error})").format(
                port=port, error=exc.strerror or exc))
            return
        self.gdb_timer.start()
        self._log(tr("GDB server on port {port}: in a terminal, {gdb} -ex "
                     "'target remote :{port}'").format(
            port=self.gdb.port, gdb=gdb_program(self.session.options.arch)))
        self._show_gdb_state()

    def _stop_gdb(self) -> None:
        if self.gdb is not None:
            self.gdb.close()
            self.gdb = None
        self.gdb_timer.stop()
        self._lock_for_gdb(False)
        self.gdb_label.hide()
        if self.actions["gdb_server"].isChecked():
            self.actions["gdb_server"].setChecked(False)

    def _lock_for_gdb(self, on: bool) -> None:
        for k in self.GDB_LOCKED:
            self.actions[k].setEnabled(not on)

    def _gdb_tick(self) -> None:
        g = self.gdb
        if g is None:
            return
        what = g.poll()
        if what == "connected":
            self._lock_for_gdb(True)
            self._log(tr("GDB connected: it runs the program now"))
        elif what == "running":
            self._gdb_ticks += 1
            if self.session.needs_input() and not getattr(self.console, "asking", False):
                self.console.ask_input(True)
                self.docks["console"].raise_()
            if self._gdb_ticks % 10 == 0:
                self.refresh(light=True)
        elif what == "stopped":
            self.refresh()
        elif what == "left":
            self._lock_for_gdb(False)
            self._log(tr("GDB left") if g.exited is None else
                      tr("the program exited ({code}); GDB left").format(code=g.exited))
            g.exited = None
            self.refresh()
        if what:
            self._show_gdb_state()

    def _show_gdb_state(self) -> None:
        g = self.gdb
        if g is None:
            self.gdb_label.hide()
            return
        text = {"waiting": tr("GDB: waiting on :{port}"), "running": tr("GDB: running"),
                "stopped": tr("GDB: connected")}[g.state].format(port=g.port)
        self.gdb_label.setText(text)
        self.gdb_label.show()

    def request_rnmi(self) -> None:
        """The Interrupts tab's RNMI button: the selected hart's RNMI input
        (Smrnmi, cause 0)."""
        s = self.session
        if s.machine is None:
            return
        try:
            s.rnmi(0)
        except ValueError as exc:
            self.status.showMessage(str(exc), 4000)
            return
        self.status.showMessage(tr("resumable NMI requested: taken before an instruction "
                                   "once mnstatus.NMIE = 1"), 4000)
        self.refresh()

    def debug_interrupt(self) -> None:
        """Run > Debug Interrupt: the DINT input of the selected core."""
        s = self.session
        if s.machine is None:
            return
        if s.machine.debug_mode:
            self.status.showMessage(tr("the core is in Debug Mode: a debug interrupt is "
                                       "ignored"), 4000)
            return
        s.debug_interrupt()
        self.status.showMessage(tr("debug interrupt requested: taken before the next "
                                   "instruction"), 4000)
        self.refresh()

    def set_probe(self, on: bool, trap: bool) -> None:
        """The Debug tab's probe boxes: connect / remove the EJTAG probe."""
        s = self.session
        if s.machine is None:
            return
        self.pause()
        boot = s.probe_boot if on else False
        s.set_probe(on, trap=trap, boot=boot)
        self.status.showMessage(tr("EJTAG probe connected — the program restarted") if on
                                else tr("EJTAG probe removed — the program restarted"), 4000)
        self.refresh()

    def finish(self) -> None:
        self._line_step(lambda s: s.finish(limit=2_000_000))

    def _line_step(self, how) -> None:
        if not self._ensure_assembled() or self.session.program is None:
            return
        self.pause()
        self._report(how(self.session))
        self.refresh()

    def step_back(self) -> None:
        self.pause()
        if not self.session.step_back():
            self.status.showMessage(tr("at the start"), 2000)
        self.refresh()

    def run(self) -> None:
        if not self._ensure_assembled() or self.session.program is None:
            return
        self.session.budget = self.session.options.step_limit or None
        self._speed_changed(self.speed.value())
        self.timer.start()

    def run_to_cursor(self) -> None:
        if not self._ensure_assembled():
            return
        addr = self.session.line_address(self.editor.current_line())
        if addr is None:
            self.status.showMessage(tr("no instruction at or after the cursor"), 4000)
            return
        self.session.run_to = addr
        self.run()

    def pause(self) -> None:
        if hasattr(self, "timer"):
            self.timer.stop()
        self.session.budget = None

    def _speed_changed(self, value: int) -> None:
        self.timer.setInterval(SPEEDS[value][1])
        self._put("speed", str(value))

    def _tick(self) -> None:
        n = SPEEDS[self.speed.value()][0]
        r = self.session.run(max_steps=n)
        if r.stopped or r.error:
            self.pause()
            self._report(r)
        self.refresh(light=n > 50 and not r.stopped)

    def give_input(self, line: str) -> None:
        self.session.give_input(line)
        self.console.ask_input(False)
        self._log(tr("input: {line}").format(line=line))

    def type_keys(self, text: str) -> None:
        self.session.type_keys(text)
        self._log(tr("keyboard: {text}").format(text=repr(text)))

    def toggle_breakpoint_line(self, line: int) -> None:
        s = self.session
        if s.program is None:
            return
        if s.c_file is not None:                # a C line: every place its code is
            addrs = sorted(a for a in s.line_addresses(s.c_file, line)   # (inlined copies too)
                           if s.editor_line(a) == line)
            if not addrs:
                return
            on = not any(a in s.breakpoints for a in addrs)
            for a in addrs:
                (s.breakpoints.add if on else s.breakpoints.discard)(a)
            self._log((tr("breakpoint set at 0x{addr:08x} (line {line})") if on else
                       tr("breakpoint cleared at 0x{addr:08x} (line {line})"))
                      .format(addr=addrs[0], line=line)
                      + (tr(" and {n} more places (inlined)").format(n=len(addrs) - 1)
                         if len(addrs) > 1 else ""))
            self.refresh()
            return
        addrs = sorted(a for a, src in s.program.source_map.items() if src.line == line)
        if not addrs:
            return
        on = s.toggle_breakpoint(addrs[0])
        self._log((tr("breakpoint set at 0x{addr:08x} (line {line})") if on else
                   tr("breakpoint cleared at 0x{addr:08x} (line {line})"))
                  .format(addr=addrs[0], line=line))
        self.refresh()

    def _report(self, r) -> None:
        if r.error:
            self._fail(r.error)
        elif r.stopped == "input":
            self.console.ask_input(True)
            self.docks["console"].raise_()
            self.status.showMessage(tr("the program waits for input (Console)"), 8000)
        elif r.stopped == "cursor":
            self.status.showMessage(tr("reached the cursor line"), 4000)
        elif r.stopped == "breakpoint":
            self.status.showMessage(tr("breakpoint"), 4000)
        elif r.stopped == "limit":
            self.status.showMessage(tr("step limit reached (Settings > Run) — press Run "
                                       "to continue"), 8000)
        elif r.stopped == "exit":
            self.status.showMessage(tr("the program has ended"), 4000)
        elif r.stopped:
            self.status.showMessage(r.stopped, 4000)

    def _fail(self, message: str, log: str | None = None) -> None:
        self._log(log or message)
        self.status.showMessage(message, 8000)

    def _log(self, text: str) -> None:
        self.messages.appendPlainText(text)

    # ============================================================== refresh

    def refresh(self, light: bool = False) -> None:
        """Redraw everything from the session (``light``: while running fast,
        skip the expensive views)."""
        s = self.session
        self._sync_cores()
        if s.machine is None:                   # no program: empty views
            self.editor.set_core_marks({}, False)
            for view in (self.datapath, self.registers, self.console, self.chart,
                         self.disasm, self.memory, self.caches, self.tlb, self.counters,
                         self.trace, self.board,
                         self.system, self.mtview, self.source):
                view.refresh(s)
            self.details.show_view(None)
            self.editor.breakpoint_lines = set()
            self.state_label.setText(" " + tr("no program — type one, open a file or "
                                              "pick File > Examples") + " ")
            return
        self.registers.refresh(s)
        self.console.refresh(s)
        if light:
            self.state_label.setText(" " + tr("running… {n} steps").format(n=s.steps) + " ")
            return
        lines: dict[int, str] = {}
        if s.pipeline is not None:
            for st, line in s.stage_lines().items():   # (the P8700's: textbook colours)
                lines.setdefault(line, (theme.stage_colours(st)
                                        or theme.colors().stage["ID"])[0])
        else:
            cur = s.current_line()
            if cur:
                lines[cur] = theme.colors().current_line
        self.editor.highlight(lines)
        cur = s.current_line()
        if cur:
            self.editor.show_line(cur)
        self.editor.breakpoint_lines = s.editor_breakpoint_lines()
        self.editor.gutter.update()
        if s.pipeline is not None and s.pipeline.name in ("24K", "34K"):
            self.datapath_pane.setCurrentWidget(self.datapath24k)
            self.datapath24k.refresh(s)
        elif s.pipeline is not None and s.pipeline.name == "p8700":
            self.datapath_pane.setCurrentWidget(self.core8700)
            self.core8700.refresh(s)
        else:
            self.datapath_pane.setCurrentWidget(self.datapath)
            self.datapath.refresh(s)
        self.chart.refresh(s)
        self.disasm.refresh(s)
        self.details.show_view(s.last_view)
        self.memory.refresh(s)
        self.caches.refresh(s)
        self.tlb.refresh(s)
        self.counters.refresh(s)
        self.trace.refresh(s)
        self.board.refresh(s)
        self.system.refresh(s)
        self.mtview.refresh(s)
        self.source.refresh(s)
        if self.part_dialog is not None and self.part_dialog.isVisible():
            self.part_dialog.refresh(s)
        marks: dict[int, list[int]] = {}
        if s.cps is not None:                   # where the other cores are
            for c in s.cores():
                if not c["selected"] and c["line"] and c["running"] and not c["halted"]:
                    marks.setdefault(c["line"], []).append(c["cpu"])
        self.editor.set_core_marks(marks, s.cps is not None)
        m = s.machine
        if s.pipeline is not None:
            text = tr("{n} cycles").format(n=s.steps)
            if s.pipeline.retired:
                text += f", CPI {s.pipeline.cycles / s.pipeline.retired:.2f}"
        else:
            text = tr("{n} steps").format(n=s.steps)
        text += f", pc 0x{m.pc:08x}"
        if s.cps is not None:
            text = tr("core {n}").format(n=s.cpu) + ": " + text
        if m.debug_mode:
            text += " — " + tr("Debug Mode")
        if s.halted:
            text += " — " + tr("ended")
        self.state_label.setText(f" {text} ")


def _error_line(msg: str) -> int | None:
    m = re.match(r"line (\d+):", msg) or re.match(r"[^\s:]+\.c:(\d+):", msg)
    return int(m.group(1)) if m else None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("eMIPS")
    app.setOrganizationName("eMIPS")
    settings = default_settings()
    i18n.install(str(settings.value("language") or "en"), app)
    win = MainWindow(argv[0] if argv else None, settings=settings)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
