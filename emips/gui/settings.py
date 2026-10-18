"""Configure > Settings: one dialog, six tabs (Processor, Assembler,
Memory & boot, Caches, Exceptions & I/O, Run).  It edits a copy of the values and
hands them back through :attr:`SettingsDialog.applied` on OK / Apply; the
main window decides what has to be re-assembled or reset."""

from __future__ import annotations

import html

from ..cache import Geometry
from ..pipeline import LATENCY
from ..timing import MEMORY_TIMING
from .i18n import N_, tr
from ..options import MAX_CORES, MAX_SHADOW_SETS, Options
from .qt import QtCore, QtWidgets, Signal

Qt = QtCore.Qt
RELEASES = ["R1", "R2", "R3", "R5", "R6"]
RELEASE_NOTES = {
    "R1": N_("MIPS32 Release 1 (the original MIPS32)"),
    "R2": N_("Release 2: ROTR, EXT/INS, WSBH, SEB/SEH, EHB, 64-bit FPU"),
    "R3": N_("Release 3: adds EVA and microMIPS (microMIPS is not simulated)"),
    "R5": N_("Release 5: the full pre-R6 set (default)"),
    "R6": N_("Release 6: compact branches, no branch-likely, new encodings"),
}
LATENCY_LABELS = {
    "MUL": N_("MUL (multiply into a register)"),
    "MULT": N_("MULT / MADD / MSUB (into HI / LO)"),
    "DIV": N_("DIV / DIVU"),
    "FP_ADD": N_("FP add / subtract"),
    "FP_MUL": N_("FP multiply"),
    "FP_MADD": N_("FP multiply-add"),
    "FP_CVT": N_("FP conversions"),
    "FP_CMP": N_("FP compare"),
    "FP_DIV_S": N_("FP divide, single"),
    "FP_DIV_D": N_("FP divide, double"),
    "FP_SQRT_S": N_("FP square root, single"),
    "FP_SQRT_D": N_("FP square root, double"),
}
MEMORY_LABELS = {
    "L2": N_("L1 miss served by the L2"),
    "L3": N_("L1 miss served by the L3"),
    "MEM": N_("L1 miss served by memory; uncached read"),
    "BEAT": N_("each further doubleword of a line"),
    "BYPASS": N_("extra for an uncached read past an L2"),
    "WRITE": N_("the system taking one write"),
    "UTLB": N_("micro-TLB miss: refill from the JTLB"),
    "C2C": N_("coherent miss served by another core's cache"),
    "UPG": N_("store to a Shared line: invalidating the others"),
    "INTV": N_("coherent request: until the other cores see it"),
}
BOOT_TEXT = {
    "kernel": (N_("Kernel mode, like MARS (default)"),
               N_("Status: kernel mode, FPU enabled, exceptions go to the handler at "
                  "0x80000180 (.ktext); $sp = 0x7fffeffc, $gp = 0x10008000; the PC "
                  "starts at main.")),
    "user": (N_("User mode"),
             N_("As above, but the program runs in user mode: privileged instructions "
                "(mfc0, mtc0, eret ...) and kernel addresses (MMIO at 0xffff0000) raise "
                "exceptions — the operating-system view.")),
    "reset": (N_("Hardware reset state"),
              N_("Status after a real reset: BEV = ERL = 1 (exceptions go to "
                 "0xbfc00380), interrupts and the FPU disabled. For boot code. The PC "
                 "starts at main.")),
}


def _note(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(html.escape(text))
    lab.setWordWrap(True)
    lab.setTextFormat(Qt.TextFormat.RichText)
    lab.setStyleSheet("color: palette(placeholder-text);")
    return lab


class SettingsDialog(QtWidgets.QDialog):
    """Edits release, byte order, forwarding and :class:`Options`."""

    #: (release, endian, forwarding, options) on OK / Apply
    applied = Signal(str, str, bool, object)

    def __init__(self, release: str, endian: str, forwarding: bool, options: Options,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Settings"))
        self.tabs = QtWidgets.QTabWidget()
        for page, title in ((self._processor_tab(), tr("Processor")),
                            (self._assembler_tab(), tr("Assembler")),
                            (self._boot_tab(), tr("Memory && boot")),
                            (self._cache_tab(), tr("Caches")),
                            (self._timing_tab(), tr("Memory timing")),
                            (self._io_tab(), tr("Exceptions && I/O")),
                            (self._run_tab(), tr("Run"))):
            self.tabs.addTab(page, title)
        B = QtWidgets.QDialogButtonBox.StandardButton
        self.buttons = QtWidgets.QDialogButtonBox(
            B.Ok | B.Cancel | B.Apply | B.RestoreDefaults)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(B.Apply).clicked.connect(self.apply)
        self.buttons.button(B.RestoreDefaults).clicked.connect(self.restore_defaults)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.tabs)
        lay.addWidget(self.buttons)
        self.set_values(release, endian, forwarding, options)
        # tall enough for the Processor tab: word-wrapped notes make a form
        # under-report its height, which would squeeze what is below them
        screen = self.screen().availableGeometry() if self.screen() else None
        self.resize(720, min(780, screen.height() - 40) if screen else 780)

    # -- tabs ----------------------------------------------------------------

    def _processor_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        self.release = QtWidgets.QComboBox()
        for r in RELEASES:
            self.release.addItem(f"MIPS32 {r}", r)
        self.release_note = _note("")
        self.release.currentIndexChanged.connect(
            lambda _: self.release_note.setText(html.escape(tr(
                RELEASE_NOTES[self.release.currentData()]))))
        form.addRow(tr("Release"), self.release)
        form.addRow("", self.release_note)
        self.endian = QtWidgets.QComboBox()
        self.endian.addItem(tr("Big-endian"), "big")
        self.endian.addItem(tr("Little-endian"), "little")
        form.addRow(tr("Byte order"), self.endian)
        self.model = QtWidgets.QComboBox()
        self.model.addItem(tr("5-stage (textbook)"), "5-stage")
        self.model.addItem(tr("MIPS 24K (8 stages, branch prediction)"), "24k")
        form.addRow(tr("Pipeline model"), self.model)
        self.forwarding = QtWidgets.QCheckBox(tr("Forwarding (bypassing) in the pipeline"))
        self.forwarding.setToolTip(tr("The 5-stage model only: the 24K always has its "
                                      "bypasses"))
        form.addRow("", self.forwarding)
        self.count_rate = QtWidgets.QComboBox()
        self.count_rate.addItem(tr("every cycle"), 1)
        self.count_rate.addItem(tr("every 2 cycles (typical)"), 2)
        form.addRow(tr("Count timer increments"), self.count_rate)
        self.mmu = QtWidgets.QComboBox()
        self.mmu.addItem(tr("Fixed mapping (no TLB)"), "fixed")
        self.mmu.addItem(tr("Block address translation (BAT)"), "bat")
        for n in (16, 32, 64):
            self.mmu.addItem(tr("TLB, {n} entries (24K)").format(n=n), f"tlb:{n}")
        self.mmu.addItem(tr("VTLB 64 + FTLB 512 (P5600)"), "ftlb")
        self.mmu.addItem(tr("VTLB + FTLB, segmentation control, EVA, page table walker"),
                         "ftlb+sc+pw")
        form.addRow(tr("MMU"), self.mmu)
        form.addRow("", _note(tr("Every MMU maps memory like the fixed mapping after boot "
                                 "(kuseg + 1 GB): the TLBs through the boot firmware's "
                                 "refill handler, BAT through its six base-and-bounds "
                                 "entries.")))
        self.cores = QtWidgets.QSpinBox()
        self.cores.setRange(1, MAX_CORES)
        form.addRow(tr("Cores"), self.cores)
        form.addRow("", _note(tr("More than one core: a Coherent Processing System (the "
                                 "Coherence Manager's GCRs and the Cluster Power "
                                 "Controller) on one memory. Kernel / user boot starts "
                                 "every core at the entry, reset boot core 0 only. The "
                                 "data caches are kept coherent (MESI, cacheability 4 / "
                                 "5), the L2 / L3 are shared.")))
        self.shadow_sets = QtWidgets.QSpinBox()
        self.shadow_sets.setRange(1, MAX_SHADOW_SETS)
        form.addRow(tr("GPR shadow sets"), self.shadow_sets)
        form.addRow("", _note(tr("Copies of the 32 GPRs an exception or interrupt can "
                                 "switch to (SRSCtl, SRSMap; Release 2 and later). The "
                                 "24K is built with 1, 2 or 4; 1 = the normal GPRs only.")))
        box = QtWidgets.QGroupBox(tr("Pipeline latencies (clock cycles)"))
        grid = QtWidgets.QGridLayout(box)
        self.latency: dict[str, QtWidgets.QSpinBox] = {}
        for i, key in enumerate(LATENCY):
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 200)
            spin.setToolTip(key)
            self.latency[key] = spin
            grid.addWidget(QtWidgets.QLabel(tr(LATENCY_LABELS.get(key, key))), i // 2,
                           (i % 2) * 2)
            grid.addWidget(spin, i // 2, (i % 2) * 2 + 1)
        form.addRow(box)
        form.addRow(_note(tr("Latencies apply in the 5-stage pipeline model (the 24K "
                             "has its own, from its manual): MUL and the "
                             "FPU hold EX that long; MULT / DIV keep the multiply / "
                             "divide unit busy while independent instructions go on.")))
        return w

    def _assembler_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.pseudo = QtWidgets.QCheckBox(tr("Allow pseudo-instructions (li, la, move, "
                                             "blt, lw from a label ...)"))
        lay.addWidget(self.pseudo)
        lay.addWidget(_note(tr("Off = strict mode: only real machine instructions; each "
                               "pseudo-instruction is an error that tells which "
                               "instructions to write instead.")))
        self.reorder = QtWidgets.QCheckBox(tr("Fill branch delay slots with nop "
                                              "automatically (.set reorder)"))
        lay.addWidget(self.reorder)
        lay.addWidget(_note(tr("A MIPS CPU always executes the instruction after a branch "
                               "or jump (the delay slot). On: the assembler puts a nop "
                               "there. Off: the next instruction you write is the delay "
                               "slot. A program can still switch with .set reorder / "
                               ".set noreorder.")))
        self.auto_assemble = QtWidgets.QCheckBox(tr("Assemble automatically when the "
                                                    "source changed (Run / Step)"))
        lay.addWidget(self.auto_assemble)
        lay.addStretch()
        return w

    def _boot_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.boot = QtWidgets.QButtonGroup(w)
        for i, (key, (title, text)) in enumerate(BOOT_TEXT.items()):
            rb = QtWidgets.QRadioButton(tr(title))
            rb.setProperty("key", key)
            self.boot.addButton(rb, i)
            lay.addWidget(rb)
            lay.addWidget(_note(tr(text)))
        lay.addSpacing(12)
        lay.addWidget(QtWidgets.QLabel("<b>" + html.escape(tr("Memory map")) + "</b>"))
        lay.addWidget(_note(tr(".text 0x00400000 (code) · .data 0x10010000 · heap "
                               "0x10040000 (sbrk) · stack below 0x7fffeffc · .ktext "
                               "0x80000180 (exception handler) · .kdata 0x90000000 · "
                               "MMIO 0xffff0000")))
        lay.addStretch()
        return w

    def _cache_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()
        self.icache = self._cache_combo()
        self.dcache = self._cache_combo()
        self.l2 = self._cache_combo(("128K", "256K", "512K", "1M"))
        self.l3 = self._cache_combo(("1M", "2M", "4M:8:64"))
        form.addRow(tr("Instruction cache"), self.icache)
        form.addRow(tr("Data cache"), self.dcache)
        form.addRow(tr("L2 cache"), self.l2)
        form.addRow(tr("L3 cache"), self.l3)
        lay.addLayout(form)
        lay.addWidget(_note(tr("Level-1 caches as in a MIPS 24K core: 4-way, 32-byte "
                               "lines, least recently used replacement, virtually "
                               "indexed and physically tagged. The L2 / L3 are "
                               "physically indexed, 8-way (pseudo-LRU), write-back "
                               "without write allocate, like the 24K's SOC-it L2. Type "
                               "SIZE:WAYS:LINE for another shape, e.g. 16K:2:32 or "
                               "512K:4:64. kseg0 and kuseg are cached write-back after "
                               "boot (Config.K0 / KU = 3), kseg1 never.")))
        self.cache_parity = QtWidgets.QCheckBox(tr("Parity on the L1 cache arrays "
                                                   "(ErrCtl.PE, Cache Error exception)"))
        lay.addWidget(self.cache_parity)
        self.cache_ecc = QtWidgets.QCheckBox(tr("ECC on the L2 / L3 (ErrCtl.L2P: 1-bit "
                                                "errors corrected, 2-bit errors are an "
                                                "imprecise Cache Error)"))
        lay.addWidget(self.cache_ecc)
        self.write_buffer = QtWidgets.QCheckBox(tr("Write buffer (4 × 32 bytes): merges "
                                                   "write-through stores, gathers uncached "
                                                   "accelerated ones; SYNC empties it"))
        lay.addWidget(self.write_buffer)
        lay.addSpacing(12)
        lay.addWidget(QtWidgets.QLabel("<b>" + html.escape(tr("Contents after reset"))
                                       + "</b>"))
        self.cache_random = QtWidgets.QRadioButton(tr("Undefined, like real silicon "
                                                      "(random tags and data)"))
        self.cache_hw = QtWidgets.QRadioButton(tr("Invalid: initialized by hardware "
                                                  "(Config7.HCI = 1)"))
        lay.addWidget(self.cache_random)
        lay.addWidget(self.cache_hw)
        lay.addWidget(_note(tr("The kernel and user boot modes run the boot firmware, "
                               "which invalidates every line. In the hardware reset "
                               "state your boot code must do it (CACHE Index Store Tag) "
                               "before it uses kseg0.")))
        lay.addStretch()
        return w

    def _timing_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        box = QtWidgets.QGroupBox(tr("Cycles an access waits (5-stage pipeline: with caches; "
                                     "UTLB with the TLB)"))
        grid = QtWidgets.QGridLayout(box)
        self.mem_timing: dict[str, QtWidgets.QSpinBox] = {}
        for i, key in enumerate(MEMORY_TIMING):
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 1000)
            spin.setToolTip(key)
            self.mem_timing[key] = spin
            grid.addWidget(QtWidgets.QLabel(tr(MEMORY_LABELS.get(key, key))), i, 0)
            grid.addWidget(spin, i, 1)
        lay.addWidget(box)
        lay.addWidget(_note(tr("Defaults from the MIPS 24K manuals: an L2 hit costs "
                               "10-12 cycles, main memory 50 or more; the rest of a line "
                               "follows one doubleword per cycle. A data access that "
                               "waits holds MEM and the stages behind it; an "
                               "instruction miss holds IF. The write buffer has 4 "
                               "entries: a store that finds them all busy waits. "
                               "Without caches memory takes one cycle.")))
        self.nonblocking = QtWidgets.QCheckBox(tr("Loads miss in the background (hit "
                                                  "under miss, like the 24K)"))
        lay.addWidget(self.nonblocking)
        lay.addWidget(_note(tr("Off: a miss stops the pipeline until the line arrives "
                               "(the textbook blocking cache). On: the pipeline goes on; "
                               "only an instruction that needs the loaded register waits.")))
        lay.addStretch()
        return w

    @staticmethod
    def _cache_combo(sizes: tuple[str, ...] = ("8K", "16K", "32K", "64K")
                     ) -> QtWidgets.QComboBox:
        box = QtWidgets.QComboBox()
        box.setEditable(True)
        box.addItem(tr("none"), "none")
        for size in sizes:
            box.addItem(size, size)
        return box

    @staticmethod
    def _set_cache(box: QtWidgets.QComboBox, spec: str) -> None:
        i = box.findData(spec)
        if i < 0:
            box.setEditText(spec)
        else:
            box.setCurrentIndex(i)

    @staticmethod
    def _cache_spec(box: QtWidgets.QComboBox, level: int = 1) -> str:
        text = box.currentText().strip()
        i = box.findText(text)
        spec = box.itemData(i) if i >= 0 else text
        try:
            return "none" if Geometry.parse(spec, level) is None else spec
        except ValueError:
            return "none"

    def _io_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("<b>" + html.escape(tr("When an exception "
                                                             "happens")) + "</b>"))
        self.exc_cp0 = QtWidgets.QRadioButton(tr("Take it, like the hardware: CP0 records "
                                                 "it (Cause, EPC, BadVAddr) and the CPU "
                                                 "jumps to the exception handler"))
        self.exc_stop = QtWidgets.QRadioButton(tr("Stop the simulation and show the "
                                                  "exception (debugging)"))
        for rb in (self.exc_cp0, self.exc_stop):
            lay.addWidget(rb)
        lay.addSpacing(12)
        self.syscalls = QtWidgets.QCheckBox(tr("System call services (MARS / SPIM: "
                                               "print, read, exit ...)"))
        lay.addWidget(self.syscalls)
        lay.addWidget(_note(tr("Off: syscall takes the real System Call exception — "
                               "write your own handler.")))
        form = QtWidgets.QFormLayout()
        self.tx_delay = QtWidgets.QSpinBox()
        self.tx_delay.setRange(0, 10_000)
        self.rx_delay = QtWidgets.QSpinBox()
        self.rx_delay.setRange(0, 10_000)
        form.addRow(tr("MMIO display busy after a character (cycles)"), self.tx_delay)
        form.addRow(tr("MMIO keyboard: cycles between keys"), self.rx_delay)
        lay.addLayout(form)
        lay.addStretch()
        return w

    def _run_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        self.step_limit = QtWidgets.QSpinBox()
        self.step_limit.setRange(0, 100_000_000)
        self.step_limit.setSingleStep(1000)
        self.step_limit.setSpecialValueText(tr("no limit"))
        form.addRow(tr("Run stops after (steps / cycles)"), self.step_limit)
        form.addRow(_note(tr("Protects against endless loops; press Run again to "
                             "continue.")))
        return w

    # -- values ----------------------------------------------------------------

    def set_values(self, release: str, endian: str, forwarding: bool, o: Options) -> None:
        self.release.setCurrentIndex(max(0, RELEASES.index(release)
                                         if release in RELEASES else 3))
        self.release_note.setText(html.escape(tr(RELEASE_NOTES[self.release.currentData()])))
        self.endian.setCurrentIndex(0 if endian == "big" else 1)
        self.forwarding.setChecked(forwarding)
        self.model.setCurrentIndex(max(0, self.model.findData(o.model)))
        self.cores.setValue(o.cores)
        self.shadow_sets.setValue(o.shadow_sets)
        self.count_rate.setCurrentIndex(0 if o.count_rate == 1 else 1)
        spec = "tlb:32" if o.mmu == "tlb" else o.mmu
        if self.mmu.findData(spec) < 0:
            self.mmu.addItem(spec, spec)                 # e.g. from --mmu
        self.mmu.setCurrentIndex(self.mmu.findData(spec))
        for key, spin in self.latency.items():
            spin.setValue(o.latency.get(key, LATENCY[key]))
        self.pseudo.setChecked(o.pseudo)
        self.reorder.setChecked(o.reorder)
        self.auto_assemble.setChecked(o.auto_assemble)
        for b in self.boot.buttons():
            b.setChecked(b.property("key") == o.boot)
        (self.exc_stop if o.exceptions == "stop" else self.exc_cp0).setChecked(True)
        self.syscalls.setChecked(o.syscalls)
        self.tx_delay.setValue(o.tx_delay)
        self.rx_delay.setValue(o.rx_delay)
        self.step_limit.setValue(o.step_limit)
        self._set_cache(self.icache, o.icache)
        self._set_cache(self.dcache, o.dcache)
        self._set_cache(self.l2, o.l2)
        self._set_cache(self.l3, o.l3)
        self.cache_parity.setChecked(o.cache_parity)
        self.cache_ecc.setChecked(o.cache_ecc)
        self.write_buffer.setChecked(o.write_buffer)
        for key, spin in self.mem_timing.items():
            spin.setValue(o.mem_timing.get(key, MEMORY_TIMING[key]))
        self.nonblocking.setChecked(o.nonblocking)
        (self.cache_hw if o.cache_init == "hardware" else self.cache_random).setChecked(True)

    def values(self) -> tuple[str, str, bool, Options]:
        boot = next((b.property("key") for b in self.boot.buttons() if b.isChecked()),
                    "kernel")
        o = Options(
            pseudo=self.pseudo.isChecked(), reorder=self.reorder.isChecked(),
            auto_assemble=self.auto_assemble.isChecked(),
            count_rate=self.count_rate.currentData(),
            latency={k: s.value() for k, s in self.latency.items()},
            boot=boot, exceptions="stop" if self.exc_stop.isChecked() else "cp0",
            syscalls=self.syscalls.isChecked(), tx_delay=self.tx_delay.value(),
            rx_delay=self.rx_delay.value(), step_limit=self.step_limit.value(),
            icache=self._cache_spec(self.icache), dcache=self._cache_spec(self.dcache),
            cache_parity=self.cache_parity.isChecked(),
            cache_init="hardware" if self.cache_hw.isChecked() else "random",
            l2=self._cache_spec(self.l2, 2), l3=self._cache_spec(self.l3, 3),
            cache_ecc=self.cache_ecc.isChecked(),
            write_buffer=self.write_buffer.isChecked(),
            mem_timing={k: s.value() for k, s in self.mem_timing.items()},
            nonblocking=self.nonblocking.isChecked(), mmu=self.mmu.currentData(),
            model=self.model.currentData(), cores=self.cores.value(),
            shadow_sets=self.shadow_sets.value())
        return (self.release.currentData(), self.endian.currentData(),
                self.forwarding.isChecked(), o.validated())

    def restore_defaults(self) -> None:
        self.set_values("R5", "big", True, Options())

    def apply(self) -> None:
        self.applied.emit(*self.values())

    def accept(self) -> None:
        self.apply()
        super().accept()
