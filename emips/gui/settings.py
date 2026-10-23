"""Configure > Settings: one dialog, six tabs (Processor, Assembler,
Memory & boot, Caches, Exceptions & I/O, Run).  It edits a copy of the values and
hands them back through :attr:`SettingsDialog.applied` on OK / Apply; the
main window decides what has to be re-assembled or reset."""

from __future__ import annotations

import html

from ..cores import get as core_preset
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
    "R3": N_("Release 3: adds EVA, segmentation control and the microMIPS manual"),
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
    "DMULT": N_("DMULT / DMUL (64-bit multiply)"),
    "DDIV": N_("DDIV / DMOD (64-bit divide)"),
    "MSA": N_("MSA integer / logic / permute"),
    "MSA_COUNT": N_("MSA PCNT / SAT"),
    "MSA_FP": N_("MSA FP add / convert"),
    "MSA_MUL": N_("MSA multiply / dot product"),
    "MSA_FMADD": N_("MSA FP multiply-add"),
    "MSA_DIV": N_("MSA integer divide"),
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


def _scrolled(page: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    """A tab page that keeps its natural size and scrolls when the window
    is smaller (word-wrapped notes, long forms)."""
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(page)
    return area


class SettingsDialog(QtWidgets.QDialog):
    """Edits release, byte order, forwarding and :class:`Options`."""

    #: (release, endian, forwarding, options) on OK / Apply
    applied = Signal(str, str, bool, object)

    def __init__(self, release: str, endian: str, forwarding: bool, options: Options,
                 parent=None) -> None:
        super().__init__(parent)
        #: the options given (what the dialog has no widget for passes through:
        #: a RISC-V hart's, PLAN M25 — the window's RISC-V settings are phase 4b)
        self._given = options
        self.setWindowTitle(tr("Settings"))
        self.tabs = QtWidgets.QTabWidget()
        for page, title in ((self._processor_tab(), tr("Processor")),
                            (self._assembler_tab(), tr("Assembler")),
                            (self._boot_tab(), tr("Memory && boot")),
                            (self._cache_tab(), tr("Caches")),
                            (self._timing_tab(), tr("Memory timing")),
                            (self._io_tab(), tr("Exceptions && I/O")),
                            (self._run_tab(), tr("Run"))):
            self.tabs.addTab(_scrolled(page), title)
        B = QtWidgets.QDialogButtonBox.StandardButton
        self.buttons = QtWidgets.QDialogButtonBox(
            B.Ok | B.Cancel | B.Apply | B.RestoreDefaults)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(B.Apply).clicked.connect(self.apply)
        self.buttons.button(B.RestoreDefaults).clicked.connect(self.restore_defaults)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.tabs)
        #: what the last change adjusted to keep the settings together (emips.compat)
        self.compat_note = QtWidgets.QLabel()
        self.compat_note.setWordWrap(True)
        lay.addWidget(self.compat_note)
        lay.addWidget(self.buttons)
        self._constraining = True
        self.set_values(release, endian, forwarding, options)
        self._constraining = False
        self.arch.currentIndexChanged.connect(self._arch_chosen)   # (before _constrain)
        for box in (self.arch, self.release, self.board, self.mmu, self.model, self.icache,
                    self.dcache, self.l2, self.l3, self.compressed, self.c_isa, self.dsp,
                    self.msa, self.mt):
            box.currentIndexChanged.connect(self._constrain)
        self.fpu.toggled.connect(self._constrain)
        self.nms.toggled.connect(self._constrain)
        for box in (self.icache, self.dcache, self.l2, self.l3):
            box.editTextChanged.connect(self._constrain)
        for spin in (self.cores, self.vps, self.shadow_sets):
            spin.valueChanged.connect(self._constrain)
        self.boot.buttonToggled.connect(self._constrain)
        self._constrain(note=False)
        # tall enough for the Processor tab: word-wrapped notes make a form
        # under-report its height, which would squeeze what is below them
        screen = self.screen().availableGeometry() if self.screen() else None
        self.resize(720, min(780, screen.height() - 40) if screen else 780)

    # -- tabs ----------------------------------------------------------------

    def _processor_tab(self) -> QtWidgets.QWidget:
        from ..cores import CORES
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        self.core = QtWidgets.QComboBox()
        self.core.addItem(tr("Generic eMIPS core (any setting)"), "")
        for core in CORES.values():
            self.core.addItem(core.title, core.name)
        form.addRow(tr("Core"), self.core)
        self.core_note = _note("")
        form.addRow("", self.core_note)
        self.core.currentIndexChanged.connect(self._core_chosen)
        self.arch = QtWidgets.QComboBox()
        self.arch.addItem(tr("MIPS32"), "mips32")
        self.arch.addItem(tr("MIPS64: 64-bit registers, the doubleword instructions"),
                          "mips64")
        self.arch.addItem(tr("nanoMIPS: the I7200's 16 / 32 / 48-bit instructions "
                             "(Release 6)"), "nanomips")
        form.addRow(tr("Architecture"), self.arch)
        # the nanoMIPS core's options (MD01251: Config5.NMS / CRCP / GI)
        self.nms = QtWidgets.QCheckBox(tr("nanoMIPS subset only (NMS)"))
        self.nms.setToolTip(tr("A core with the nanoMIPS subset (Config5.NMS = 1): the "
                               "instructions MD01247 leaves out of it are Reserved "
                               "Instructions"))
        form.addRow("", self.nms)
        self.crc = QtWidgets.QComboBox()
        for key, text in (("auto", tr("Automatic: when the program has CRC32 instructions")),
                          ("none", tr("None")), ("crc", tr("CRC32B ... CRC32CW"))):
            self.crc.addItem(text, key)
        self.crc.setToolTip(tr("nanoMIPS: the CRC32 instructions (Config5.CRCP)"))
        form.addRow(tr("nanoMIPS CRC"), self.crc)
        self.ginv = QtWidgets.QComboBox()
        for key, text in (("auto", tr("Automatic: what the program uses")),
                          ("none", tr("None")), ("ginvi", tr("GINVI (Config5.GI = 2)")),
                          ("ginvt", tr("GINVI and GINVT (Config5.GI = 3)"))):
            self.ginv.addItem(text, key)
        self.ginv.setToolTip(tr("nanoMIPS: the global invalidate instructions (Config5.GI)"))
        form.addRow(tr("nanoMIPS global invalidate"), self.ginv)
        self.release = QtWidgets.QComboBox()
        for r in RELEASES:
            self.release.addItem(tr("Release {n}").format(n=r[1:]), r)
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
        self.compressed = QtWidgets.QComboBox()
        for key, text in (("auto", tr("Automatic: MIPS16e in Release 1-5, none in Release 6")),
                          ("none", tr("None: MIPS32 only")),
                          ("mips16e", tr("MIPS16e (16-bit instructions)")),
                          ("micromips", tr("microMIPS and MIPS32 (Release 2+)")),
                          ("micromips-only", tr("microMIPS only, no MIPS32 mode "
                                                "(Release 2+)"))):
            self.compressed.addItem(text, key)
        form.addRow(tr("Compressed instructions"), self.compressed)
        form.addRow("", _note(tr("Automatic is the 24K's choice; a program with .set mips16 "
                                 "/ .set micromips code gets that ISA anyway (not on a "
                                 "real core).  microMIPS needs Release 2+ (the M14K) and "
                                 "runs on the 5-stage model (the 24K has none).")))
        self.fpu = QtWidgets.QCheckBox(tr("Floating-point unit (CP1)"))
        self.fpu.setToolTip(tr("Off: a core without an FPU (the 24Kc, the M14K): Status.CU1 "
                               "reads 0, every FPU instruction is Coprocessor Unusable, C is "
                               "built soft-float"))
        form.addRow("", self.fpu)
        self.dsp = QtWidgets.QComboBox()
        for key, text in (("auto", tr("Automatic: Revision 2 when the program has DSP "
                                      "instructions")),
                          ("none", tr("None (the 24K)")),
                          ("dsp", tr("Revision 1 (the 24KE, the 1004K)")),
                          ("dspr2", tr("Revision 2 (the interAptiv, the microAptiv; "
                                       "Revision 3 in Release 6)"))):
            self.dsp.addItem(text, key)
        self.dsp.setToolTip(tr("The DSP Module (MD00374): SIMD on bytes and halfwords, "
                               "saturating fractional arithmetic, the accumulators ac1-ac3, "
                               "DSPControl; Status.MX enables it (Release 2+)"))
        form.addRow(tr("DSP Module"), self.dsp)
        self.msa = QtWidgets.QComboBox()
        for key, text in (("auto", tr("Automatic: when the program has MSA instructions")),
                          ("none", tr("None")),
                          ("msa", tr("MSA (the P5600, P6600, I6400)"))):
            self.msa.addItem(text, key)
        self.msa.setToolTip(tr("The MIPS SIMD Architecture (MD00866): 32 vector registers "
                               "of 128 bits over the FPU's, integer, fixed-point and "
                               "floating-point vector instructions, MSACSR; Config5.MSAEn "
                               "enables it (Release 5+, needs the FPU)"))
        form.addRow(tr("MSA"), self.msa)
        self.msa_wrp = QtWidgets.QSpinBox()
        self.msa_wrp.setRange(0, 32)
        self.msa_wrp.setSpecialValueText(tr("No partitioning"))
        self.msa_wrp.setToolTip(tr("Vector register partitioning (MSAIR.WRP, MD00866 3.4): "
                                   "this many physical vector registers shared by the "
                                   "architectural $w0-$w31; the kernel maps them with "
                                   "MSAMap / MSAUnmap"))
        form.addRow(tr("MSA physical registers"), self.msa_wrp)
        self.mt = QtWidgets.QComboBox()
        for key, text in (("auto", tr("Automatic: 2 VPEs, 5 TCs when the program has MT "
                                      "instructions")),
                          ("none", tr("None")),
                          ("1:2", tr("1 VPE, 2 TCs")), ("1:5", tr("1 VPE, 5 TCs")),
                          ("2:2", tr("2 VPEs, 2 TCs")), ("2:5", tr("2 VPEs, 5 TCs (a 34K)")),
                          ("2:9", tr("2 VPEs, 9 TCs (the 34K's largest)")),
                          ("3:9", tr("3 VPEs, 9 TCs (the I7200's largest)"))):
            self.mt.addItem(text, key)
        self.mt.setToolTip(tr("The MT Module (MD00378): virtual processors (VPEs) with their "
                              "own CP0, thread contexts (TCs) issuing in turn; FORK, YIELD, "
                              "MFTR / MTTR (Release 2-5; on a CPS per core: the 1004K, the "
                              "interAptiv; pipeline mode runs the 34K's pipeline)"))
        form.addRow(tr("MT Module"), self.mt)
        self.mt_pm = QtWidgets.QComboBox()
        for key, text in (("rr", tr("Basic round-robin")),
                          ("wrr", tr("Weighted round-robin (TCSchedule groups)")),
                          ("wrr2", tr("Weighted round-robin with throttles (WRR2)"))):
            self.mt_pm.addItem(text, key)
        self.mt_pm.setToolTip(tr("The 34K's policy manager (MD00534 13.2): which thread "
                                 "context the dispatch scheduler issues from"))
        form.addRow(tr("MT policy manager"), self.mt_pm)
        self.itc = QtWidgets.QComboBox()
        for key, text in (("auto", tr("4 FIFO + 8 semaphore cells")), ("none", tr("None")),
                          ("0:16", tr("16 semaphore cells")), ("16:0", tr("16 FIFO cells")),
                          ("16:16", tr("16 FIFO + 16 semaphore cells"))):
            self.itc.addItem(text, key)
        self.itc.setToolTip(tr("Inter-Thread Communication storage (MD00534 ch. 12): "
                               "gating storage cells whose loads and stores block a thread "
                               "until the cell is full / empty"))
        form.addRow(tr("ITC storage"), self.itc)
        self.itu = QtWidgets.QComboBox()
        for key, text in (("auto", tr("4 FIFO cells of 4 entries + 8 single-entry cells")),
                          ("none", tr("None")), ("16:16:4", tr("16 FIFO + 16 single-entry cells")),
                          ("8:24:8", tr("8 FIFO cells of 8 entries + 24 single-entry cells"))):
            self.itu.addItem(text, key)
        self.itu.setToolTip(tr("The ITU of a Release 6 core with VPs (the I6400, MD01196 "
                               "ch. 11): gating storage placed by CP0 SAAR, shared by the "
                               "core's VPs"))
        form.addRow(tr("ITU storage"), self.itu)
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
        self.mmu.addItem(tr("MPU, 16 regions (I7200, nanoMIPS)"), "mpu:16")
        self.mmu.addItem(tr("Fixed mapping + 16 MPU regions (I7200, nanoMIPS)"),
                         "fixed+mpu:16")
        form.addRow(tr("MMU"), self.mmu)
        form.addRow("", _note(tr("Every MMU maps memory like the fixed mapping after boot "
                                 "(kuseg + 1 GB): the TLBs through the boot firmware's "
                                 "refill handler, BAT through its six base-and-bounds "
                                 "entries.")))
        self.board = QtWidgets.QComboBox()
        self.board.addItem(tr("Flat memory (every address is RAM, the MARS layout)"), "flat")
        self.board.addItem(tr("MIPS Malta (CoreLV card, GT-64120)"), "malta")
        form.addRow(tr("Board"), self.board)
        self.ram = QtWidgets.QSpinBox()
        self.ram.setRange(1, 256)
        self.ram.setSuffix(" MB")
        self.ram.setValue(128)
        form.addRow(tr("Malta SDRAM"), self.ram)
        self.switches = QtWidgets.QSpinBox()
        self.switches.setRange(0, 255)
        self.switches.setDisplayIntegerBase(16)
        self.switches.setPrefix("0x")
        form.addRow(tr("Malta switches S2"), self.switches)
        self.mhz = QtWidgets.QSpinBox()
        self.mhz.setRange(1, 1000)
        self.mhz.setSuffix(" MHz")
        self.mhz.setValue(10)
        self.mhz.setToolTip(tr("The core's clock: the serial ports, the timer and the "
                               "real-time clock count in real time"))
        form.addRow(tr("Malta core clock"), self.mhz)
        #: the IDE disks (emips.ata): an image file or blank:SIZE per slot
        self.disks: dict[str, QtWidgets.QLineEdit] = {}
        for slot, label in (("pri0", N_("IDE primary master")),
                            ("pri1", N_("IDE primary slave")),
                            ("sec0", N_("IDE secondary master")),
                            ("sec1", N_("IDE secondary slave"))):
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText(tr("none — an image file or blank:SIZE (blank:8M)"))
            browse = QtWidgets.QToolButton()
            browse.setText("…")
            browse.setToolTip(tr("Choose a disk image file"))
            browse.clicked.connect(lambda _c=False, e=edit: self._browse_disk(e))
            row = QtWidgets.QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(browse)
            holder = QtWidgets.QWidget()
            holder.setLayout(row)
            row.setContentsMargins(0, 0, 0, 0)
            form.addRow(tr(label), holder)
            self.disks[slot] = edit
        form.addRow("", _note(tr("The disks are ATA drives on the PIIX4's IDE cables "
                                 "(ports 1F0h / 170h, IRQ 14 / 15). Every run starts from "
                                 "the image; the Board pane saves what a program wrote.")))
        self.board.currentIndexChanged.connect(self._board_changed)
        form.addRow("", _note(tr("The Malta: the GT-64120's memory map (SDRAM, the monitor "
                                 "flash, the board registers; anything else is a bus "
                                 "error), programs in kseg0 (.text at 0x80100000). The "
                                 "Board pane shows its LEDs and display.")))
        self.cores = QtWidgets.QSpinBox()
        self.cores.setRange(1, MAX_CORES)
        form.addRow(tr("Cores"), self.cores)
        self.vps = QtWidgets.QSpinBox()
        self.vps.setRange(1, 4)
        self.vps.setToolTip(tr("Release 6 virtual processors a core (DVP / EVP; the CPC "
                               "starts and stops them): more than one makes a CM3 system, "
                               "as the I6400's"))
        form.addRow(tr("Virtual processors a core"), self.vps)
        self.iocu = QtWidgets.QCheckBox(tr("I/O Coherence Unit (IOCU): the Malta's DMA is "
                                           "coherent with the data caches"))
        form.addRow("", self.iocu)
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
        self.relax = QtWidgets.QCheckBox(tr("nanoMIPS: link-time relaxation (ld --relax)"))
        lay.addWidget(self.relax)
        lay.addWidget(_note(tr("For a nanoMIPS program with .linkrelax: the linker shrinks "
                               "the branches, calls and $gp accesses that fit a shorter "
                               "form once every address is known.")))
        self.auto_assemble = QtWidgets.QCheckBox(tr("Assemble automatically when the "
                                                    "source changed (Run / Step)"))
        lay.addWidget(self.auto_assemble)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(tr("C programs: optimisation")))
        self.c_optimize = QtWidgets.QComboBox()
        for level, text in (("0", tr("-O0: none — every C line has its own code (default)")),
                            ("1", tr("-O1: moderate")), ("2", tr("-O2: more")),
                            ("3", tr("-O3: aggressive")), ("s", tr("-Os: small code"))):
            self.c_optimize.addItem(text, level)
        row.addWidget(self.c_optimize, 1)
        lay.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(tr("C programs: instructions")))
        self.c_isa = QtWidgets.QComboBox()
        for key, text in (("mips32", tr("MIPS32 (default)")),
                          ("mips16e", tr("MIPS16e (needs a MIPS GCC)")),
                          ("micromips", tr("microMIPS (Release 2 / 3 / 5)"))):
            self.c_isa.addItem(text, key)
        row.addWidget(self.c_isa, 1)
        lay.addLayout(row)
        lay.addWidget(_note(tr("Build (Ctrl+B) compiles a C program with clang or a MIPS "
                               "GCC for the selected release; -O0 keeps the code close "
                               "to the C, higher levels show what an optimiser does.")))
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
        self.arch.setCurrentIndex(max(0, self.arch.findData(o.arch)))
        self.nms.setChecked(o.nms)
        self.crc.setCurrentIndex(max(0, self.crc.findData(o.crc)))
        self.ginv.setCurrentIndex(max(0, self.ginv.findData(o.ginv)))
        self.release.setCurrentIndex(max(0, RELEASES.index(release)
                                         if release in RELEASES else 3))
        self.release_note.setText(html.escape(tr(RELEASE_NOTES[self.release.currentData()])))
        self.endian.setCurrentIndex(0 if endian == "big" else 1)
        self.forwarding.setChecked(forwarding)
        self.model.setCurrentIndex(max(0, self.model.findData(o.model)))
        self.compressed.setCurrentIndex(max(0, self.compressed.findData(o.compressed)))
        self.fpu.setChecked(o.fpu)
        self.dsp.setCurrentIndex(max(0, self.dsp.findData(o.dsp)))
        self.msa.setCurrentIndex(max(0, self.msa.findData(o.msa)))
        self.msa_wrp.setValue(o.msa_wrp)
        if self.mt.findData(o.mt) < 0:          # another VPES:TCS (the command line)
            self.mt.addItem(o.mt, o.mt)
        self.mt.setCurrentIndex(max(0, self.mt.findData(o.mt)))
        self.mt_pm.setCurrentIndex(max(0, self.mt_pm.findData(o.mt_pm)))
        if self.itc.findData(o.itc) < 0:        # another FIFOS:SEMAPHORES
            self.itc.addItem(o.itc, o.itc)
        self.itc.setCurrentIndex(max(0, self.itc.findData(o.itc)))
        if self.itu.findData(o.itu) < 0:        # another FIFOS:SINGLES[:DEPTH]
            self.itu.addItem(o.itu, o.itu)
        self.itu.setCurrentIndex(max(0, self.itu.findData(o.itu)))
        self.core.blockSignals(True)
        self.core.setCurrentIndex(max(0, self.core.findData(o.core)))
        self.core.blockSignals(False)
        self._core_text()
        self.c_isa.setCurrentIndex(max(0, self.c_isa.findData(o.c_isa)))
        self.cores.setValue(o.cores)
        self.vps.setValue(o.vps)
        self.iocu.setChecked(o.iocu)
        self.shadow_sets.setValue(o.shadow_sets)
        self.count_rate.setCurrentIndex(0 if o.count_rate == 1 else 1)
        parts = o.board.split(":")
        self.board.setCurrentIndex(max(0, self.board.findData(parts[0])))
        self.ram.setValue(int(parts[1], 0) if len(parts) > 1 and parts[1] else 128)
        self.switches.setValue(int(parts[2], 0) if len(parts) > 2 and parts[2] else 0)
        self.mhz.setValue(int(parts[3], 0) if len(parts) > 3 and parts[3] else 10)
        from ..malta import parse_disks
        try:
            disks = dict(parse_disks(o.disks))
        except ValueError:
            disks = {}
        for slot, edit in self.disks.items():
            edit.setText(disks.get(slot, ""))
        self._board_changed()
        spec = "tlb:32" if o.mmu == "tlb" else o.mmu
        if self.mmu.findData(spec) < 0:
            self.mmu.addItem(spec, spec)                 # e.g. from --mmu
        self.mmu.setCurrentIndex(self.mmu.findData(spec))
        for key, spin in self.latency.items():
            spin.setValue(o.latency.get(key, LATENCY[key]))
        self.pseudo.setChecked(o.pseudo)
        self.c_optimize.setCurrentIndex(max(0, self.c_optimize.findData(o.c_optimize)))
        self.reorder.setChecked(o.reorder)
        self.relax.setChecked(o.relax)
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
            relax=self.relax.isChecked(),
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
            vps=self.vps.value(), itu=self.itu.currentData(),
            iocu=self.iocu.isChecked(),
            shadow_sets=self.shadow_sets.value(), board=self._board_spec(),
            disks=self._disks_spec(), c_optimize=self.c_optimize.currentData(),
            compressed=self.compressed.currentData(), c_isa=self.c_isa.currentData(),
            fpu=self.fpu.isChecked(), dsp=self.dsp.currentData(), msa=self.msa.currentData(),
            msa_wrp=self.msa_wrp.value(), mt=self.mt.currentData(),
            mt_pm=self.mt_pm.currentData(), itc=self.itc.currentData(),
            core=self.core.currentData(), arch=self.arch.currentData(),
            nms=self.nms.isChecked(), crc=self.crc.currentData(),
            ginv=self.ginv.currentData(), rv_extensions=self._given.rv_extensions,
            rv_modes=self._given.rv_modes, rv_relax=self._given.rv_relax)
        return (self.release.currentData(), self.endian.currentData(),
                self.forwarding.isChecked(), o.validated())

    def _board_spec(self) -> str:
        if self.board.currentData() == "flat":
            return "flat"
        return f"malta:{self.ram.value()}:0x{self.switches.value():02x}:{self.mhz.value()}"

    def _disks_spec(self) -> str:
        if self.board.currentData() != "malta":
            return ""
        return ",".join(f"{slot}={e.text().strip()}" for slot, e in self.disks.items()
                        if e.text().strip())

    def _browse_disk(self, edit: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Disk image"), edit.text(), tr("Disk images (*.img *.bin *.raw);;"
                                                   "All files (*)"))
        if path:
            edit.setText(path)

    def _board_changed(self, _index: int = 0) -> None:
        malta = self.board.currentData() == "malta"
        self.ram.setEnabled(malta)
        self.switches.setEnabled(malta)
        self.mhz.setEnabled(malta)
        for e in self.disks.values():
            e.parentWidget().setEnabled(malta)

    def _arch_chosen(self, _index: int = 0) -> None:
        """nanoMIPS is Release 6 based (MD01247 ch. 1): choosing it chooses
        Release 6 (otherwise the release would send it back to MIPS32)."""
        if self.arch.currentData() == "nanomips" and self.release.currentData() != "R6":
            self.release.blockSignals(True)
            self.release.setCurrentIndex(self.release.findData("R6"))
            self.release.blockSignals(False)
            self.release_note.setText(html.escape(tr(RELEASE_NOTES["R6"])))
        elif self.arch.currentData() != "nanomips" and not self.fpu.isEnabled() \
                and getattr(core_preset(self.core.currentData()), "arch", "nanomips") \
                == "nanomips":                  # (no core, or the I7200: it goes too)
            self.fpu.blockSignals(True)         # (leaving nanoMIPS: the FPU it took)
            self.fpu.setChecked(True)
            self.fpu.blockSignals(False)

    def _constrain(self, *_args, note: bool = True) -> None:
        """Keep the settings together (emips.compat): adjust what depends
        on the release / board, disable the values that do not go with the
        rest (the reason as a tooltip), grey out what does not apply."""
        from ..compat import adjust, forbidden
        if self._constraining:
            return
        self._constraining = True
        try:
            release, _e, _f, o = self.values()
            fixed, notes = adjust(o, release)
            if notes:
                self.shadow_sets.setValue(fixed.shadow_sets)
                self.cores.setValue(fixed.cores)
                self.vps.setValue(fixed.vps)
                i = self.mmu.findData(fixed.mmu)
                if i < 0:
                    self.mmu.addItem(fixed.mmu, fixed.mmu)
                    i = self.mmu.findData(fixed.mmu)
                self.mmu.setCurrentIndex(i)
                for b in self.boot.buttons():
                    b.setChecked(b.property("key") == fixed.boot)
                self._set_cache(self.l3, fixed.l3)
                for box, value in ((self.compressed, fixed.compressed),
                                   (self.board, fixed.board.split(":")[0]),
                                   (self.c_isa, fixed.c_isa), (self.model, fixed.model),
                                   (self.dsp, fixed.dsp), (self.msa, fixed.msa),
                                   (self.mt, fixed.mt),
                                   (self.core, fixed.core), (self.arch, fixed.arch)):
                    box.setCurrentIndex(max(0, box.findData(value)))
                self._core_text()
            if note:
                self.compat_note.setText(
                    tr("Adjusted to go with the rest: {notes}").format(notes="; ".join(notes))
                    if notes else "")
            for box, field in ((self.mmu, "mmu"), (self.compressed, "compressed"),
                               (self.c_isa, "c_isa"), (self.model, "model"), (self.dsp, "dsp"),
                               (self.msa, "msa"), (self.mt, "mt"),
                               (self.release, "release"), (self.icache, "icache"),
                               (self.dcache, "dcache"), (self.l2, "l2"), (self.l3, "l3")):
                model = box.model()
                text = box.currentText() if box.isEditable() else None
                for i in range(box.count()):
                    why = forbidden(field, box.itemData(i), fixed, release)
                    model.item(i).setEnabled(why is None)
                    box.setItemData(i, why or "", QtCore.Qt.ItemDataRole.ToolTipRole)
                if text is not None and box.currentText() != text:
                    box.setEditText(text)       # (an item change resets the typed text)
            from ..cores import get as get_core
            core = get_core(fixed.core)
            for spin, field, top in ((self.cores, "cores", MAX_CORES), (self.vps, "vps", 4),
                                     (self.shadow_sets, "shadow_sets", 16)):
                if core is not None:            # the core's build options
                    allowed = getattr(core, field)
                    spin.setRange(min(allowed), max(allowed))
                    spin.setToolTip(tr("The {core} is built with {values}").format(
                        core=core.title, values=core.choices(field)))
                    continue
                why = forbidden(field, 2, fixed, release)
                spin.setRange(1, 1 if why else top)
                spin.setToolTip(why or "")
            self.fpu.setChecked(fixed.fpu)      # (nanoMIPS: none)
            why = forbidden("fpu", not fixed.fpu, fixed, release)
            self.fpu.setEnabled(why is None)
            self.fpu.setToolTip(why or self.fpu.toolTip())
            nano = fixed.arch == "nanomips"
            for w in (self.nms, self.crc, self.ginv):
                w.setEnabled(nano)
            self.iocu.setChecked(fixed.iocu)
            why = forbidden("iocu", True, fixed, release)
            self.iocu.setEnabled(why is None)
            self.iocu.setToolTip(why or "")
            for b in self.boot.buttons():
                why = forbidden("boot", b.property("key"), fixed, release)
                b.setEnabled(why is None)
                b.setToolTip(why or "")
            five = self.model.currentData() == "5-stage"
            self.forwarding.setEnabled(five)
            for spin in self.latency.values():
                spin.setEnabled(five)
            l1 = fixed.icache != "none" or fixed.dcache != "none"
            any_cache = l1 or fixed.l2 != "none" or fixed.write_buffer
            tlb = fixed.mmu.split("+")[0].split(":")[0] in ("tlb", "ftlb")
            self.cache_parity.setEnabled(l1)
            self.cache_ecc.setEnabled(fixed.l2 != "none")
            self.l3.setEnabled(fixed.l2 != "none")
            self.nonblocking.setEnabled(any_cache)
            for spin in self.mem_timing.values():   # the caches' and the micro-TLBs' waits
                spin.setEnabled(any_cache or tlb)
        finally:
            self._constraining = False

    def _core_chosen(self, _index: int = 0) -> None:
        """A core preset: the release and its build (emips.cores.apply)."""
        from ..cores import apply
        if self._constraining:
            return
        release, endian, forwarding, o = self.values()
        o, core_release = apply(o, self.core.currentData())
        self._constraining = True
        try:
            self.set_values(core_release or release, endian, forwarding, o)
        finally:
            self._constraining = False
        self._constrain()

    def _core_text(self) -> None:
        from ..cores import get as get_core
        core = get_core(self.core.currentData())
        if core is None:
            self.core_note.setText(html.escape(tr(
                "Any combination eMIPS can model; PRId 0 (no licensed core).")))
            return
        lines = [tr("{arch} Release {r}, PRId 0x{p:08x}; the settings below offer what "
                    "its manuals allow (the others are greyed out).").format(
                        arch=core.arch.upper(), r=core.release[1:], p=core.prid)]
        if core.missing:
            lines.append(tr("Not modelled: {what}.").format(what="; ".join(core.missing)))
        lines += list(core.notes)
        self.core_note.setText(html.escape(" ".join(lines)))

    def restore_defaults(self) -> None:
        self.set_values("R5", "big", True, Options())

    def apply(self) -> None:
        self.applied.emit(*self.values())

    def accept(self) -> None:
        self.apply()
        super().accept()
