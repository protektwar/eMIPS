"""The P8700's Debug Unit and a probe (LF16; MD01502 rev. 1.84 ch. 13 the
OCI, 5.1 / Table 5.1 the register ring bus and its IDs, Table 5.8 the
DbgRead / DbgWrite commands, 5.14.1.18 DBU_REV, 5.14.4 / 5.14.5 the CPC's
DBU domain and ROCC_CTL.DBU_ROCC; the product brief: "the single TAP
controller in the cluster ... a RAM to hold instructions and data accessed
by the cores while in debug mode"; MIPS EJTAG MD00047 ch. 4 for the TAP).

:class:`DBU` — one per cluster (``Cluster.dbu``):

- **The RAM** (dmxseg, 13.1.1.4; :data:`RAM_SIZE` — "from configuration",
  assumed): what a hart in Debug Mode sees at dmseg while the probe's
  ProbEn is set (emips.p8700dbg), the probe's monitor and mailboxes.
- **The TAP** (:class:`DBUTap`, 13.1.1.2: the cluster's single TAP; an
  IEEE 1149.1 controller — :class:`emips.tap.TAP`'s state machine — with
  EJTAG's instruction numbers): IDCODE (the JEDEC code mvendorid names),
  IMPCODE (EJTAG 5, 64-bit, DINTsup, 8-bit ASIDs, NoDMA), ADDRESS (64 bits),
  DATA (64 bits), CONTROL, ALL, FDC (emips.p8700trace), BYPASS.  CONTROL is
  EJTAG's ECR where its bits apply: Rocc (the DBU was reset: written 0 to
  acknowledge, updates are ignored until then), Psz (1 / 2 / 4 / 8 bytes),
  PRnW, PrAcc (written 1: the DBU performs the access now and clears it),
  ProbEn, ProbTrap, EjtagBrk (written 1: a DINT to every hart not in Debug
  Mode), DM (a hart is in Debug Mode).
- **The register-bus master** (ring ID 0x23, DbgRead / DbgWrite /
  RegRead / RegWrite): ADDRESS = dest_id << 56 | offset (Table 5.1's IDs;
  the packet format is not published — this layout is eMIPS's): a core
  (0-5: offset = hart << 24 | a drseg offset, or 0x100000 + 8 x a CSR
  number — the hart's CSRs, "DMXSEG, DRSEG and CSR"), the interrupt
  controller (0x18) / CPC (0x21) / GCR (0x22) — offsets in the GCR block —,
  dmxseg (0x24 normal, 0x25 debug: the RAM).  Anything else has no
  destination: reads 0, writes are dropped, ``error`` is set.
- **The CPC's DBU domain** (0x9100 CMD, 0x9108 STAT_CONF: SEQ_STATE,
  EJTAG_PROBE — set when a probe first scans, RW0C —, CMD) and
  ROCC_CTL.DBU_ROCC (17: set at the cluster's reset).

:class:`Probe` drives only the TAP: loads :data:`MONITOR` (eMIPS's debug
monitor, ``emips/firmware/p8700_monitor.s``) into the RAM, sets ProbEn /
ProbTrap, halts a hart (CPC DBG_DBRK_REG through the master), and reads /
writes its registers and memory through the monitor's mailbox.  Choices:
``sim.DEVIATIONS["P8700 debug"]``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from . import tap as T
from .p8700dbg import DRSEG_BASE

M32, M64 = 0xFFFFFFFF, (1 << 64) - 1
RAM_SIZE = 64 * 1024
DBU_REVISION = 0x0100                    # (5.14.1.18: MAJOR 1, MINOR 0 — "from configuration")
#: Table 5.1's ring IDs the DBU's master reaches
RING_IC, RING_CPC, RING_GCR, RING_DBU = 0x18, 0x21, 0x22, 0x23
RING_DMXSEG, RING_DMXSEG_DEBUG = 0x24, 0x25
RING_NAMES = {**{c: f"core {c}" for c in range(6)}, RING_IC: "interrupt controller",
              RING_CPC: "CPC", RING_GCR: "GCR", RING_DMXSEG: "dmxseg",
              RING_DMXSEG_DEBUG: "dmxseg (debug)"}
#: a core's register-bus space: the hart, drseg, the CSRs
HART_SHIFT, CSR_BASE = 24, 0x100000
#: the CPC's DBU domain (5.14.5: 0x9100 / 0x9108)
DBU_CMD, DBU_STAT_CONF = 0x9100, 0x9108
EJTAG_PROBE = 1 << 15
#: the TAP's data registers (64-bit ADDRESS / DATA)
FDC = 0x17
DR_LENGTH = {T.IDCODE: 32, T.IMPCODE: 32, T.ADDRESS: 64, T.DATA: 64, T.CONTROL: 32,
             T.ALL: 160, FDC: 38}
INSTRUCTIONS = {T.IDCODE: "IDCODE", T.IMPCODE: "IMPCODE", T.ADDRESS: "ADDRESS",
                T.DATA: "DATA", T.CONTROL: "CONTROL", T.ALL: "ALL", FDC: "FDC",
                T.BYPASS: "BYPASS"}
#: IDCODE: Version / PartNumber not published (0), ManufID = mvendorid's
#: JEDEC code (7.1.5: 2 continuation codes, offset 0x27), bit 0 = 1
IDCODE_VALUE = 0x127 << 1 | 1
#: IMPCODE (MD00047 Table 4.4): EJTAGver 5, DINTsup, ASIDsize 2, NoDMA, MIPS64
IMPCODE_VALUE = 5 << 29 | 1 << 24 | 2 << 21 | 1 << 14 | 1
#: the monitor's mailboxes (p8700_monitor.s)
MAILBOX, MAILBOX_SIZE, VECTOR = 0x1000, 0x200, 0x200
MB_STATE, MB_COMMAND, MB_ADDRESS, MB_DATA, MB_DPC, MB_ERROR, MB_ENTRIES = (
    0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30)
MB_REGS = 0x100                          # (x1 at + 0x108 ... x31 at + 0x1F8)
CMD_RESUME, CMD_STEP, CMD_READ, CMD_WRITE = 1, 2, 3, 4


@lru_cache(maxsize=1)
def monitor_image() -> bytes:
    """The debug monitor's bytes from offset 0 (position independent)."""
    from .rvasm import assemble
    src = (Path(__file__).parent / "firmware" / "p8700_monitor.s").read_text()
    prog = assemble(src, arch="rv64", extensions="imac_zicsr")
    base = prog.symbols["entry"] - VECTOR
    end = prog.symbols["mailboxes"]
    mem = prog.image if isinstance(prog.image, dict) else {}
    out = bytearray(end - base)
    for a in range(base, end):
        out[a - base] = mem.get(a, 0)
    return bytes(out)


class DBUTap(T.TAP):
    """The DBU's TAP: :class:`emips.tap.TAP`'s controller, the DBU's data
    registers (see the module docstring)."""

    def __init__(self, dbu: "DBU") -> None:
        self.dbu = dbu
        super().__init__()

    def _ecr_reset(self) -> None:
        self.rocc = True
        self.prob_en = self.prob_trap = False
        self.address = self.data = 0
        self.psz = 3
        self.prnw = False
        self.pending = None

    def processor_reset(self, m) -> None:      # (no processor behind this TAP)
        pass

    def _dr_length(self) -> int:
        return DR_LENGTH.get(self.ir, 1)

    def impcode(self) -> int:
        return IMPCODE_VALUE

    def ecr(self) -> int:
        d = self.dbu
        v = (T.ECR_ROCC if self.rocc else 0) | self.psz << T.ECR_PSZ_SHIFT \
            | (T.ECR_PRNW if self.prnw else 0) | (T.ECR_PROBEN if self.prob_en else 0) \
            | (T.ECR_PROBTRAP if self.prob_trap else 0)
        if any(m.debug_mode for m in d.cl.machines):
            v |= T.ECR_DM
        return v

    def _capture(self) -> int:
        self.dbu.probe_seen()
        ir = self.ir
        if ir == T.IDCODE:
            return IDCODE_VALUE
        if ir == T.IMPCODE:
            return IMPCODE_VALUE
        if ir == T.ADDRESS:
            return self.address
        if ir == T.DATA:
            return self.data
        if ir == T.CONTROL:
            return self.ecr()
        if ir == T.ALL:
            return self.address << 96 | self.data << 32 | self.ecr()
        if ir == FDC:
            f = self.dbu.cl.fdc
            return f.tap_capture() if f is not None else 0
        return 0

    def _update_ir(self, value: int) -> None:
        self.ir = value

    def _update_dr(self, value: int) -> None:
        ir = self.ir
        if ir == T.ADDRESS:
            self.address = value & M64
        elif ir == T.DATA:
            self.data = value & M64
        elif ir == T.CONTROL:
            self._write_ecr(value & M32)
        elif ir == T.ALL:
            self.address = value >> 96 & M64
            self.data = value >> 32 & M64
            self._write_ecr(value & M32)
        elif ir == FDC:
            f = self.dbu.cl.fdc
            if f is not None:
                f.tap_update(value)

    def _write_ecr(self, value: int) -> None:
        """Rocc first (MD00047 Table 4.9: while it is 1 the rest is not
        updated); then the bits; PrAcc = 1 performs the access."""
        if self.rocc:
            if not value & T.ECR_ROCC:
                self.rocc = False
            return
        self.psz = value >> T.ECR_PSZ_SHIFT & 3
        self.prnw = bool(value & T.ECR_PRNW)
        self.prob_en = bool(value & T.ECR_PROBEN)
        self.prob_trap = bool(value & T.ECR_PROBTRAP)
        if value & T.ECR_EJTAGBRK:
            cl = self.dbu.cl
            for i in cl.running():
                cl.debug_interrupt(i)
        if value & T.ECR_PRACC:
            size = 1 << self.psz
            if self.prnw:
                self.dbu.master_write(self.address, self.data, size)
            else:
                self.data = self.dbu.master_read(self.address, size)


class DBU:
    """The cluster's Debug Unit (see the module docstring)."""

    #: the GCR's ACCESS gates cores and IOCUs, not the DBU's master
    is_dbu = True

    def __init__(self, cl: Any) -> None:
        self.cl = cl
        self.ram = bytearray(RAM_SIZE)
        self.size = RAM_SIZE
        self.tap = DBUTap(self)
        self.cmd = 3                            # (the DBU domain: powered up)
        self.ejtag_probe = False
        self.error = False
        #: the master's accesses (the views): (write, address, value, size)
        self.log: list[tuple] = []

    @property
    def prob_en(self) -> bool:
        return self.tap.prob_en

    @property
    def prob_trap(self) -> bool:
        return self.tap.prob_trap

    def probe_seen(self) -> None:
        self.ejtag_probe = True

    # -- the RAM (dmxseg) ------------------------------------------------------------

    def ram_read(self, off: int, size: int) -> int:
        return int.from_bytes(self.ram[off:off + size], "little")

    def ram_write(self, off: int, value: int, size: int) -> None:
        self.ram[off:off + size] = (value & ((1 << 8 * size) - 1)).to_bytes(size, "little")

    # -- the CPC's DBU domain ------------------------------------------------------------

    def cpc_get(self, off: int) -> int | None:
        if off == DBU_CMD:
            return self.cmd
        if off == DBU_STAT_CONF:                # (SEQ_STATE U5 / D0, EJTAG_PROBE, CMD)
            return (6 if self.cmd == 3 else 0) << 19 | (EJTAG_PROBE if self.ejtag_probe
                                                          else 0) | self.cmd
        return None

    def cpc_put(self, off: int, value: int, mask: int) -> bool:
        if off == DBU_CMD:
            if mask & 0xF and value & 0xF in (1, 2, 3, 4):
                if value & 0xF == 4:            # (Reset: ROCC_CTL.DBU_ROCC, the TAP's Rocc)
                    self.tap._ecr_reset()
                    self.cl.rocc |= 1 << 17
                self.cmd = 3 if value & 0xF == 4 else value & 0xF
            return True
        if off == DBU_STAT_CONF:
            if mask & EJTAG_PROBE and not value & EJTAG_PROBE:
                self.ejtag_probe = False        # (RW0C)
            return True
        return False

    # -- the register-bus master -----------------------------------------------------------

    def _target(self, address: int) -> tuple[str, Any, int]:
        dest, off = address >> 56 & 0x3F, address & ((1 << 56) - 1)
        cl = self.cl
        if dest < 6 and dest < cl.cores:
            h = off >> HART_SHIFT
            if h >= cl.harts:
                return "none", None, 0
            m = cl.machines[cl.index(dest, h)]
            rest = off & ((1 << HART_SHIFT) - 1)
            if rest >= CSR_BASE:
                return "csr", m, (rest - CSR_BASE) >> 3
            return "drseg", m, rest
        if dest in (RING_IC, RING_CPC, RING_GCR):
            if dest == RING_CPC and off < 0x8000:
                off += 0x8000                   # (a CPC offset: the CPC's own)
            return "gcr", None, off
        if dest in (RING_DMXSEG, RING_DMXSEG_DEBUG):
            return "ram", None, off
        return "none", None, 0

    def master_read(self, address: int, size: int = 8) -> int:
        kind, m, off = self._target(address)
        v = 0
        if kind == "drseg":
            v = m.hybrid.load(DRSEG_BASE + off, size)
        elif kind == "csr":
            v = m.csrs.peek(off) if hasattr(m.csrs, "peek") else m.csrs.read(off)
        elif kind == "gcr":
            v = self.cl.gcr.load(None, self.cl.base + off, size)
        elif kind == "ram" and off + size <= self.size:
            v = self.ram_read(off, size)
        else:
            self.error = True
        v &= (1 << 8 * size) - 1
        self.log.append((False, address, v, size))
        del self.log[:-32]
        return v

    def master_write(self, address: int, value: int, size: int = 8) -> None:
        kind, m, off = self._target(address)
        value &= (1 << 8 * size) - 1
        if kind == "drseg":
            m.hybrid.store(DRSEG_BASE + off, value, size)
        elif kind == "csr":
            m.csrs.write(off, value)
        elif kind == "gcr":
            self.cl.gcr.store(None, self.cl.base + off, value, size)
        elif kind == "ram" and off + size <= self.size:
            self.ram_write(off, value, size)
        else:
            self.error = True
        self.log.append((True, address, value, size))
        del self.log[:-32]

    # -- views ---------------------------------------------------------------------------

    def lines(self) -> list[str]:
        t = self.tap
        out = [f"Debug Unit (DBU_REV 0x{DBU_REVISION:04x}): TAP {t.state}, IR "
               f"{INSTRUCTIONS.get(t.ir, f'0x{t.ir:02x}')}, CONTROL 0x{t.ecr():08x} "
               f"(Rocc {int(t.rocc)}, ProbEn {int(t.prob_en)}, ProbTrap {int(t.prob_trap)}), "
               f"ADDRESS 0x{t.address:x}, DATA 0x{t.data:x}"
               + (", a master access had no destination" if self.error else "")]
        mon = self.ram[VECTOR:VECTOR + 4] != b"\0\0\0\0"
        out.append(f"  RAM {self.size // 1024} KB (dmxseg): "
                   + ("the monitor is loaded" if mon else "no monitor"))
        if mon:
            cl = self.cl
            for i, m in enumerate(cl.machines):
                c, h = divmod(i, cl.harts)
                b = MAILBOX + MAILBOX_SIZE * (2 * c + h)
                st = self.ram_read(b + MB_STATE, 8)
                n = self.ram_read(b + MB_ENTRIES, 8)
                if n or st:
                    out.append(f"  hart {i}: {'in the monitor' if st else 'running'}, "
                               f"entered {n} time(s), dpc 0x{self.ram_read(b + MB_DPC, 8):x}")
        for w, a, v, n in self.log[-4:]:
            dest = a >> 56 & 0x3F
            out.append(f"  {'write' if w else 'read '} {RING_NAMES.get(dest, f'ring 0x{dest:02x}')}"
                       f" +0x{a & ((1 << 56) - 1):x}: 0x{v:x} ({n} bytes)")
        return out


class Probe:
    """A JTAG probe on the DBU's TAP (see the module docstring); every
    operation is scans."""

    def __init__(self, dbu: DBU) -> None:
        self.dbu = dbu
        self.tap = dbu.tap

    # -- the TAP -----------------------------------------------------------------------

    def control(self, value: int) -> int:
        """Write CONTROL (Rocc acknowledged first), return what it read."""
        t = self.tap
        t.scan_ir(T.CONTROL)
        old = t.scan_dr(0)                      # (Capture: the current value)
        if old & T.ECR_ROCC:
            t.scan_dr(0)                        # (acknowledge Rocc: write it 0)
        t.scan_ir(T.CONTROL)
        return t.scan_dr(value)

    def _keep(self) -> int:
        t = self.tap
        return (T.ECR_PROBEN if t.prob_en else 0) | (T.ECR_PROBTRAP if t.prob_trap else 0)

    def attach(self, *, trap: bool = True) -> None:
        """ProbEn (dmseg is the DBU's RAM) and ProbTrap (the vector there)."""
        self.control(T.ECR_PROBEN | (T.ECR_PROBTRAP if trap else 0))

    def detach(self) -> None:
        self.control(0)

    def read(self, address: int, size: int = 8) -> int:
        t = self.tap
        t.scan_ir(T.ADDRESS)
        t.scan_dr(address)
        psz = size.bit_length() - 1
        self.control(self._keep() | psz << T.ECR_PSZ_SHIFT | T.ECR_PRACC)
        t.scan_ir(T.DATA)
        return t.scan_dr(0)

    def write(self, address: int, value: int, size: int = 8) -> None:
        t = self.tap
        t.scan_ir(T.ADDRESS)
        t.scan_dr(address)
        t.scan_ir(T.DATA)
        t.scan_dr(value)
        psz = size.bit_length() - 1
        self.control(self._keep() | psz << T.ECR_PSZ_SHIFT | T.ECR_PRNW | T.ECR_PRACC)

    # -- the monitor ---------------------------------------------------------------------

    def load_monitor(self) -> None:
        img = monitor_image()
        for k in range(0, len(img), 8):
            self.write(RING_DMXSEG << 56 | k, int.from_bytes(img[k:k + 8], "little"))

    def _slot(self, i: int) -> int:
        cl = self.dbu.cl
        c, h = divmod(i, cl.harts)
        return MAILBOX + MAILBOX_SIZE * (2 * c + h)

    def mailbox(self, i: int, field: int) -> int:
        return self.read(RING_DMXSEG << 56 | self._slot(i) + field)

    def halt(self, i: int) -> None:
        """A DINT to hart ``i``: CPC DBG_DBRK_REG of its core."""
        cl = self.dbu.cl
        c, h = divmod(i, cl.harts)
        self.write(RING_CPC << 56 | 0xA040 + 0x100 * c, 1 << h)

    def halted(self, i: int) -> bool:
        """Hart ``i`` waits in the monitor."""
        return self.mailbox(i, MB_STATE) == 1 and self.mailbox(i, MB_COMMAND) == 0

    def command(self, i: int, cmd: int, address: int = 0, data: int = 0) -> None:
        b = RING_DMXSEG << 56 | self._slot(i)
        if cmd in (CMD_READ, CMD_WRITE):
            self.write(b + MB_ADDRESS, address)
            self.write(b + MB_DATA, data)
        self.write(b + MB_COMMAND, cmd)

    def register(self, i: int, n: int) -> int:
        """x``n`` of a halted hart (as it left the program)."""
        return 0 if n == 0 else self.mailbox(i, MB_REGS + 8 * n)

    def set_register(self, i: int, n: int, value: int) -> None:
        if n:
            self.write(RING_DMXSEG << 56 | self._slot(i) + MB_REGS + 8 * n, value)

    def pc(self, i: int) -> int:
        return self.mailbox(i, MB_DPC)

    def set_pc(self, i: int, value: int) -> None:
        self.write(RING_DMXSEG << 56 | self._slot(i) + MB_DPC, value)

    def csr(self, i: int, num: int) -> int:
        cl = self.dbu.cl
        c, h = divmod(i, cl.harts)
        return self.read(c << 56 | h << HART_SHIFT | CSR_BASE + 8 * num)

    def drseg(self, i: int, off: int, value: int | None = None) -> int:
        """Read (or write) hart ``i``'s drseg register at ``off``."""
        cl = self.dbu.cl
        c, h = divmod(i, cl.harts)
        a = c << 56 | h << HART_SHIFT | off
        if value is not None:
            self.write(a, value)
            return value
        return self.read(a)
