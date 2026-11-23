"""A bitmap display: MARS's / RARS's "Bitmap Display" tool as a device
(PLAN M27 phase S2; MARS 4.5's ``mars/tools/BitmapDisplay.java``).

The display shows a rectangle of memory as pixels: one 32-bit word per
pixel, its low 24 bits ``0x00RRGGBB`` (the top byte ignored, as Java's
``new Color(int)``), row-major from a base address.  Its size is given in
screen pixels (MARS's default 512 x 256) and a unit size (each memory
pixel drawn as a block of unit-width x unit-height screen pixels, 1-32);
the memory it shows is ``(height / unit_h)`` rows of ``(width / unit_w)``
words.  MARS's base addresses keep their names: ``data`` 0x10000000 (the
global data), ``gp`` 0x10008000, ``static`` 0x10010000 (the default),
``heap`` 0x10040000, ``mmio`` 0xFFFF0000; any other address may be given.

Real MIPS defines no display.  eMIPS's is a device that **scans out
physical memory below the caches**, as a real frame buffer's DMA does
(the Malta's bus masters, Lab 33): the base is a virtual address, read
through the MMU's side-effect-free view a page at a time, and the words
come from ``Machine.mem`` — a pixel still dirty in a write-back D-cache
does not appear until it is written back (CACHE, a flush, or drawing
through kseg1 / an uncached attribute).  MARS repaints a cell when the
program writes its word; a scan-out also shows what memory held before
(the ``.data`` a program starts with) — ``sim.DEVIATIONS["bitmap
display"]``.  Unwritten memory reads 0: black.  Reading a frame changes
nothing (no access is counted, no cache line moves).

The control registers (PLAN M27 phase S6; eMIPS's own — MARS has none):
eight words at :data:`REGS` (0xFFFF0020, next to the MMIO console; kseg3 on
MIPS: kernel mode), clocked by the cycles of the core that owns them (the
console's: core 0):

====  =========  ========================================================
0x00  CONTROL    bit 0 ON (reset 1: the picture is shown; 0: black),
                 bit 1 IE (reset 0: the vertical-blank interrupt)
0x04  STATUS     bit 0 VBLANK: set at every vertical blank, cleared by
                 writing 1 (the interrupt is VBLANK and IE: a level)
0x08  BASE       the next frame's base: a write takes effect at the next
                 vertical blank (double buffering: draw one buffer, show
                 the other, swap without tearing); reads what was written
0x0C  FRONT      the base being shown (read-only)
0x10  FRAME      the number of vertical blanks so far (read-only)
0x14  PERIOD     cycles from one vertical blank to the next (reset
                 :data:`PERIOD` = 16667: 60 Hz on the 1 MHz clock of the
                 time services; at least :data:`MIN_PERIOD`)
0x18  COLUMNS    the words of a row (read-only)
0x1C  ROWS       the rows (read-only)
====  =========  ========================================================

The interrupt goes where the console's go (``Session._display_irq``):
hardware line 2 (Cause.IP4) of a single core, GIC source :data:`GIC_SOURCE`
(routed to core 0's pin 2 by the boot firmware when there is a display) on a CPS, a RISC-V hart's
external interrupt (MEIP, shared with the console), a platform's PLIC /
APLIC source 3, the P8700's APLIC source 8; the Malta's lines belong to its
own devices — there the register works, the interrupt is not wired.  Every
reset starts from the reset values, so a run replays exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import struct
import zlib

REGS = 0xFFFF0020                           # the control registers (8 words)
PERIOD, MIN_PERIOD = 16667, 100             # cycles a frame (60 Hz at 1 MHz)
IRQ_LINE = 2                                # a single MIPS core: HW2 = Cause.IP4
GIC_SOURCE = 4                              # a CPS's GIC
RV_SOURCE, P8700_SOURCE = 3, 8              # a RISC-V platform's / the P8700's APLIC
CONTROL, STATUS, BASE, FRONT, FRAME, PERIOD_REG, COLUMNS, ROWS = range(8)
REG_NAMES = ("CONTROL", "STATUS", "BASE", "FRONT", "FRAME", "PERIOD", "COLUMNS", "ROWS")

#: MARS's base address choices (BitmapDisplay.initializeDisplayBaseChoices)
BASES = {"data": 0x10000000, "gp": 0x10008000, "static": 0x10010000,
         "heap": 0x10040000, "mmio": 0xFFFF0000}
BASE_NAMES = {v: k for k, v in BASES.items()}
UNITS = (1, 2, 4, 8, 16, 32)                # MARS's unit sizes
MAX_SIZE = 2048                             # (MARS offers 64-1024)
PAGE = 1 << 10                              # (the smallest MIPS page: 1 KB)

_SPEC = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*(?::\s*(\d+)(?:\s*x\s*(\d+))?)?\s*"
                   r"(?:@\s*(\S+))?\s*$", re.I)


@dataclass(frozen=True)
class DisplaySpec:
    """``WxH[:U[xV]][@BASE]``: the display's size in screen pixels, the
    unit width / height, the base (an address or a MARS name)."""
    width: int = 512
    height: int = 256
    unit_w: int = 1
    unit_h: int = 1
    base: int = BASES["static"]

    @property
    def columns(self) -> int:
        return self.width // self.unit_w

    @property
    def rows(self) -> int:
        return self.height // self.unit_h

    @property
    def words(self) -> int:
        return self.rows * self.columns

    def address(self, column: int, row: int) -> int:
        """The address of the word drawn at (column, row)."""
        return (self.base + 4 * (row * self.columns + column)) & 0xFFFFFFFF

    def text(self) -> str:
        unit = f":{self.unit_w}" if self.unit_w == self.unit_h else \
            f":{self.unit_w}x{self.unit_h}"
        base = BASE_NAMES.get(self.base, f"0x{self.base:08x}")
        return f"{self.width}x{self.height}{'' if unit == ':1' else unit}@{base}"

    def describe(self) -> str:
        name = BASE_NAMES.get(self.base)
        return (f"{self.width} x {self.height} pixels, unit {self.unit_w} x {self.unit_h}: "
                f"{self.columns} x {self.rows} words from 0x{self.base:08x}"
                + (f" ({name})" if name else "") + f" to 0x{self.base + 4 * self.words - 1:08x}")


def parse(text: str) -> DisplaySpec | None:
    """``""`` / ``none`` / ``off``: no display; else a :class:`DisplaySpec`
    (ValueError with the reason)."""
    if text is None or text.strip().lower() in ("", "none", "off", "no"):
        return None
    m = _SPEC.match(text)
    if not m:
        raise ValueError(f"display {text!r}: WIDTHxHEIGHT[:UNIT[xUNIT]][@BASE], e.g. "
                         "512x256:8@static")
    w, h = int(m.group(1)), int(m.group(2))
    uw = int(m.group(3) or 1)
    uh = int(m.group(4) or uw)
    if not (1 <= w <= MAX_SIZE and 1 <= h <= MAX_SIZE):
        raise ValueError(f"display {text!r}: width and height 1-{MAX_SIZE}")
    for u in (uw, uh):
        if u not in UNITS:
            raise ValueError(f"display {text!r}: a unit size is one of "
                             + ", ".join(map(str, UNITS)))
    if w < uw or h < uh:
        raise ValueError(f"display {text!r}: smaller than one unit")
    b = (m.group(5) or "static").lower()
    if b in BASES:
        base = BASES[b]
    else:
        try:
            base = int(b, 0)
        except ValueError:
            raise ValueError(f"display {text!r}: the base is an address or one of "
                             + ", ".join(BASES)) from None
        if base % 4 or not 0 <= base <= 0xFFFFFFFF:
            raise ValueError(f"display {text!r}: the base must be a word-aligned "
                             "32-bit address")
    return DisplaySpec(w, h, uw, uh, base)


@dataclass
class Frame:
    """One scan-out: ``rows`` x ``columns`` pixels as RGB bytes (row-major,
    3 bytes a pixel), the words that came from memory and the pages the MMU
    did not map (black)."""
    spec: DisplaySpec
    rgb: bytes
    written: int                              # pixels whose word was ever written
    unmapped: int                             # pixels on pages nothing maps
    base: int = 0                             # the base it was read from (FRONT)
    on: bool = True                           # CONTROL.ON (off: black)

    def pixel(self, column: int, row: int) -> int:
        i = 3 * (row * self.spec.columns + column)
        return int.from_bytes(self.rgb[i:i + 3], "big")

    def png(self, scale: bool = True) -> bytes:
        """The frame as a PNG (zlib: no image library); ``scale``: each
        memory pixel as its unit-size block, as the display draws it."""
        sp = self.spec
        uw, uh = (sp.unit_w, sp.unit_h) if scale else (1, 1)
        width, height = sp.columns * uw, sp.rows * uh
        raw = bytearray()
        for r in range(sp.rows):
            line = self.rgb[3 * r * sp.columns: 3 * (r + 1) * sp.columns]
            if uw > 1:
                line = b"".join(line[i:i + 3] * uw for i in range(0, len(line), 3))
            raw += (b"\0" + line) * uh
        def chunk(kind: bytes, data: bytes) -> bytes:
            return (struct.pack(">I", len(data)) + kind + data
                    + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
        return (b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
                + chunk(b"IEND", b""))


class BitmapDisplay:
    """The device: :meth:`scan` reads a frame from a Machine's physical
    memory (no side effects); :attr:`frames` counts the scans."""

    def __init__(self, spec: DisplaySpec, irq=None) -> None:
        self.spec = spec
        self.frames = 0                         # scans (views)
        #: ``irq(level)``: the vertical-blank interrupt (the Session routes it)
        self.irq = irq
        self.owner = None                       # the core whose clock runs it
        self.on, self.ie, self.vblank = True, False, False
        self.front = self.pending = spec.base
        self.frame = 0                          # vertical blanks so far
        self.period = PERIOD
        self.now = 0                            # cycles since reset (the owner's)
        self.next = PERIOD                      # the next vertical blank
        self._level = False

    # -- the bus ------------------------------------------------------------------

    def attach(self, machine, owner: bool = True) -> "BitmapDisplay":
        """On ``machine``'s bus (first: before a board's catch-all);
        ``owner``: its clock runs the display (core 0, also when the CPC
        starts it again)."""
        machine.devices.insert(0, self)
        if owner:
            self.owner = machine
        return self

    def covers(self, addr: int) -> bool:
        return REGS <= addr < REGS + 32

    def register(self, i: int) -> int:
        sp = self.spec
        return (int(self.on) | int(self.ie) << 1 if i == CONTROL else
                int(self.vblank) if i == STATUS else
                self.pending if i == BASE else self.front if i == FRONT else
                self.frame & 0xFFFFFFFF if i == FRAME else self.period if i == PERIOD_REG else
                sp.columns if i == COLUMNS else sp.rows)

    def _write(self, i: int, value: int) -> None:
        if i == CONTROL:
            self.on, self.ie = bool(value & 1), bool(value & 2)
        elif i == STATUS:
            if value & 1:
                self.vblank = False
        elif i == BASE:
            self.pending = value & 0xFFFFFFFC
        elif i == PERIOD_REG:
            self.period = max(value, MIN_PERIOD)
            self.next = self.now + self.period
        self._update()

    def load(self, machine, addr: int, size: int) -> int:
        out = 0
        for k in range(size):
            a = addr + k
            word = self.register((a - REGS) >> 2)
            shift = 8 * (3 - (a & 3)) if machine.big_endian else 8 * (a & 3)
            b = word >> shift & 0xFF
            out = (out << 8) | b if machine.big_endian else out | b << (8 * k)
        return out

    def store(self, machine, addr: int, value: int, size: int) -> None:
        words: dict[int, int] = {}
        for k in range(size):
            a = addr + k
            i = (a - REGS) >> 2
            shift = 8 * (size - 1 - k) if machine.big_endian else 8 * k
            b = value >> shift & 0xFF
            w = words.get(i, self.register(i))
            at = 8 * (3 - (a & 3)) if machine.big_endian else 8 * (a & 3)
            words[i] = w & ~(0xFF << at) | b << at
        for i, w in words.items():
            if i in (CONTROL, STATUS, BASE, PERIOD_REG):
                self._write(i, w)

    def tick(self, machine, cycles: int) -> None:
        if machine is not self.owner:
            return                              # one clock: the owner's
        self.now += cycles
        if self.now < self.next:
            return
        while self.now >= self.next:            # (a long sleep: several blanks)
            self.next += self.period
            self.frame += 1
        self.front = self.pending               # the new base, between frames
        self.vblank = True
        self._update()

    def _update(self) -> None:
        level = self.vblank and self.ie
        if level != self._level:
            self._level = level
            if self.irq is not None:
                self.irq(level)

    def lines(self) -> list[str]:
        """The control registers (the debugger's ``display``)."""
        sp = self.spec
        out = [f"  CONTROL: {'on' if self.on else 'off'}, vertical-blank interrupt "
               f"{'enabled' if self.ie else 'disabled'}"
               f"{', pending' if self.vblank and self.ie else ''}   "
               f"STATUS.VBLANK {int(self.vblank)}",
               f"  FRONT 0x{self.front:08x}"
               + (f", BASE 0x{self.pending:08x} from the next frame"
                  if self.pending != self.front else "")
               + f"   FRAME {self.frame}   PERIOD {self.period} cycles "
               f"({1_000_000 / self.period:.1f} Hz at 1 MHz), the next in "
               f"{self.next - self.now}",
               f"  registers at 0x{REGS:08x}: " + ", ".join(
                   f"{n} +0x{4 * i:x}" for i, n in enumerate(REG_NAMES))]
        return out

    def physical(self, m, va: int) -> int | None:
        """The physical address of *va* (the MMU's view), or None."""
        try:
            t = m.translate(va)
        except Exception:                       # (an address in no segment ...)
            return None
        return t.pa

    def scan(self, m) -> Frame:
        sp = self.spec
        n = sp.words
        big = getattr(m, "big_endian", False)
        mem = m.mem.get_memory()                # address (hex) -> byte (hex): below the caches
        pamask = getattr(m, "pamask", 0xFFFFFFFF)
        # the frame's physical pieces: (pa_lo, pa_hi, the word index of pa_lo)
        pieces, unmapped = [], 0
        if not self.on:                         # CONTROL.ON = 0: black
            self.frames += 1
            return Frame(sp, bytes(3 * n), 0, 0, self.front, False)
        first, end = self.front, self.front + 4 * n
        page = first & ~(PAGE - 1)
        while page < end:
            lo, hi = max(page, first), min(page + PAGE, end)
            pa = self.physical(m, lo)
            if pa is None:
                unmapped += (hi - lo) // 4
            else:
                pieces.append((pa & pamask, (pa & pamask) + hi - lo, (lo - first) // 4))
            page += PAGE
        words: dict[int, bytearray] = {}
        if len(mem) < 4 * n:                    # few bytes written: walk them
            by_page = {}
            for piece in pieces:
                for pg in range(piece[0] >> 10, ((piece[1] - 1) >> 10) + 1):
                    by_page.setdefault(pg, []).append(piece)
            for key, val in mem.items():
                a = int(key, 16)
                for lo, hi, i0 in by_page.get(a >> 10, ()):
                    if lo <= a < hi:
                        w = words.get(i0 + (a - lo) // 4)
                        if w is None:
                            w = words[i0 + (a - lo) // 4] = bytearray(4)
                        w[(a - lo) % 4] = int(val, 16)
                        break
        else:                                   # a full screen: look each word up
            for lo, hi, i0 in pieces:
                for k in range((hi - lo) // 4):
                    a = lo + 4 * k
                    b = [mem.get(format((a + j) & pamask, "x")) for j in range(4)]
                    if b != [None, None, None, None]:
                        words[i0 + k] = bytearray(int(x, 16) if x is not None else 0
                                                  for x in b)
        out = bytearray(3 * n)
        for i, w in words.items():              # 0x00RRGGBB in the core's byte order
            o = 3 * i
            out[o:o + 3] = w[1:4] if big else bytes((w[2], w[1], w[0]))
        self.frames += 1
        return Frame(sp, bytes(out), len(words), unmapped, self.front)

    def word_at(self, m, column: int, row: int) -> tuple[int, int | None, int]:
        """(virtual address, physical address or None, the word in memory)."""
        va = self.front + 4 * (row * self.spec.columns + column)
        pa = self.physical(m, va)
        if pa is None:
            return va, None, 0
        mem = m.mem.get_memory()
        pamask = getattr(m, "pamask", 0xFFFFFFFF)
        b = [int(mem.get(format((pa + j) & pamask, "x"), "0"), 16) for j in range(4)]
        v = int.from_bytes(bytes(b), "big" if getattr(m, "big_endian", False) else "little")
        return va, pa, v
