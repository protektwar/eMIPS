"""The eMIPS boot monitor for the Malta (PLAN M15 phase 5).

``firmware/monitor.asm`` is a small YAMON-compatible monitor — real MIPS
code in the Malta's monitor flash: the reset code sets the board up and
starts the application in YAMON's application context; the YAMON
function table at 0x1FC00500 and the exception handler serve it
(MD00008, YAMON User's Manual rev. 02.19: chapter 5; the function
offsets and prototypes of ``yamon_api.h`` as the NetBSD / FreeBSD /
RTEMS copies have them).  :func:`image` builds it for a board (YAMON is
built per board: its SDRAM, clock, byte order, release), :func:`load`
puts it in the flash, :func:`describe_application` writes what YAMON's
``load`` and ``go`` know about the application (the descriptor, argv,
the environment) and :func:`start_state` gives the registers ``go``
leaves — what the kernel boot (eMIPS's firmware in Python) sets too.

A nanoMIPS CPU on the Malta (PLAN M23 phase 4d; QEMU's Malta runs the
I7200 with a nanoMIPS boot loader of its own) gets the same monitor
assembled as nanoMIPS code (:func:`_nano_text`, monitor.asm's header):
the same board initialisation, function table, exception handling and
application start, its functions with the p32 convention.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from . import cp0 as C
from . import gt64120 as GT

SOURCE = Path(__file__).resolve().parent / "firmware" / "monitor.asm"
#: the function table (yamon_api.h: YAMON_FUNCTION_BASE) and its entries
FUNCTION_BASE = 0x1FC00500
FUNCTIONS = {"print_count": 0x04, "exit": 0x20, "flush_cache": 0x2C, "print": 0x34,
             "register_cpu_isr": 0x38, "deregister_cpu_isr": 0x3C,
             "register_ic_isr": 0x40, "deregister_ic_isr": 0x44,
             "register_esr": 0x48, "deregister_esr": 0x4C, "getchar": 0x50,
             "syscon_read": 0x54}
#: the monitor's RAM (kseg0): the application descriptor and its strings
DESC = 0x8000C000
STRINGS = 0x8000C400
#: its tables, its flags, an exception's saved registers (a 64-bit CPU's
#: frame of doublewords: above the strings), the exception stack
CPU_ISR, IC_ISR, ESR, MON_FLAGS = 0x8000C100, 0x8000C180, 0x8000C210, 0x8000C2A0
FRAME, FRAME64, EXC_STACK = 0x8000C300, 0x8000C600, 0x8000FFF0
#: YAMON_DEFAULT_HANDLER (yamon_api.h): the "default" ISR / ESR number
DEFAULT_ID = 0xFFFFFFF0
#: descriptor flags: the application has its own exception handler /
#: TLB refill handler (the monitor leaves the vectors alone)
OWN_HANDLER, OWN_REFILL = 1, 2
#: where the monitor's code starts (after the function table)
RESET = 0xBFC00000
#: the labels at the monitor's fixed addresses (checked in the nanoMIPS
#: build, which reaches them with .balign)
NANO_FIXED = {"reset": 0xBFC00000, "bev_refill": 0xBFC00200, "bev_cache_error": 0xBFC00300,
              "bev_general": 0xBFC00380, "bev_interrupt": 0xBFC00400,
              "bev_debug": 0xBFC00480, "functions": 0xBFC00500, "start": 0xBFC00600}
#: the bus and PCI clocks syscon_read reports (the Malta's 33.333 MHz PCI)
BUS_HZ, PCI_KHZ = 33_333_333, 33_333


def _swap(v: int) -> int:
    return int.from_bytes((v & 0xFFFFFFFF).to_bytes(4, "big"), "little")


def _cfg(bus: int, dev: int, fn: int, reg: int) -> int:
    return 0x80000000 | bus << 16 | dev << 11 | fn << 8 | reg


def _preprocess(text: str, values: dict[str, str], flags: set[str]) -> str:
    out, keep = [], [True]
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#@if "):
            name = s[5:].strip()
            if not name.isidentifier():         # (a comment there would drop the block)
                raise ValueError(f"monitor.asm: '#@if {name}': one flag name only")
            keep.append(keep[-1] and name in flags)
            continue
        if s == "#@else":
            top = keep.pop()
            keep.append(keep[-1] and not top)
            continue
        if s == "#@endif":
            keep.pop()
            continue
        if not keep[-1]:
            continue
        for k, v in values.items():
            line = line.replace(f"@{k}@", v)
        out.append(line)
    return "\n".join(out) + "\n"


#: the o32 temporaries as register numbers (the 64-bit build: emips's
#: MIPS64 assembler takes n64 names, in which $t0-$t3 are $12-$15)
_O32_TEMPS = re.compile(r"\$t([0-9])(?![0-9\w])")


def _o32_numbers(text: str) -> str:
    return _O32_TEMPS.sub(lambda m: f"${(8 + int(m.group(1))) if m.group(1) < '8' else 16 + int(m.group(1))}",
                          text)


#: the nanoMIPS build: the o32 temporaries and $v0 / $v1 as p32 numbers
#: ($rN: DN00179 3.1 makes the names mandatory), any other $N GPR too
_GPR_NUMBER = re.compile(r"\$(t[0-9]|v[01]|[0-9]{1,2})(?![0-9\w])")
_O32 = {**{f"t{i}": 8 + i for i in range(8)}, "t8": 24, "t9": 25, "v0": 2, "v1": 3}
#: nanoMIPS's LUI takes the immediate as bits 31:12 (MD01247 LUI): MIPS's
#: "lui rt, 0xNNNN" is "li rt, 0xNNNN0000" there
_LUI = re.compile(r"^(\s*(?:\w+:\s*)?)lui(\s+)(\$\w+),\s*(0x[0-9a-fA-F]{1,4})\b")
#: the monitor's fixed addresses: ".text ADDRESS" (a subsection number in
#: GNU as's nanoMIPS port) -> an alignment that reaches it
_TEXT_AT = re.compile(r"^(\s*)\.text\s+0xbfc0([0-9a-f]{4})\b")


def _nano_text(text: str) -> str:
    """monitor.asm (preprocessed) for the nanoMIPS assembler."""
    out = []
    for line in text.splitlines():
        m = _TEXT_AT.match(line)
        if m:
            off = int(m.group(2), 16)
            line = (f"{m.group(1)}.text" if off == 0 else
                    f"{m.group(1)}.balign 0x{off & -off:x}, 0   # 0xbfc0{m.group(2)}")
            out.append(line)
            continue
        line = _LUI.sub(lambda m: f"{m.group(1)}li{m.group(2)}{m.group(3)}, "
                                  f"0x{int(m.group(4), 16) << 16:08x}", line)
        code, hash_, comment = line.partition("#")
        mn = code.split(":")[-1].split()
        if mn and mn[0] in ("mfc0", "mtc0"):    # (only rt: the CP0 register stays)
            head, comma, rest = code.partition(",")
            code = _GPR_NUMBER.sub(_rn, head) + comma + rest
        else:
            code = _GPR_NUMBER.sub(_rn, code)
        out.append(code + hash_ + comment)
    return "\n".join(out) + "\n"


def _rn(m: re.Match) -> str:
    name = m.group(1)
    return f"$r{_O32.get(name, name)}"


def source(*, big_endian: bool, release: str, ram: int, cpu_hz: int, cps: bool,
           arch: str = "mips32") -> str:
    """The monitor's source for a board (``arch`` "mips64": the 64-bit
    CPU's, "nanomips": a nanoMIPS CPU's; see monitor.asm)."""
    from .malta import BMIBA, MaltaBoard
    PIIX4_DEVICE = MaltaBoard.PIIX4_DEVICE
    from .malta import user_window
    from .tlb import firmware_refill, firmware_xtlb_refill
    wide = arch == "mips64"
    nano = arch == "nanomips"
    le = _swap if big_endian else (lambda v: v)
    cca = 5 if cps else 3
    top = ram - 1
    park = 0x7F80
    gt = [(GT.SCS10LD, 0x000), (GT.SCS10HD, (top >> 21) & 0x7F),
          (GT.SCS0LD, 0x00), (GT.SCS0HD, top >> 20),
          (GT.SCS1LD, 0xFF), (GT.SCS1HD, 0x00),
          (GT.SCS32LD, park), (GT.SCS32HD, 0x00),
          (GT.SCS2LD, 0xFF), (GT.SCS2HD, 0x00), (GT.SCS3LD, 0xFF), (GT.SCS3HD, 0x00),
          (GT.CS20LD, park), (GT.CS20HD, 0x00),
          (GT.CS3BOOTLD, 0x1E000000 >> 21), (GT.CS3BOOTHD, 0x7F),
          (GT.CS3LD, 0xFF), (GT.CS3HD, 0x00), (GT.BOOTLD, 0xE0), (GT.BOOTHD, 0xFF),
          (GT.PCI0IOLD, 0x18000000 >> 21), (GT.PCI0IOHD, 0x40),
          (GT.PCI0M0LD, 0x10000000 >> 21), (GT.PCI0M0HD, 0x3F),
          (GT.PCI0M1LD, 0x18200000 >> 21), (GT.PCI0M1HD, 0x5E),
          (GT.PCI0_BS[0], (1 << (ram - 1).bit_length()) - 1 & 0xFFFFF000),
          (GT.PCI0_BAR_EN, 0x0F | 0xE0)]
    gt_init = "\n".join(f"        li    $t1, 0x{le(v):08x}\n        sw    $t1, 0x{off:03x}($s7)"
                        for off, v in gt)
    refill = firmware_refill(cca, user_window(ram), arch=arch).split("__tlb_refill:\n", 1)[1]
    xrefill = firmware_xtlb_refill(cca, user_window(ram)).split("__xtlb_refill:\n", 1)[1]

    def addr(v: int) -> str:                    # (MIPS64: sign-extended kseg0 / kseg1)
        return str(v - (1 << 32)) if wide and v & 0x80000000 else f"0x{v:08x}"
    values = {
        "MON_DESC": addr(DESC), "CPU_ISR": addr(CPU_ISR), "IC_ISR": addr(IC_ISR),
        "ESR": addr(ESR), "MON_FLAGS": addr(MON_FLAGS),
        "FRAME": addr(FRAME64 if wide else FRAME), "EXC_STACK": addr(EXC_STACK),
        "DEFAULT_ID": addr(DEFAULT_ID),
        "ISD": f"0x{le(0x1BE00000 >> 21):08x}",
        "GT_INIT": gt_init,
        "CFG_PIRQ": f"0x{le(_cfg(0, PIIX4_DEVICE, 0, 0x60)):08x}",
        "CFG_IDE_CMD": f"0x{le(_cfg(0, PIIX4_DEVICE, 1, 0x04)):08x}",
        "CFG_IDE_TIM": f"0x{le(_cfg(0, PIIX4_DEVICE, 1, 0x40)):08x}",
        "CFG_IDE_BMIBA": f"0x{le(_cfg(0, PIIX4_DEVICE, 1, 0x20)):08x}",
        "CFG_GT_CMD": f"0x{le(_cfg(0, 0, 0, 0x04)):08x}",
        "CFG_GT_BAR0": f"0x{le(_cfg(0, 0, 0, 0x10)):08x}",
        "GT_CMD": f"0x{le(0x06):08x}",
        "BMIBA": f"0x{BMIBA:08x}",
        "DISK_TIMEOUT": "200000",
        "CONFIG_CLEAR": f"0x{~(7 << 28 | 7 << 25 | 7) & 0xFFFFFFFF:08x}",
        "CONFIG_SET": f"0x{2 << 28 | cca << 25 | cca:08x}",
        "STATUS": f"0x{C.ST_CU1 | C.ST_EXL:08x}",
        "CPU_HZ": str(cpu_hz), "BUS_HZ": str(BUS_HZ), "PCI_KHZ": str(PCI_KHZ),
        "BANNER": (f"\\neMIPS monitor (YAMON application interface): Malta, "
                   f"{ram >> 20} MB SDRAM, {cpu_hz // 1_000_000} MHz\\n"),
        "REFILL": refill if nano else "        .set noreorder\n" + refill + "        .set reorder",
        "RET": "        move  $a0, $v0" if nano else "",
        "XREFILL": "        .set noreorder\n" + xrefill + "        .set reorder",
    }
    flags = set()
    if release != "R6":
        flags.add("HILO")
    if wide:
        flags.add("WIDE")
    if nano:
        flags.add("NANO")
    text = _preprocess(SOURCE.read_text(encoding="utf-8"), values, flags)
    return _nano_text(text) if nano else _o32_numbers(text) if wide else text


@lru_cache(maxsize=32)
def image(*, big_endian: bool, release: str, ram: int, cpu_hz: int, cps: bool,
          arch: str = "mips32"):
    """The assembled monitor (a :class:`~emips.assembler.Program`; a
    nanoMIPS one for the nanoMIPS subset: it runs on every nanoMIPS core)."""
    from .assembler import assemble
    text = source(big_endian=big_endian, release=release, ram=ram, cpu_hz=cpu_hz, cps=cps,
                  arch=arch)
    if arch != "nanomips":
        return assemble(text, endian="big" if big_endian else "little", release=release,
                        arch=arch)
    prog = assemble(text, endian="big" if big_endian else "little", release="R6",
                    arch=arch, bases={"text": RESET},
                    nano={"nms": True, "tlb": True, "crc": False, "ginv": False,
                          "eva": False})
    for name, at in NANO_FIXED.items():       # (.balign reached them: say if not)
        if prog.symbols.get(name) != at:
            raise AssertionError(f"monitor.asm for nanoMIPS: {name} at "
                                 f"0x{prog.symbols.get(name, 0):08x}, not 0x{at:08x}")
    return prog


def load(board, m) -> object:
    """The monitor into the board's flash (its boot alias 0x1FC00000 is
    flash offset 0); returns the program (for its symbols)."""
    from .malta import FLASH_BASE
    prog = image(big_endian=board.big_endian, release=m.release, ram=board.ram_size,
                 cpu_hz=board.cpu_hz, cps=board.cps is not None or
                 getattr(m, "system", None) is not None, arch=m.arch)
    for addr, byte in prog.image.items():
        m.phys_write_byte(FLASH_BASE + (addr - RESET), byte)
    return prog


def describe_application(board, m, program, *, sp: int | None = None) -> None:
    """What YAMON's ``load`` and ``go`` know: the descriptor at
    :data:`DESC` (entry, flags, argc, argv, the environment, the memory
    size, $sp, $gp), argv = ["go"], the environment (MD00008 Table 2.3)."""
    env = [("memsize", f"0x{board.ram_size:08x}"), ("modetty0", "38400,n,8,1,hw"),
           ("modetty1", "38400,n,8,1,hw"), ("bootserport", "tty0"), ("prompt", "YAMON")]
    buf = bytearray()
    strings: dict[str, int] = {}

    def put(text: str) -> int:
        if text not in strings:
            strings[text] = STRINGS + len(buf)
            buf.extend(text.encode("ascii") + b"\0")
        return strings[text]
    argv = [put("go")]
    pairs = [(put(k), put(v)) for k, v in env]
    while len(buf) % 4:
        buf.append(0)
    argv_at = STRINGS + len(buf)
    words = argv + [0]
    env_at = argv_at + 4 * len(words)
    for k, v in pairs:
        words += [k, v]
    words += [0, 0]
    flags = 0
    code = program.image if program is not None else {}
    if any(0x80000180 <= a < 0x80000280 for a in code):
        flags |= OWN_HANDLER
    if any(0x80000000 <= a < 0x80000100 for a in code):
        flags |= OWN_REFILL
    entry = program.entry if program is not None else 0
    gp = program.board_gp() if program is not None else 0
    desc = [0x654D4F4E, entry, flags, len(argv), argv_at, env_at, board.ram_size,
            board.stack_top() if sp is None else sp, gp]
    desc = [w & 0xFFFFFFFF for w in desc]       # (YAMON's are 32-bit: a 64-bit CPU's
    order = "big" if m.big_endian else "little"  # kseg0 addresses sign-extended)
    raw = b"".join(w.to_bytes(4, order) for w in desc)
    for i, byte in enumerate(raw):
        m.phys_write_byte((DESC & 0x1FFFFFFF) + i, byte)
    for i, byte in enumerate(bytes(buf) + b"".join(w.to_bytes(4, order) for w in words)):
        m.phys_write_byte((STRINGS & 0x1FFFFFFF) + i, byte)


def start_state(board, m, monitor) -> dict[int, int]:
    """The registers ``go`` leaves (Table 5.1) besides $sp / $gp: a0 argc,
    a1 argv, a2 the environment, a3 the memory size, ra = back to the
    monitor (kseg0)."""
    def word(va: int) -> int:
        pa = va & 0x1FFFFFFF
        raw = bytes(m.phys_read_byte(pa + i) for i in range(4))
        return int.from_bytes(raw, "big" if m.big_endian else "little")
    return {4: word(DESC + 12), 5: word(DESC + 16), 6: word(DESC + 20),
            7: word(DESC + 24), 31: monitor.symbols["appl_return"] - 0x20000000}


def registrations(m) -> list[str]:
    """What the application registered with the monitor (its tables in
    RAM, read as a view): CPU / IC ISRs (with their data) and ESRs."""
    def word(va: int) -> int:
        raw = bytes(m.read_byte(((va & 0x1FFFFFFF) | 0xA0000000) + i) for i in range(4))
        return int.from_bytes(raw, "big" if m.big_endian else "little")
    out = []
    for n in range(9):
        isr = word(CPU_ISR + 8 * n)
        if isr:
            what = "default" if n == 8 else f"CPU interrupt {n}"
            out.append(f"{what} ISR 0x{isr:08x} (data 0x{word(CPU_ISR + 8 * n + 4):08x})")
    for n in range(17):
        isr = word(IC_ISR + 8 * n)
        if isr:
            what = "default IC" if n == 16 else f"IRQ {n}"
            out.append(f"{what} ISR 0x{isr:08x} (data 0x{word(IC_ISR + 8 * n + 4):08x})")
    for n in range(33):
        esr = word(ESR + 4 * n)
        if esr:
            what = "default" if n == 32 else f"ExcCode {n}"
            out.append(f"{what} ESR 0x{esr:08x}")
    return out


def stub_words(monitor) -> list[tuple[int, int]]:
    """The vector stubs the monitor writes (jump to its handler): (va, byte)."""
    at = monitor.symbols["exc_stub"]
    raw = [monitor.image.get(at + i, 0) for i in range(16)]
    out = []
    for vec in (0x80000180, 0x80000200):
        out += [(vec + i, b) for i, b in enumerate(raw)]
    return out
