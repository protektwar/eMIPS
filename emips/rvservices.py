"""RARS's environment-call services for RISC-V programs (PLAN M25 phase 3c,
decision 10): ``ecall`` with the service number in a7, arguments in a0-a3 /
fa0, results in a0 / a1 / fa0 — RARS 1.6's ``Syscall.properties`` numbers
and each service's ``simulate`` (``rars/riscv/syscalls/*.java``,
``rars/util/SystemIO.java`` / ``Binary.java``), installed as the hart's
``syscall_handler`` (emips.simrv asks it before trapping).

RARS reads registers as Java ints (``RegisterFile.getValue``: the low 32
bits) and writes them sign-extended; on RV64 eMIPS does the same.  Output
formats are Java's (``Integer.toString``, ``Float.toString`` /
``Double.toString``: the shortest digits, scientific below 10^-3 and from
10^7 up; ``Binary.intToHexString``: 0x + 8 lowercase digits), and the
random streams are ``java.util.Random`` (the 48-bit LCG), so a seeded
program prints what RARS prints.

Choices (eMIPS has no host clock, sound, dialogs or files in a program's
reach, and a run must replay exactly — ``sim.DEVIATIONS["RARS services"]``):
time (30) counts the hart's cycles at 1 MHz from 0, sleep (32) and the
synchronous MIDI (33) let that time pass, MIDI (31) is silent; an unseeded
random stream starts from seed 0; GetCWD (17) gives "/"; open (1024) fails
(-1), read / write / close / lseek (57 / 62 / 63 / 64) know the standard
descriptors 0-2 only; the dialogs (50-60) are not offered (an environment
call exception, as an unknown service).  RARS's "ExitingException" errors
(an unreadable number, a negative sbrk ...) stop the program with
:class:`ServiceError`.
"""

from __future__ import annotations

import struct
from typing import Callable

from .exceptions import SimException

M32 = 0xFFFFFFFF
HEAP_BASE = 0x10040000                       # RARS's heap (= MARS's)
DATA_LIMIT = 0x10040000 + 0x3FFC0000         # (dataSegmentLimitAddress: below the stack)
CYCLES_PER_MS = 1000                         # time (30) / sleep (32): a 1 MHz clock
#: the services that read input (Session.needs_input)
READ_SERVICES = frozenset((5, 6, 7, 8, 12))


class ServiceError(SimException):
    """A RARS service stopped the program (its ExitingException)."""


def _s32(v: int) -> int:
    v &= M32
    return v - (1 << 32) if v >> 31 else v


class JavaRandom:
    """java.util.Random: the 48-bit linear congruential generator."""

    MULT, ADD, MASK = 0x5DEECE66D, 0xB, (1 << 48) - 1

    def __init__(self, seed: int = 0) -> None:
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        self.seed = (seed ^ self.MULT) & self.MASK

    def next(self, bits: int) -> int:
        self.seed = (self.seed * self.MULT + self.ADD) & self.MASK
        return _sx(self.seed >> (48 - bits), bits)

    def next_int(self, bound: int | None = None) -> int:
        if bound is None:
            return self.next(32)
        if bound <= 0:
            raise ValueError("bound must be positive")
        if bound & -bound == bound:             # a power of 2
            return (bound * (self.next(31) & 0x7FFFFFFF)) >> 31
        while True:
            bits = self.next(31) & 0x7FFFFFFF
            val = bits % bound
            if bits - val + (bound - 1) < 1 << 31:
                return val

    def next_float(self) -> float:
        return (self.next(24) & 0xFFFFFF) / float(1 << 24)

    def next_double(self) -> float:
        return (((self.next(26) & 0x3FFFFFF) << 27) + (self.next(27) & 0x7FFFFFF)) \
            * (1.0 / (1 << 53))


def _sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if bits == 32 and v >> 31 else v


def java_string(x: float, single: bool = False) -> str:
    """Java's Float.toString / Double.toString: the shortest digits that
    read back; plain from 10^-3 to 10^7, else d.dddE[-]n."""
    if x != x:
        return "NaN"
    if x in (float("inf"), float("-inf")):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0:
        return "-0.0" if struct.pack(">d", x)[0] & 0x80 else "0.0"
    def same(t: str) -> bool:
        if not single:
            return float(t) == x
        try:
            return struct.pack(">f", float(t)) == struct.pack(">f", x)
        except OverflowError:                   # (rounded past the largest float)
            return False

    text = f"{x:.16e}"
    for n in range(1, 18):                      # the fewest significant digits
        t = f"{x:.{n - 1}e}"
        if same(t):
            text = t
            break
    mant, exp = text.split("e")
    exp = int(exp)
    sign = "-" if mant.startswith("-") else ""
    digits = mant.lstrip("-").replace(".", "").rstrip("0") or "0"
    if -3 <= exp < 7:                           # plain notation
        if exp >= 0:
            whole = digits[:exp + 1].ljust(exp + 1, "0")
            frac = digits[exp + 1:] or "0"
        else:
            whole, frac = "0", "0" * (-exp - 1) + digits
        return f"{sign}{whole}.{frac}"
    return f"{sign}{digits[0]}.{digits[1:] or '0'}E{exp}"


class RarsServices:
    """RARS's services (install as ``machine.syscall_handler``).
    ``read_line()`` supplies input lines, ``write(text)`` takes output."""

    def __init__(self, *, read_line: Callable[[], str] | None = None,
                 write: Callable[[str], None] | None = None) -> None:
        self.read_line = read_line or (lambda: "")
        self._write = write
        self.output = ""
        self.heap = HEAP_BASE
        self.exit_code: int | None = None
        self.streams: dict[int, JavaRandom] = {}

    def write(self, text: str) -> None:
        self.output += text
        if self._write:
            self._write(text)

    # -- registers as RARS sees them ------------------------------------------------

    @staticmethod
    def _a(m, n: int) -> int:
        """getValue("an"): the low 32 bits, signed."""
        return _s32(m.read_reg(10 + n))

    @staticmethod
    def _set(m, reg: int, value: int) -> None:
        """updateRegister: an int, sign-extended to XLEN."""
        m.write_reg(reg, _s32(value) & m.amask)

    def _fa0_single(self, m) -> float:
        return struct.unpack("<f", (m.fpr[10] & M32).to_bytes(4, "little"))[0]

    def _stream(self, m) -> JavaRandom:
        index = self._a(m, 0)
        s = self.streams.get(index)
        if s is None:
            s = self.streams[index] = JavaRandom(0)
        return s

    def _string(self, m, addr: int) -> str:
        out = bytearray()
        for i in range(1 << 16):
            b = m.read_byte((addr + i) & m.amask)
            if b == 0:
                break
            out.append(b)
        return out.decode("utf-8", "replace")

    # -- the services -----------------------------------------------------------------

    def __call__(self, m, code: int) -> bool:
        code = _s32(code)
        a0 = self._a(m, 0)
        if code == 1:                           # PrintInt
            self.write(str(a0))
        elif code == 2:                         # PrintFloat (fa0)
            self.write(java_string(self._fa0_single(m), single=True))
        elif code == 3:                         # PrintDouble (fa0)
            self.write(java_string(struct.unpack("<d", m.fpr[10].to_bytes(8, "little"))[0]))
        elif code == 4:                         # PrintString
            self.write(self._string(m, m.read_reg(10)))
        elif code == 5:                         # ReadInt
            try:
                self._set(m, 10, _java_int(self.read_line()))
            except ValueError:
                raise ServiceError("invalid integer input (syscall 5)") from None
        elif code in (6, 7):                    # ReadFloat / ReadDouble -> fa0
            try:
                v = float(_java_float(self.read_line()))
            except ValueError:
                raise ServiceError(f"invalid {'float' if code == 6 else 'double'} input "
                                   f"(syscall {code})") from None
            if code == 6:
                bits = struct.unpack("<I", struct.pack("<f", v))[0]
                m.fpr[10] = (0xFFFFFFFF00000000 | bits) if m.flen == 64 else bits
            else:
                m.fpr[10] = struct.unpack("<Q", struct.pack("<d", v))[0]
        elif code == 8:                         # ReadString (a0 buffer, a1 length)
            self._read_string(m, a0, self._a(m, 1) - 1)
        elif code == 9:                         # Sbrk
            if a0 < 0:
                raise ServiceError(f"request ({a0}) is negative heap amount (syscall 9)")
            new = self.heap + a0
            if new % 4:
                new += 4 - new % 4
            if new >= DATA_LIMIT:
                raise ServiceError(f"request ({a0}) exceeds available heap storage (syscall 9)")
            self._set(m, 10, self.heap)
            self.heap = new
        elif code == 10:                        # Exit
            self.exit_code = 0
            m.halted = True
        elif code == 11:                        # PrintChar
            self.write(chr(a0 & 0xFF))
        elif code == 12:                        # ReadChar
            line = self.read_line()
            if not line:
                raise ServiceError("invalid char input (syscall 12)")
            self._set(m, 10, ord(line[0]))
        elif code == 17:                        # GetCWD (a0 buffer, a1 length)
            path = b"/"
            if self._a(m, 1) < len(path) + 1:
                self._set(m, 10, -1)
            else:
                for k, b in enumerate(path + b"\0"):
                    m.write_byte((m.read_reg(10) + k) & m.amask, b)
        elif code == 30:                        # Time: ms in a0 (low) / a1 (high)
            ms = m.cycles // CYCLES_PER_MS
            self._set(m, 10, ms & M32)
            self._set(m, 11, (ms >> 32) & M32)
        elif code in (31, 32, 33):              # MidiOut / Sleep / MidiOutSync
            ms = a0 if code == 32 else self._a(m, 1) if code == 33 else 0
            if ms > 0:
                m.advance(ms * CYCLES_PER_MS)
        elif code == 34:                        # PrintIntHex
            self.write(f"0x{a0 & M32:08x}")
        elif code == 35:                        # PrintIntBinary
            self.write(f"{a0 & M32:032b}")
        elif code == 36:                        # PrintIntUnsigned
            self.write(str(a0 & M32))
        elif code == 40:                        # RandSeed (a0 stream, a1 seed)
            s = self.streams.get(a0)
            if s is None:
                self.streams[a0] = JavaRandom(self._a(m, 1))
            else:
                s.set_seed(self._a(m, 1))
        elif code == 41:                        # RandInt
            self._set(m, 10, self._stream(m).next_int())
        elif code == 42:                        # RandIntRange (a1 bound)
            bound = self._a(m, 1)
            if bound <= 0:
                raise ServiceError("Upper bound of range cannot be negative (syscall 42)")
            self._set(m, 10, self._stream(m).next_int(bound))
        elif code == 43:                        # RandFloat -> fa0
            bits = struct.unpack("<I", struct.pack("<f", self._stream(m).next_float()))[0]
            m.fpr[10] = (0xFFFFFFFF00000000 | bits) if m.flen == 64 else bits
        elif code == 44:                        # RandDouble -> fa0
            m.fpr[10] = struct.unpack("<Q", struct.pack("<d", self._stream(m).next_double()))[0]
        elif code == 57:                        # Close
            pass
        elif code == 62:                        # LSeek: the standard descriptors cannot
            self._set(m, 10, -1)
        elif code == 63:                        # Read (a0 fd, a1 buffer, a2 length)
            self._read(m, a0, self._a(m, 2))
        elif code == 64:                        # Write (a0 fd, a1 buffer, a2 length)
            self._write_fd(m, a0, self._a(m, 2))
        elif code == 93:                        # Exit2 (a0 code)
            self.exit_code = a0
            m.halted = True
        elif code == 1024:                      # Open: no host files in reach
            self._set(m, 10, -1)
        else:
            return False                        # an environment call exception
        return True

    def _read_string(self, m, buf: int, max_len: int) -> None:
        """SyscallReadString: at most a1 - 1 characters, a newline when it
        fits, a NUL unless a1 < 1."""
        add_nul = max_len >= 0
        max_len = max(max_len, 0)
        text = self.read_line()
        if text.endswith("\n"):
            text = text[:-1]
        if len(text) > max_len:
            text = text[:max_len]
        data = text.encode("utf-8")[:max_len]
        n = len(data)
        for k, b in enumerate(data):
            m.write_byte((buf + k) & m.amask, b)
        if n < max_len:
            m.write_byte((buf + n) & m.amask, 0x0A)
            n += 1
        if add_nul:
            m.write_byte((buf + n) & m.amask, 0)

    def _read(self, m, fd: int, length: int) -> None:
        if fd != 0 or length < 0:
            self._set(m, 10, -1)
            return
        data = (self.read_line() + "\n").encode("utf-8")[:length]
        buf = m.read_reg(11)
        for k, b in enumerate(data):
            m.write_byte((buf + k) & m.amask, b)
        self._set(m, 10, len(data))

    def _write_fd(self, m, fd: int, length: int) -> None:
        if length < 0 or fd not in (1, 2):
            self._set(m, 10, -1)
            return
        buf = m.read_reg(11)
        data = bytes(m.read_byte((buf + k) & m.amask) for k in range(length))
        self.write(data.decode("utf-8", "replace"))
        self._set(m, 10, length)


def _java_int(text: str) -> int:
    """Integer.parseInt(trim): decimal, an optional sign, 32 bits."""
    t = text.strip()
    if not t or not (t.lstrip("+-").isdigit()) or t.count("-") + t.count("+") > 1 \
            or (len(t) > 1 and t[1:].lstrip("0123456789")):
        raise ValueError(text)
    v = int(t)
    if not -(1 << 31) <= v < 1 << 31:
        raise ValueError(text)
    return v


def _java_float(text: str) -> str:
    """Float.parseFloat / Double.parseDouble's text (trimmed; Java takes
    a trailing f / d and "Infinity" / "NaN")."""
    t = text.strip()
    if t.endswith(("f", "F", "d", "D")) and t[:-1].strip():
        t = t[:-1]
    if t.lstrip("+-") in ("Infinity", "NaN"):
        return t.replace("Infinity", "inf")
    float(t)                                    # (raises ValueError)
    if any(c.isalpha() and c not in "eE" for c in t):
        raise ValueError(text)
    return t
