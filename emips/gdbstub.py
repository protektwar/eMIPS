"""A GDB remote stub for eMIPS (PLAN M16).

``python3 -m emips gdb PROGRAM [--port N]`` loads the program as ``run``
does and waits for GDB (``target remote :N``); GDB then drives the
session over the remote serial protocol.

Reference: *Debugging with GDB*, appendix E "GDB Remote Serial Protocol"
(E.1 Overview: ``$packet#checksum``, ``+`` / ``-``; E.2 Packets; E.4 Stop
Reply Packets; E.5 General Query Packets: ``qSupported``, ``qXfer``;
E.6 the MIPS breakpoint kinds; E.9 Interrupts: a 0x03 byte; E.11
Packet Acknowledgment: ``QStartNoAckMode``) and appendix G "Target
Descriptions" (G.5.8 MIPS Features: ``org.gnu.gdb.mips.cpu`` / ``cp0`` /
``fpu``, the register numbers of GDB's ``features/mips-*.xml``).

The registers in GDB's order (the ``g`` packet): r0-r31, status (32),
lo (33), hi (34), badvaddr (35), cause (36), pc (37), f0-f31 (38-69: 32
bits with Status.FR = 0, 64 with FR = 1), fcsr (70), fir (71); each in
the target's byte order; with the DSP Module ``org.gnu.gdb.mips.dsp``
too: hi1, lo1, hi2, lo2, hi3, lo3 (72-77), dspctl (78) — GDB's
``features/mips-dsp.xml`` (M18).  Breakpoints (``Z0`` / ``Z1``) are the
session's: nothing is written into the program, as in QEMU's stub.  The
stub runs the session in slices and looks for GDB's interrupt between
them; what the program prints goes to GDB as ``O`` packets.
"""

from __future__ import annotations

import select
import socket
import sys
from contextlib import contextmanager
from typing import Callable

from . import cp0 as C
from .exceptions import BusError, FetchError, MipsException, SimException, UnimplementedError
from .isa import ISADecodeError
from .session import Watchpoint

M32 = 0xFFFFFFFF
#: GDB's signal numbers (include/gdb/signals.def)
SIGINT, SIGILL, SIGTRAP, SIGFPE, SIGBUS, SIGSEGV = 2, 4, 5, 8, 10, 11
#: the number of registers of the g packet (without / with the DSP Module)
NREGS = 72
NREGS_DSP = 79
#: steps between looks at the socket while continuing
SLICE = 2000
#: pipeline mode: the most cycles a step (or a stop) waits for an instruction
STEP_CYCLES = 10_000
#: Z2 / Z3 / Z4
WATCH_KINDS = {"2": "write", "3": "read", "4": "access"}
_RUN = "GDB runs the program (continue, step, next, finish, until)"
#: debugger commands ``monitor`` refuses, and what to do instead
MONITOR_REFUSED = {
    **dict.fromkeys(("step", "s", "next", "n", "back", "continue", "c", "r", "run",
                     "until", "u", "finish", "stepline", "nextline", "sl", "nl"), _RUN),
    **dict.fromkeys(("break", "b", "delete", "d"), "use GDB's break / watch / delete"),
    **dict.fromkeys(("reset", "load", "mode", "set"),
                    "restart emips gdb with the options you want"),
    **dict.fromkeys(("quit", "q", "exit"), "use GDB's detach or quit"),
    "input": "the program reads from the terminal emips gdb runs in",
}
MONITOR_HELP = (
    "monitor CMD runs an eMIPS debugger command on GDB's current thread (core):\n"
    "  regs cp fpu mem translate where dis list view print decode encode ref\n"
    "  cache cm cps gic tlb perf core cpu srs ejtag probe tap trace board disk keys\n"
    "e.g. monitor cache stats, monitor tlb, monitor cps, monitor translate $sp.\n"
    "GDB itself runs the program and keeps the breakpoints.\n")


def _checksum(data: bytes) -> int:
    return sum(data) & 0xFF


def _escape(data: bytes) -> bytes:
    """Binary data in a reply: # $ } * escaped (E.1)."""
    out = bytearray()
    for b in data:
        if b in b"#$}*":
            out += bytes((0x7D, b ^ 0x20))
        else:
            out.append(b)
    return bytes(out)


def _unescape(data: bytes) -> bytes:
    out = bytearray()
    it = iter(data)
    for b in it:
        out.append(next(it) ^ 0x20 if b == 0x7D else b)
    return bytes(out)


class Connection:
    """The packet layer over a socket: ``$data#cs`` framing, ``+`` / ``-``
    acknowledgements (until ``QStartNoAckMode``), the 0x03 interrupt."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buf = bytearray()
        self.ack = True
        self.interrupted = False
        self.log: list[str] = []

    def _fill(self, timeout: float | None) -> bool:
        if timeout is not None:
            r, _, _ = select.select([self.sock], [], [], timeout)
            if not r:
                return False
        data = self.sock.recv(4096)
        if not data:
            raise EOFError
        self.buf += data
        return True

    def _take(self) -> str | None:
        """A complete packet from the buffer (acked), or None; a 0x03
        sets :attr:`interrupted` and is skipped."""
        while True:
            while self.buf and self.buf[0] != ord("$"):
                if self.buf[0] == 0x03:
                    self.interrupted = True
                del self.buf[0]                  # acks, noise
            end = self.buf.find(b"#")
            if not self.buf or end < 0 or len(self.buf) < end + 3:
                return None
            body = bytes(self.buf[1:end])
            cs = bytes(self.buf[end + 1:end + 3])
            del self.buf[:end + 3]
            try:
                good = int(cs, 16) == _checksum(body)
            except ValueError:
                good = False
            if self.ack:
                self.sock.sendall(b"+" if good else b"-")
            if good:                             # (binary data: unescaped by X)
                text = body.decode("latin-1")
                self.log.append("<- " + text[:80])
                return text

    def read_packet(self) -> str:
        """The next packet's payload (blocking)."""
        while (pkt := self._take()) is None:
            self._fill(None)
        return pkt

    def poll_packet(self) -> str | None:
        """The next packet if GDB has sent one (never blocks)."""
        pkt = self._take()
        while pkt is None and self._fill(0):
            pkt = self._take()
        return pkt

    def poll_interrupt(self) -> bool:
        """Has GDB sent its interrupt (while the target runs)?"""
        while self._fill(0):
            pass
        if 0x03 in self.buf:
            self.buf.remove(0x03)
            self.interrupted = True
        hit, self.interrupted = self.interrupted, False
        return hit

    def send(self, payload: str | bytes) -> None:
        data = payload.encode("latin-1") if isinstance(payload, str) else payload
        frame = b"$" + data + b"#" + f"{_checksum(data):02x}".encode()
        self.log.append("-> " + data[:80].decode("latin-1"))
        del self.log[:-200]
        while True:
            self.sock.sendall(frame)
            if not self.ack:
                return
            while True:                          # wait for the + (or - : again)
                while not self.buf:
                    self._fill(None)
                c = self.buf.pop(0)
                if c == ord("+"):
                    return
                if c == ord("-"):
                    break
                if c == 0x03:
                    self.interrupted = True
                elif c == ord("$"):              # a packet before our ack: keep it
                    self.buf.insert(0, c)
                    return


class GdbStub:
    """The protocol over a :class:`~emips.session.Session` (core 0 in
    phase 1).  ``read_line()``: a line for a program that asks for input;
    ``exec_file``: the ELF GDB reads the symbols from (``qXfer:exec-file:
    read``, so ``target remote`` alone finds them)."""

    def __init__(self, session, *, read_line: Callable[[], str] | None = None,
                 exec_file: str | None = None) -> None:
        self.s = session
        self.exec_file = exec_file
        self.read_line = read_line or (lambda: sys.stdin.readline())
        self.printed = len(session.output)
        self.exited: int | None = None
        self.fr64 = False
        self.last_stop: tuple[int, str] = (SIGTRAP, "")
        #: the MT Module: GDB's current thread, a TC (index into _tcs())
        self.tc: int | None = None
        #: a host polls (GdbServer): resumes run in slices from poll()
        self.polled = False
        #: while GDB's continue / step runs (polled): is it a step?
        self.resuming: bool | None = None

    # -- registers -------------------------------------------------------------------------

    @property
    def m(self):
        return self.s.machine

    def _size(self, n: int) -> int:
        return 8 if 38 <= n <= 69 and self.fr64 else 4

    @property
    def nregs(self) -> int:
        """The g packet's registers: the DSP Module's 72-78 with it."""
        return NREGS_DSP if self.m.dsp else NREGS

    def available(self, n: int) -> bool:
        """f0-f31 / fcsr / fir exist only with an FPU (Config1.FP): GDB
        needs the mips.fpu feature (G.5.8), so a core without one reports
        them unavailable (``xx`` in g / p: GDB shows ``<unavailable>``)."""
        if n >= NREGS:
            return n < self.nregs
        return n < 38 or self.m.has_fpu

    def _reg_hex(self, n: int) -> str:
        if not self.available(n):
            return "xx" * self._size(n)
        return self._hex(self.get_reg(n), self._size(n))

    def get_reg(self, n: int) -> int:
        m = self.m
        if n < 32:
            return m.read_reg(n)
        if n == 32:
            return m.cp0_read(12, 0)
        if n == 33:
            return m.lo & M32
        if n == 34:
            return m.hi & M32
        if n == 35:
            return m.cp0_read(8, 0)
        if n == 36:
            return m.cp0_read(13, 0)
        if n == 37:
            return self._pc()
        if 38 <= n <= 69:
            v = m.fpr[n - 38]
            return v if self.fr64 else v & M32
        if n == 70:
            return m.fcsr & M32
        if n == 71:
            return m.fir & M32
        if 72 <= n <= 77 and m.dsp:              # hi1, lo1 ... hi3, lo3
            k = (n - 72) // 2 + 1
            return (m.dsp_hi[k] if n % 2 == 0 else m.dsp_lo[k]) & M32
        if n == 78 and m.dsp:
            return m.dspctl & M32
        raise KeyError(n)

    def set_reg(self, n: int, value: int) -> None:
        m = self.m
        if not self.available(n):
            raise KeyError(n)
        if n < 32:
            m.write_reg(n, value)
        elif n == 32:
            m.cp0_write(12, 0, value)
        elif n == 33:
            m.lo = value & M32
        elif n == 34:
            m.hi = value & M32
        elif n == 35:
            m.cp0_write(8, 0, value)
        elif n == 36:
            m.cp0_write(13, 0, value)
        elif n == 37:                            # a jump (pipeline: a flush); bit 0 = the ISA
            self.s.set_pc(value & M32)
        elif 38 <= n <= 69:
            k = n - 38
            m.fpr[k] = value if self.fr64 else (m.fpr[k] & ~M32) | (value & M32)
        elif n == 70:
            m.fcsr = value & M32
        elif 72 <= n <= 77:
            k = (n - 72) // 2 + 1
            (m.dsp_hi if n % 2 == 0 else m.dsp_lo)[k] = value & M32
        elif n == 78:
            from .dsp import DSPCONTROL_MASK
            m.dspctl = value & DSPCONTROL_MASK
        elif n != 71:                            # FIR is read-only
            raise KeyError(n)

    def _pc(self) -> int:
        """The next instruction to execute: instruction mode the Machine's
        PC; pipeline mode the one about to execute (everything older has
        executed, it has not: the precise point GDB stops at).  In MIPS16e
        / microMIPS code bit 0 is set (the ISA mode, as QEMU's stub and
        GDB's mips-tdep.c: is_compact_addr)."""
        s, m = self.s, self.m
        if s.pipeline is None:
            return (m.pc | m.isa_mode) & M32
        pc = s._stop_pc()
        nxt = s.pipeline.next_slot()
        if pc is None:
            pc = nxt.pc if nxt is not None else m.pc
        if nxt is not None and nxt.pc == pc:
            isa = nxt.isa
        else:
            isa = 1 if s.compressed_at(pc) else 0
        return (pc | isa) & M32

    def _hex(self, value: int, size: int) -> str:
        order = "big" if self.m.big_endian else "little"
        return (value & ((1 << 8 * size) - 1)).to_bytes(size, order).hex()

    def _unhex(self, text: str) -> int:
        order = "big" if self.m.big_endian else "little"
        return int.from_bytes(bytes.fromhex(text), order)

    def target_xml(self) -> str:
        """The target description: GDB's three MIPS features (G.5.8; the
        fpu one is required — without an FPU its registers are unavailable)."""
        from .sim import _fr64
        self.fr64 = _fr64(self.m)                           # Status.FR (R6: always 1)
        arch = {"R1": "mips:isa32", "R6": "mips:isa32r6"}.get(self.s.release, "mips:isa32r2")
        fbits = 64 if self.fr64 else 32
        ftype = "ieee_double" if self.fr64 else "ieee_single"
        first = {0: ' regnum="0"'}
        cpu = "".join(f'<reg name="r{i}" bitsize="32"{first.get(i, "")}/>' for i in range(32))
        first = {0: ' regnum="38"'}
        fpu = "".join(f'<reg name="f{i}" bitsize="{fbits}" type="{ftype}"{first.get(i, "")}/>'
                      for i in range(32))
        return ('<?xml version="1.0"?><!DOCTYPE target SYSTEM "gdb-target.dtd">'
                f'<target version="1.0"><architecture>{arch}</architecture>'
                '<feature name="org.gnu.gdb.mips.cpu">' + cpu +
                '<reg name="lo" bitsize="32" regnum="33"/>'
                '<reg name="hi" bitsize="32" regnum="34"/>'
                '<reg name="pc" bitsize="32" regnum="37"/></feature>'
                '<feature name="org.gnu.gdb.mips.cp0">'
                '<reg name="status" bitsize="32" regnum="32"/>'
                '<reg name="badvaddr" bitsize="32" regnum="35"/>'
                '<reg name="cause" bitsize="32" regnum="36"/></feature>'
                '<feature name="org.gnu.gdb.mips.fpu">' + fpu +
                '<reg name="fcsr" bitsize="32" group="float"/>'
                '<reg name="fir" bitsize="32" group="float"/></feature>'
                + (('<feature name="org.gnu.gdb.mips.dsp">'
                    + "".join(f'<reg name="{h}{k}" bitsize="32" regnum="{72 + 2 * (k - 1) + j}"/>'
                              for k in (1, 2, 3) for j, h in enumerate(("hi", "lo")))
                    + '<reg name="dspctl" bitsize="32" regnum="78"/></feature>')
                   if self.m.dsp else "") + '</target>')

    # -- memory --------------------------------------------------------------------------

    def read_memory(self, addr: int, n: int) -> bytes:
        """As a load would see it now (virtual addresses, no side effects)."""
        return bytes(self.m.read_byte((addr + i) & M32) for i in range(n))

    def write_memory(self, addr: int, data: bytes) -> None:
        for i, b in enumerate(data):
            self.m.write_byte((addr + i) & M32, b)
        self.s._code = None

    # -- running -------------------------------------------------------------------------

    def _output(self, conn: Connection) -> None:
        out = self.s.output[self.printed:]
        if out:
            self.printed = len(self.s.output)
            conn.send("O" + out.encode("latin-1", "replace").hex())

    def stop_reply(self, sig: int = SIGTRAP, reason: str = "") -> str:
        """``T``: the signal, pc and sp of the thread (= core) that
        stopped, and ``reason`` ("swbreak:", "watch:ADDR" ...)."""
        self.last_stop = (sig, reason)
        if self._tcs() is not None:             # the TC that stopped (about to issue)
            self.tc = self._stopped_tc()
            with self._bound():
                pc = self._hex(self.get_reg(37), 4)
                sp = self._hex(self.get_reg(29), 4)
            return (f"T{sig:02x}25:{pc};1d:{sp};thread:{self.tc + 1:x};core:{self.s.cpu:x};"
                    + (f"{reason};" if reason else ""))
        pc = self._hex(self.get_reg(37), 4)
        sp = self._hex(self.get_reg(29), 4)
        cpu = self.s.cpu
        return (f"T{sig:02x}25:{pc};1d:{sp};thread:{cpu + 1:x};core:{cpu:x};"
                + (f"{reason};" if reason else ""))

    def resume(self, conn: Connection, *, step: bool) -> str:
        """Run (one instruction, or until something stops it) and say why
        it stopped."""
        while True:
            reply = self._slice(conn, step)
            if reply is not None:
                return reply
            if conn.poll_interrupt():
                return self.interrupt()

    def interrupt(self) -> str:
        """GDB's Ctrl-C: the stop reply (pipeline mode: at an instruction)."""
        if self.s.pipeline is not None:
            self._settle()
        return self.stop_reply(SIGINT)

    def _slice(self, conn: Connection, step: bool) -> str | None:
        """Run a step or a slice of a continue: the stop reply, or None
        while the program goes on (or waits for input ``read_line`` does
        not have yet)."""
        s = self.s
        if s.pipeline is not None:
            r = self._step_instruction() if step else s.run(max_steps=SLICE)
        elif step and self._tcs() is not None:
            r = self._step_tc()
        else:
            r = s.step() if step else s.run(max_steps=SLICE)
        self._output(conn)
        if r.error:
            return self.stop_reply(_signal_of(r.fault))
        if r.stopped == "exit" or (self.m.halted and r.stopped in ("halt", "exit")):
            code = getattr(s.services, "exit_code", None) if s.services else None
            self.exited = code or 0
            return f"W{self.exited & 0xFF:02x}"
        if r.stopped == "input":
            line = self.read_line()
            if line is None:                     # (the GUI's console gives it)
                return None
            s.give_input(line.rstrip("\n"))
            return self.stop_reply() if step else None
        if r.stopped == "breakpoint":
            return self.stop_reply(SIGTRAP, "swbreak:")
        if r.stopped == "watch":
            w = r.fault.watch
            kind = {"write": "watch", "read": "rwatch"}.get(w.kind, "awatch")
            return self.stop_reply(SIGTRAP, f"{kind}:{r.fault.addr:x}")
        if r.stopped or step:
            return self.stop_reply()
        return None

    def _step_instruction(self):
        """Pipeline mode: clock cycles until the next instruction is the
        one about to execute (a branch on the way is held for it in the
        5-stage pipeline, as at a breakpoint)."""
        s = self.s
        p = s.pipeline
        cur = p.entering_ex()
        keep = p.debug_stop
        p.debug_stop = lambda pc: True
        try:
            for _ in range(STEP_CYCLES):
                r = s.step()
                if r.stopped or r.error:
                    return r
                e = s.pipeline.entering_ex()
                if e is not None and e is not cur:
                    return r
            return r
        finally:
            p.debug_stop = keep

    def _step_tc(self):
        """The MT Module: cycles until GDB's thread has issued an
        instruction (the others' issue too unless vCont named it alone)."""
        s = self.s
        p, tc, _ = self._tcs()[self._thread() - 1]
        before = p.issued[tc.n]
        r = s.step()
        for _ in range(STEP_CYCLES):
            if r.stopped or r.error or p.issued[tc.n] != before:
                break
            r = s.step()
        self.s.select(self._tcs()[self.tc][2] + tc.vpe)
        return r

    def _settle(self) -> None:
        """Pipeline mode, stopped between instructions (Ctrl-C): clock on
        to a cycle where an instruction is about to execute."""
        s = self.s
        for _ in range(STEP_CYCLES):
            if s.pipeline.entering_ex() is not None:
                return
            r = s.step()
            if r.stopped or r.error:
                return

    # -- the packets ---------------------------------------------------------------------

    def handle(self, conn: Connection, pkt: str) -> str | None:
        """The reply to one packet (None: no reply; "" = not supported).
        With the MT Module GDB's current thread (a TC) is bound to its VPE
        while the packet reads or writes registers / memory (its ASID)."""
        if not pkt:
            return ""
        if self._tcs() is not None and pkt[0] in "gGpPmMXZz?":
            with self._bound():
                return self._handle(conn, pkt)
        return self._handle(conn, pkt)

    def _handle(self, conn: Connection, pkt: str) -> str | None:
        c, rest = pkt[0], pkt[1:]
        if c == "?":
            return self.stop_reply(*self.last_stop)
        if c == "g":
            return "".join(self._reg_hex(n) for n in range(self.nregs))
        if c == "G":
            at = 0
            for n in range(self.nregs):
                w = 2 * self._size(n)
                if at + w > len(rest):
                    break
                if n != 71 and self.available(n):
                    self.set_reg(n, self._unhex(rest[at:at + w]))
                at += w
            return "OK"
        if c == "p":
            n = int(rest, 16)
            try:
                if n >= self.nregs:
                    raise KeyError(n)
                return self._reg_hex(n)
            except KeyError:
                return "E00"
        if c == "P":
            n, _, v = rest.partition("=")
            try:
                self.set_reg(int(n, 16), self._unhex(v))
            except KeyError:
                return "E00"
            return "OK"
        if c == "m":
            a, _, n = rest.partition(",")
            try:
                return self.read_memory(int(a, 16), int(n, 16)).hex()
            except (SimException, ValueError):
                return "E14"
        if c in "MX":
            head, _, data = rest.partition(":")
            a, _, n = head.partition(",")
            raw = bytes.fromhex(data) if c == "M" else _unescape(data.encode("latin-1"))
            try:
                self.write_memory(int(a, 16), raw[:int(n, 16)])
            except (SimException, ValueError):
                return "E14"
            return "OK"
        if c in "Zz":
            kind, addr, length = (rest.split(";")[0].split(",") + ["", ""])[:3]
            a = int(addr, 16)
            if kind in ("0", "1"):               # software / hardware breakpoint
                a &= ~1                          # (kinds 2 / 3 / 5: MIPS16e / microMIPS)
                (self.s.breakpoints.add if c == "Z" else self.s.breakpoints.discard)(a)
                return "OK"
            if kind in ("2", "3", "4"):          # write / read / access watchpoint
                w = Watchpoint(a, int(length, 16), WATCH_KINDS[kind])
                wps = self.s.watchpoints
                if c == "Z":
                    wps.append(w)
                elif w in wps:
                    wps.remove(w)
                return "OK"
            return ""
        if c in "cs":
            if rest:
                self.set_reg(37, int(rest, 16))
            return self._resume(conn, c == "s")
        if c == "D":
            return "OK"
        if c == "k":
            return None
        if c == "H":                             # Hg: the registers' thread; Hc: all run
            tid = self._tid(rest[1:])
            if rest[:1] == "g" and tid is not None and tid > 0:
                if tid > self._nthreads():
                    return "E01"
                self._select_thread(tid)
            return "OK"
        if c == "T":
            tid = self._tid(rest)
            return "OK" if tid is not None and 0 < tid <= self._nthreads() else "E01"
        if pkt.startswith("vCont?"):
            return "vCont;c;C;s;S"
        if pkt.startswith("vCont;"):
            return self._vcont(conn, pkt[6:])
        if pkt.startswith("vMustReplyEmpty"):
            return ""
        if pkt.startswith("qSupported"):
            return ("PacketSize=4000;qXfer:features:read+;swbreak+;hwbreak+;"
                    "QStartNoAckMode+;vContSupported+"
                    + (";qXfer:exec-file:read+" if self.exec_file else ""))
        if pkt == "QStartNoAckMode":
            conn.send("OK")
            conn.ack = False
            return None
        if pkt.startswith("qXfer:features:read:"):
            name, _, span = pkt[len("qXfer:features:read:"):].partition(":")
            if name != "target.xml":
                return "E00"
            return self._xfer(self.target_xml().encode(), span)
        if pkt.startswith("qXfer:exec-file:read:") and self.exec_file:
            span = pkt.rsplit(":", 1)[1]                 # annex = the pid (ours: 1)
            return self._xfer(self.exec_file.encode(), span)
        if pkt in ("qC",):
            return f"QC{self._thread():x}"
        if pkt in ("qfThreadInfo",):
            return "m" + ",".join(f"{i + 1:x}" for i in range(self._nthreads()))
        if pkt in ("qsThreadInfo",):
            return "l"
        if pkt.startswith("qThreadExtraInfo,"):
            return self._thread_info(self._tid(pkt.split(",", 1)[1])).encode().hex()
        if pkt.startswith("qRcmd,"):
            return self.monitor(conn, bytes.fromhex(pkt[6:]).decode("latin-1"))
        if pkt.startswith("qAttached"):
            return "1"
        if pkt in ("qOffsets",):
            return "Text=0;Data=0;Bss=0"
        if pkt.startswith("qSymbol"):
            return "OK"
        return ""

    # -- threads = cores ---------------------------------------------------------------

    @staticmethod
    def _tid(text: str) -> int | None:
        """A thread id (``-1`` all, ``0`` any; ``pP.T`` multiprocess)."""
        text = text.split(".")[-1]
        try:
            return int(text, 16) if text != "-1" else -1
        except ValueError:
            return None

    # -- threads = the MT Module's TCs ----------------------------------------------------

    def _tcs(self) -> list | None:
        """With the MT Module: every (processor, TC, VP base), core by core —
        GDB's threads; else None (the threads are the cores)."""
        mt = self.s.mt
        if mt is None:
            return None
        if hasattr(mt, "procs"):
            vpes = self.s.cps.vpes
            return [(p, tc, c * vpes) for c, p in enumerate(mt.procs) for tc in p.tcs]
        return [(mt, tc, 0) for tc in mt.tcs]

    def _nthreads(self) -> int:
        tcs = self._tcs()
        return len(tcs) if tcs is not None else len(self.s.machines)

    def _thread(self) -> int:
        """GDB's current thread id."""
        tcs = self._tcs()
        if tcs is None:
            return self.s.cpu + 1
        if self.tc is None or self.tc >= len(tcs):
            self.tc = self._stopped_tc()
        return self.tc + 1

    def _stopped_tc(self) -> int:
        """The TC bound to the selected VP (the one about to issue there)."""
        tcs = self._tcs()
        vp = self.s.cpu
        for i, (p, tc, base) in enumerate(tcs):
            if base + tc.vpe == vp and p.is_bound(tc):
                return i
        return 0

    def _select_thread(self, tid: int) -> None:
        tcs = self._tcs()
        if tcs is None:
            self.s.select(tid - 1)
            return
        self.tc = tid - 1
        p, tc, base = tcs[self.tc]
        self.s.select(base + tc.vpe)

    @contextmanager
    def _bound(self):
        """GDB's thread (a TC) bound to its VPE for a moment."""
        tcs = self._tcs()
        tid = self._thread()
        p, tc, base = tcs[tid - 1]
        if self.s.cpu != base + tc.vpe:
            self.s.select(base + tc.vpe)
        with p.bound(tc):
            yield

    def _thread_info(self, tid: int | None) -> str:
        tcs = self._tcs()
        if tcs is not None:
            if tid is None or not 0 < tid <= len(tcs):
                return ""
            p, tc, base = tcs[tid - 1]
            where = f"VP {base + tc.vpe}" if base or len(tcs) > len(p.tcs) else f"VPE {tc.vpe}"
            return f"TC {tc.n} on {where}, {p.state(tc)}"
        rows = self.s.cores()
        if tid is None or not 0 < tid <= len(rows):
            return ""
        r = rows[tid - 1]
        what = "waiting (WAIT)" if r["waiting"] else "halted" if r["halted"] else \
            "running" if r["running"] else r["state"]
        return f"core {tid - 1}, {what}" if len(rows) > 1 or what != "running" else "core 0"

    def _vcont(self, conn: Connection, actions: str) -> str | None:
        """All-stop ``vCont``: ``c`` / ``s`` for every thread, or for named
        threads only (GDB's step over a breakpoint: ``vCont;c:1`` — the
        other cores stay stopped, :attr:`Session.only`); a step action
        names the thread the stop reports."""
        step, only, everyone = None, set(), False
        n = self._nthreads()
        for a in actions.split(";"):
            act, _, tid = a.partition(":")
            if act[:1] not in "cCsS":
                continue
            t = self._tid(tid) if tid else -1
            if t is None or t > n:
                return "E01"
            if t in (-1, 0):
                everyone = True
                if act[:1] in "sS" and step is None:
                    step = self._thread()
                continue
            only.add(t - 1)
            if act[:1] in "sS" and step is None:
                step = t
        if step is not None:
            self._select_thread(step)
        return self._resume(conn, step is not None, None if everyone else only)

    def _resume(self, conn: Connection, step: bool, only: set | None = None) -> str | None:
        """Run now (the blocking server), or, for a host that polls
        (:class:`GdbServer`), note it: :attr:`resuming` until it stops.
        ``only``: the cores that run (None: all)."""
        tcs = self._tcs()
        if tcs is not None:                     # threads are TCs: the scheduler's
            self.s.only = None
            if only is not None:
                for p in {id(p): p for p, _, _ in tcs}.values():
                    p.only = {tcs[i][1] for i in only if tcs[i][0] is p}
        else:
            self.s.only = only
        if self.polled:
            self.resuming = step
            return None
        try:
            return self.resume(conn, step=step)
        finally:
            self._unrestrict()

    def _unrestrict(self) -> None:
        self.s.only = None
        for p, _, _ in self._tcs() or ():
            p.only = None

    # -- monitor -------------------------------------------------------------------------

    def monitor(self, conn: Connection, line: str) -> str:
        """``monitor CMD``: an eMIPS debugger command on GDB's current
        thread; its output goes to GDB (O packets).  Commands that run
        the program or change what GDB keeps (breakpoints, the program,
        the core) are refused: GDB does those itself."""
        from .cli import Debugger
        import io
        words = line.split()
        if not words or words[0] == "help":
            text = MONITOR_HELP
        elif words[0] in MONITOR_REFUSED or (words[0] == "cpu" and len(words) > 1):
            text = (f"'{words[0]}' is GDB's job here: " +
                    MONITOR_REFUSED.get(words[0], "use GDB's thread N") + "\n")
        else:
            out = io.StringIO()
            d = Debugger(self.s, stdout=out)
            d._printed = len(self.s.output)          # (the program's output is GDB's)
            d.onecmd(line)
            text = out.getvalue()
        if text:
            for k in range(0, len(text), 1024):
                conn.send("O" + text[k:k + 1024].encode("latin-1", "replace").hex())
        return "OK"

    @staticmethod
    def _xfer(doc: bytes, span: str) -> bytes:
        """A qXfer read reply: ``OFFSET,LENGTH`` of ``doc`` (m = more)."""
        off, _, length = span.partition(",")
        o, n = int(off, 16), int(length, 16)
        return (b"m" if o + n < len(doc) else b"l") + _escape(doc[o:o + n])

    def answer(self, conn: Connection, pkt: str) -> bool:
        """Handle one packet and send its reply; False when the session
        is over (detach, kill, the program exited)."""
        try:
            reply = self.handle(conn, pkt)
        except (EOFError, OSError):
            raise
        except Exception as exc:                 # a bad packet: an error, never a dead stub
            conn.log.append(f"!! {type(exc).__name__}: {exc}")
            reply = "E01"
        if reply is not None:
            conn.send(reply)
        if pkt in ("D", "k") or pkt.startswith("vKill"):
            return False
        return not self.finished(reply)

    def finished(self, reply) -> bool:
        """Was ``reply`` the program's exit (W)?"""
        return self.exited is not None and isinstance(reply, str) and reply.startswith("W")

    def serve(self, conn: Connection) -> None:
        """Answer GDB until it detaches, kills or goes away."""
        try:
            while self.answer(conn, conn.read_packet()):
                pass
        except (EOFError, OSError):
            return


#: GDB's signal by Cause.ExcCode (what a MIPS Linux kernel sends: AdE /
#: TLB / Mod → SIGSEGV, IBE / DBE → SIGBUS, RI / CpU → SIGILL, Bp / Tr /
#: Watch → SIGTRAP, Ov / FPE → SIGFPE)
_SIGNALS = {C.MOD: SIGSEGV, C.TLBL: SIGSEGV, C.TLBS: SIGSEGV, C.ADEL: SIGSEGV,
            C.ADES: SIGSEGV, C.TLBRI: SIGSEGV, C.TLBXI: SIGSEGV,
            C.IBE: SIGBUS, C.DBE: SIGBUS, C.RI: SIGILL, C.CPU: SIGILL,
            C.OV: SIGFPE, C.FPE: SIGFPE}


def _signal_of(fault: BaseException | None) -> int:
    """GDB's signal for a simulator stop (``StepResult.fault``)."""
    if isinstance(fault, MipsException):
        return _SIGNALS.get(fault.exc_code, SIGTRAP)
    if isinstance(fault, BusError):
        return SIGBUS
    if isinstance(fault, ISADecodeError):
        return SIGILL                    # a word that is no instruction (RI)
    if isinstance(fault, UnimplementedError):
        return SIGILL
    if isinstance(fault, FetchError):
        return SIGSEGV                   # no code there / nothing maps the PC
    return SIGTRAP


class GdbServer:
    """The stub for a host that owns the session's run loop (the GUI):
    never blocks — :meth:`poll` from a timer accepts GDB, answers what it
    sent and runs a slice of a continue / step it asked for.  ``exec_file``
    may be a function: the ELF is made when GDB connects (the program as
    it is then).  A program waiting for input waits for the host to give
    it (``Session.give_input``)."""

    def __init__(self, session, port: int = 1234, *, host: str = "127.0.0.1",
                 exec_file: str | Callable[[], str] | None = None) -> None:
        self.s = session
        self.exec_file = exec_file
        self.srv = listen(port, host)
        self.srv.setblocking(False)
        self.host, self.port = host, self.srv.getsockname()[1]
        self.conn: Connection | None = None
        self.stub: GdbStub | None = None
        #: what happened at the last poll that a host shows
        self.exited: int | None = None

    @property
    def state(self) -> str:
        """"waiting" (no GDB yet), "running" (GDB's continue / step) or
        "stopped" (GDB is in charge, the program waits)."""
        if self.stub is None:
            return "waiting"
        return "running" if self.stub.resuming is not None else "stopped"

    def poll(self) -> str | None:
        """One look: "connected", "stopped" (a resume ended: refresh the
        views), "running", "left" (GDB went: the host has the program
        again), or None (nothing new)."""
        if self.conn is None:
            try:
                sock, _ = self.srv.accept()
            except (BlockingIOError, InterruptedError):
                return None
            sock.setblocking(True)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if self.s.machine is not None and self.s.halted:
                self.s.reset()                   # an ended program starts again
            self.conn = Connection(sock)
            ef = self.exec_file() if callable(self.exec_file) else self.exec_file
            self.stub = GdbStub(self.s, read_line=lambda: None, exec_file=ef)
            self.stub.polled = True
            return "connected"
        conn, stub = self.conn, self.stub
        try:
            if stub.resuming is not None:
                reply = stub.interrupt() if conn.poll_interrupt() else \
                    stub._slice(conn, stub.resuming)
                if reply is None:
                    return "running"
                stub.resuming = None
                stub._unrestrict()
                conn.send(reply)
                if stub.finished(reply):
                    self.exited = stub.exited
                    return self._drop()
                return "stopped"
            while (pkt := conn.poll_packet()) is not None:
                if not stub.answer(conn, pkt):
                    return self._drop()
                if stub.resuming is not None:
                    return "running"
        except (EOFError, OSError):
            return self._drop()
        return None

    def _drop(self) -> str:
        """GDB is gone (detach, kill, exit, a closed socket): wait for the
        next one."""
        if self.stub is not None:
            self.stub._unrestrict()
        self.s.only = None
        if self.conn is not None:
            try:
                self.conn.sock.close()
            except OSError:
                pass
        self.conn = self.stub = None
        return "left"

    def close(self) -> None:
        self._drop()
        self.srv.close()


def exec_file_for(session, path: str | None = None) -> str:
    """The ELF GDB reads the symbols from: the program's own (C, ELF), or
    one :mod:`emips.elfwrite` writes for an assembly program (its labels,
    its source lines) in a temporary folder."""
    import tempfile
    from pathlib import Path
    from .elfwrite import write_elf
    if session.kind == "elf":
        return str(Path(session.source or path).resolve())
    src = Path(path).resolve() if path else Path("program.asm")
    out = Path(tempfile.mkdtemp(prefix="emips-gdb-")) / (src.stem + ".elf")
    out.write_bytes(write_elf(session.program, source=src))
    return str(out)


def listen(port: int, host: str = "127.0.0.1") -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    return srv


def serve(session, port: int, *, host: str = "127.0.0.1", on_ready=None,
          read_line=None, server: socket.socket | None = None,
          exec_file: str | None = None) -> int | None:
    """Wait for GDB on ``port`` and serve it; the program's exit code
    (None: GDB left first)."""
    srv = server or listen(port, host)
    if on_ready is not None:
        on_ready(srv.getsockname()[1])
    sock, _ = srv.accept()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    stub = GdbStub(session, read_line=read_line, exec_file=exec_file)
    try:
        stub.serve(Connection(sock))
    finally:
        sock.close()
        if server is None:
            srv.close()
    return stub.exited
