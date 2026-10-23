"""Exercise checks (PLAN M21): what a program must do, written next to it
or in a file of its own, and a run that says which checks pass (Qt-free).

A check is a line — a comment in the program (``#`` in assembly, ``//``
or ``/* ... */`` in C), or a line of a checks file (the comment marker is
optional there)::

    # check: output is "36"
    # check: cycles <= 61          (no stall left: 61 cycles)
    # check: $t2 == 36
    # check: word[nums+4] == 2
    # check: exceptions == 0

A trailing ``(...)`` is a note, shown with the result (``eMIPS:``,
``limit:`` and ``after:`` lines may have one too).  A program's
checks are (:func:`checks_of`): a checks file chosen for it (``--checks
FILE``), else its own ``check:`` lines, else the checks file named like
it (``prog.check`` next to ``prog.asm``).  What the run is like comes from
the program's ``# eMIPS:`` header (emips.profile), and from settings lines
before the checks:

    ``eMIPS: FLAGS``   the machine for this run (the header's flags otherwise)
    ``input: LINE``    one line for a read syscall (one per read)
    ``keys: TEXT``     keystrokes for the MMIO keyboard (``\\n`` ... understood)
    ``limit: N``       the most steps the run may take (the ``run`` command's
                       default otherwise); a program that has not finished
                       by then fails
    ``after: N``       check after N steps (the program need not finish)
    ``run: NAME``      a new run: the lines after it describe it (several
                       runs: several inputs, machines ...)

Subjects:

    ``output`` ``is`` / ``contains`` / ``matches`` ``"TEXT"`` — the program's
    console output (``is``: the whole of it, trailing white space ignored;
    ``matches``: a Python regular expression found in it); ``==`` / ``!=``
    are ``is`` / ``is not``.

    ``$t0`` ``$8`` ``hi`` ``lo`` ``pc``, ``$f2`` (its 32 bits), ``$f2.s`` /
    ``$f2.d`` (the single / double it holds), ``word[LOC]`` ``half[LOC]``
    ``byte[LOC]`` ``float[LOC]`` ``double[LOC]`` (LOC: a label, a number,
    ``$reg``, with ``+N`` / ``-N``) — compared with a number, a character
    ``'A'`` or a label (its address, ``+N`` / ``-N``); ``==`` / ``!=`` compare the bits (a
    negative number is its two's complement), ``<`` ... ``>=`` the signed
    values.  ``cpu N`` before one reads core / VPE N.

    ``cycles`` (pipeline: clock cycles; instruction mode: the machine's
    cycles), ``instructions`` (executed / retired), ``steps``, ``stalls``
    / ``stalls KIND`` (pipeline mode: ``load-use``, ``branch`` ...),
    ``exceptions`` (taken), ``exit code`` (syscall 17's; 0 after syscall 10).

Every run also checks that the program finishes (exits, or runs past its
last instruction) without an error, unless it has ``after:``.
"""

from __future__ import annotations

import re
import shlex
import struct
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .syntax import gpr_number, parse_fpr

M32 = 0xFFFFFFFF

#: a settings / check line (in a program: behind a comment marker)
_LINE = re.compile(r"^\s*(?:(?:#|//|/\*+|\*)\s*)?(check|input|keys|limit|after|run|eMIPS)"
                   r"\s*:\s?(.*?)\s*(?:\*/)?\s*$")
_IN_PROGRAM = re.compile(r"^\s*(?:#|//|/\*+|\*)\s*(?:check|input|keys|limit|after|run)\s*:")
_NOTE = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")
_OPS = ("==", "!=", "<=", ">=", "<", ">")
_COUNTS = ("cycles", "instructions", "steps", "exceptions")


class CheckError(ValueError):
    """A line that is not a check (``line``: its number, 1-based;
    ``source``: the checks file, "" for the program's own lines)."""

    def __init__(self, line: int, message: str) -> None:
        super().__init__(f"line {line}: {message}" if line else message)
        self.line = line
        self.source = ""


@dataclass
class Check:
    line: int
    text: str                   # as written (without the note)
    subject: tuple              # ("output",) / ("reg", "$t0") / ("mem", kind, loc) / ...
    op: str                     # == != < <= > >= is contains matches
    value: Any                  # int / float / str / ("label", name, offset)
    cpu: int | None = None
    note: str = ""


@dataclass
class Run:
    name: str = ""
    line: int = 0                       # where it begins
    flags: list[str] | None = None      # eMIPS: (None: the program's header)
    flags_line: int = 0
    inputs: list[str] = field(default_factory=list)
    keys: str = ""
    limit: int | None = None
    after: int | None = None
    checks: list[Check] = field(default_factory=list)


@dataclass
class Outcome:
    check: Check
    passed: bool
    found: str                          # what the run had ("77", "35 (0x23)" ...)


@dataclass
class RunResult:
    run: Run
    output: str = ""
    finished: bool = False
    why: str = ""                       # why it did not finish
    summary: str = ""                   # "77 cycles, 57 instructions retired ..."
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (self.finished or self.run.after is not None) and \
            all(o.passed for o in self.outcomes)


# ------------------------------------------------------------------ parsing

@dataclass
class Checks:
    """A program's checks and where they come from."""
    runs: list[Run]
    file: Path | None = None            # the checks file (None: the program's lines)


def checks_of(text: str | None, path: str | Path | None = None,
              chosen: str | Path | None = None) -> Checks:
    """The checks of a program (``text``: its source, None for HEX / ELF;
    ``path``: its file): the ``chosen`` checks file, else its own
    ``check:`` lines, else ``NAME.check`` next to it — no runs when it has
    none.  A file that cannot be read is an OSError, a bad line a
    :class:`CheckError` (with its ``source``)."""
    def from_file(f: Path) -> Checks:
        try:
            return Checks(parse(f.read_text(encoding="utf-8"), program=False), f)
        except CheckError as exc:
            exc.source = str(f)
            raise
    if chosen is not None:
        return from_file(Path(chosen))
    runs = parse(text) if text is not None else []
    if runs:
        return Checks(runs)
    if path is not None:
        sibling = Path(path).with_suffix(".check")
        if sibling.is_file() and sibling != Path(path):
            return from_file(sibling)
    return Checks([])


def has_checks(text: str) -> bool:
    """Does the program carry checks (a ``# check:`` comment ...)?"""
    return any(_IN_PROGRAM.match(ln) and "check" in ln.split(":", 1)[0]
               for ln in text.splitlines())


def parse(text: str, *, program: bool = True) -> list[Run]:
    """The runs and checks of a program (``program``: only comment lines
    count — not a label ``check:``) or of a checks file.  A run is a
    ``run:`` line or the first setting / check, and what follows it; a
    setting after a run's checks begins the next run.  A line that is not
    a check, or a run without checks (when there are checks), is a
    :class:`CheckError`."""
    runs: list[Run] = []
    cur: Run | None = None
    for no, raw in enumerate(text.splitlines(), 1):
        if program and not _IN_PROGRAM.match(raw):
            continue
        m = _LINE.match(raw)
        if m is None:
            if not program and raw.strip() and not raw.lstrip().startswith(("#", "//")):
                raise CheckError(no, f"not a check line: {raw.strip()!r}")
            continue
        key, body = m.group(1), m.group(2)
        if key == "run":
            cur = Run(name=body.strip(), line=no)
            runs.append(cur)
            continue
        if key == "check":
            if cur is None:
                cur = Run(line=no)
                runs.append(cur)
            cur.checks.append(parse_check(body, no))
            continue
        if cur is None or cur.checks:   # a setting after the checks: the next run
            cur = Run(line=no)
            runs.append(cur)
        _setting(cur, key, body, no)
    if any(r.checks for r in runs):
        for r in runs:
            if not r.checks:
                raise CheckError(r.line, "a run without checks (settings belong before "
                                         "the checks of their run)")
    return [r for r in runs if r.checks]


def _setting(run: Run, key: str, body: str, no: int) -> None:
    if key == "eMIPS":
        run.flags = (run.flags or []) + shlex.split(_NOTE.sub(r"\1", body))
        run.flags_line = run.flags_line or no
    elif key == "input":
        run.inputs.append(body)
    elif key == "keys":
        run.keys += body.encode("latin-1", "backslashreplace").decode("unicode_escape")
    else:                               # limit / after (a note may follow)
        try:
            n = int(_NOTE.sub(r"\1", body).strip().replace("_", "").replace(",", ""), 0)
        except ValueError:
            raise CheckError(no, f"{key}: not a number of steps: {body!r}") from None
        if n <= 0:
            raise CheckError(no, f"{key}: must be at least 1")
        setattr(run, key, n)


def parse_check(body: str, line: int = 0) -> Check:
    """One check (the text after ``check:``)."""
    text, note = body.strip(), ""
    m = _NOTE.match(text)
    if m and not _in_string(text, m.start(2)):
        text, note = m.group(1).strip(), m.group(2).strip()
    if not text:
        raise CheckError(line, "an empty check")
    rest, cpu = text, None
    cm = re.match(r"(?:cpu|core|vpe)\s+(\d+)\s+(.*)$", rest, re.I)
    if cm:
        cpu, rest = int(cm.group(1)), cm.group(2)
    om = re.match(r"output\s+(is\s+not|is|contains|matches|==|!=)\s+(.*)$", rest, re.I)
    if om:
        op = {"==": "is", "!=": "is not"}.get(om.group(1).lower(), " ".join(
            om.group(1).lower().split()))
        return Check(line, text, ("output",), op, _string(om.group(2), line), cpu, note)
    if re.match(r"output\b", rest, re.I):
        raise CheckError(line, "output is / contains / matches \"TEXT\"")
    for op in _OPS:                     # (the two-character ones first)
        i = _find_op(rest, op)
        if i >= 0:
            left, right = rest[:i].strip(), rest[i + len(op):].strip()
            break
    else:
        raise CheckError(line, f"no comparison in {text!r} (== != < <= > >=)")
    subject = _subject(left, line)
    if not right:
        raise CheckError(line, f"nothing to compare {left} with")
    return Check(line, text, subject, op, _number(right, subject, line), cpu, note)


def _in_string(text: str, pos: int) -> bool:
    return text[:pos].count('"') % 2 == 1


def _find_op(text: str, op: str) -> int:
    """Where ``op`` is in ``text`` (not inside brackets / a longer op)."""
    depth = 0
    for i, ch in enumerate(text):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif depth == 0 and text.startswith(op, i):
            if len(op) == 1 and text[i + 1:i + 2] == "=":
                continue                # "<=" / ">=": not "<" / ">"
            if op in ("<", ">") and i and text[i - 1] in "<>=!":
                continue
            return i
    return -1


def _string(text: str, line: int) -> str:
    t = text.strip()
    if len(t) < 2 or t[0] != '"' or t[-1] != '"':
        raise CheckError(line, f'a "quoted" text, not {t!r}')
    return t[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")


_MEM = {"word": 4, "half": 2, "byte": 1, "float": 4, "double": 8}
_ELEMENT = {"b": 8, "h": 16, "w": 32, "d": 64}       # MSA data formats


def _subject(t: str, line: int) -> tuple:
    low = t.lower()
    if low in _COUNTS:
        return (low,)
    if low == "exit code":
        return ("exit",)
    if low == "stalls":
        return ("stalls", None)
    sm = re.fullmatch(r"stalls\s+([\w-]+)", low)
    if sm:
        return ("stalls", sm.group(1))
    if low in ("hi", "lo", "pc"):
        return ("special", low)
    fm = re.fullmatch(r"(\$f\d+)(?:\.([sd]))?", t, re.I)
    if fm:
        n = parse_fpr(fm.group(1))
        if n is None or n > 31:
            raise CheckError(line, f"not an FPU register: {fm.group(1)}")
        return ("fpr", n, (fm.group(2) or "").lower())
    wm = re.fullmatch(r"\$w(\d+)(?:\.([bhwd])\s*\[\s*(\d+)\s*\](?:\.(f))?)?", t, re.I)
    if wm:                                          # MSA: $wN, $wN.w[2], $wN.d[1].f
        n, fmt = int(wm.group(1)), (wm.group(2) or "").lower()
        if n > 31:
            raise CheckError(line, f"not a vector register: $w{n}")
        if not fmt:
            return ("vec", n, "", 0, False)
        size, index = _ELEMENT[fmt], int(wm.group(3))
        if index >= 128 // size:
            raise CheckError(line, f"$w{n}.{fmt} has elements 0..{128 // size - 1}")
        if wm.group(4) and fmt not in "wd":
            raise CheckError(line, f".f needs a word or a doubleword element: {t}")
        return ("vec", n, fmt, index, bool(wm.group(4)))
    if low == "msacsr":
        return ("special", "msacsr")
    if t.startswith("$"):
        return ("reg", _gpr(t, line))
    mm = re.fullmatch(r"(word|half|byte|float|double)\s*\[\s*(.+?)\s*\]", t, re.I)
    if mm:
        return ("mem", mm.group(1).lower(), _location(mm.group(2), line))
    raise CheckError(line, f"unknown subject {t!r} (output, $reg, hi, lo, pc, $fN[.s|.d], "
                           "$wN[.b|.h|.w|.d[I][.f]], msacsr, word / half / byte / float / double[LOC], cycles, instructions, "
                           "steps, stalls [KIND], exceptions, exit code)")


def _gpr(t: str, line: int) -> str:
    """A GPR, kept by its name: resolved with the core's ABI when the check
    is evaluated (o32's names; MIPS64: n64's; nanoMIPS: p32's — $t0 is $8,
    $12, $12); a name of no ABI is an error now."""
    if all(gpr_number(t, abi) is None for abi in (True, "n64", "p32")):
        raise CheckError(line, f"not a register: {t}")
    return t


def _reg(m, name: str) -> int:
    """The number of the GPR ``name`` on core ``m`` (its ABI's names)."""
    n = gpr_number(name, m.names)
    if n is None:
        raise _Unavailable(f"{name} is not a register name of this core's ABI "
                           f"({'p32' if m.names == 'p32' else 'n64' if m.names == 'n64' else 'o32'})")
    return n


def _location(t: str, line: int) -> tuple:
    """(base, offset): base = an int, a ``$reg`` number as ("reg", n) or a
    label as ("label", name)."""
    m = re.fullmatch(r"(.+?)\s*([+-])\s*(0x[0-9a-fA-F]+|\d+)", t)
    off = 0
    if m and not re.fullmatch(r"-?(0x[0-9a-fA-F]+|\d+)", t):
        t, off = m.group(1), int(m.group(3), 0) * (1 if m.group(2) == "+" else -1)
    if t.startswith("$"):
        return (("reg", _gpr(t, line)), off)
    try:
        return (int(t, 0) & M32, off)
    except ValueError:
        pass
    if re.fullmatch(r"[A-Za-z_.$][\w.$]*", t):
        return (("label", t), off)
    raise CheckError(line, f"not a location: {t!r} (a label, a number or $reg, +/- N)")


def _number(t: str, subject: tuple, line: int) -> Any:
    floating = subject[0] == "fpr" and subject[2] or subject[0] == "vec" and subject[4] or \
        subject[0] == "mem" and subject[1] in ("float", "double")
    if floating:
        try:
            return float(t)
        except ValueError:
            raise CheckError(line, f"not a number: {t!r}") from None
    if len(t) == 3 and t[0] == t[2] == "'":
        return ord(t[1])
    try:
        return int(t.replace("_", ""), 0)
    except ValueError:
        pass
    m = re.fullmatch(r"([A-Za-z_.][\w.$]*)\s*(?:([+-])\s*(0x[0-9a-fA-F]+|\d+))?", t)
    if m:
        off = int(m.group(3), 0) * (1 if m.group(2) == "+" else -1) if m.group(2) else 0
        return ("label", m.group(1), off)
    raise CheckError(line, f"not a number, 'c' or a label (+/- N): {t!r}")


# ------------------------------------------------------------------ running

def machine_for(run: Run, release: str, endian: str, mode: str, forwarding: bool,
                options) -> tuple[str, str, str, bool, Any]:
    """The machine of a run: its ``eMIPS:`` flags over the given one (the
    window's settings; the command line lays them over its own flags).
    Bad flags or settings that do not go together: a ValueError."""
    if not run.flags:
        return release, endian, mode, forwarding, options
    from .compat import explain, problems
    from .profile import read
    prof = read("# eMIPS: " + shlex.join(run.flags))
    given = prof.given
    release = prof.release if "release" in given else release
    endian = "little" if given.get("little") else endian
    mode = "pipeline" if given.get("pipeline") else mode
    forwarding = False if given.get("no_forwarding") else forwarding
    options = replace(options, **prof.options).validated()
    bad = problems(options, release)
    if bad:
        raise ValueError("these settings do not go together: " + explain(bad, options, release))
    return release, endian, mode, forwarding, options


class Execution:
    """One run in progress: :meth:`advance` runs it a slice at a time (the
    window, from a timer), :meth:`result` evaluates the checks when it is
    over (:func:`execute`: all at once)."""

    def __init__(self, session, run: Run, *, limit: int) -> None:
        self.s, self.run = session, run
        for ln in run.inputs:
            session.give_input(ln)
        if run.keys:
            session.type_keys(run.keys)
        session.stop_at_end = True
        self.budget = run.after or run.limit or limit
        self.done = 0
        self.last = None                # the StepResult that ended it

    @property
    def over(self) -> bool:
        return self.last is not None

    def advance(self, steps: int) -> bool:
        """Run up to ``steps`` more steps; True when the run is over."""
        if self.last is None:
            r = self.s.run(max_steps=min(steps, self.budget - self.done))
            self.done += r.steps
            if r.stopped or r.error or self.done >= self.budget:
                self.last = r
        return self.last is not None

    def result(self) -> RunResult:
        return _finish(self.s, self.run, self.last, self.budget)


def default_limit(session) -> int:
    """The step limit of a run that names none: the ``run`` command's
    (100 000; on the Malta 1 000 000 — its serial output is slow)."""
    return 1_000_000 if session.board is not None else 100_000


def execute(session, run: Run, *, limit: int) -> RunResult:
    """Run the loaded ``session`` as ``run`` says (its inputs are given,
    keys typed here) and evaluate its checks; ``limit``: the step limit
    when the run names none."""
    ex = Execution(session, run, limit=limit)
    ex.advance(ex.budget)
    return ex.result()


def _finish(s, run: Run, r, budget: int) -> RunResult:
    res = RunResult(run)
    res.output = s.output
    if run.after is not None:
        res.finished = r.stopped in ("exit", "end") and not r.error
        res.why = "" if res.finished else f"stopped after {s.steps} steps, as asked"
    elif r.error:
        res.why = f"stopped: {r.error}"
    elif r.stopped in ("exit", "end"):
        res.finished = True
    elif r.stopped == "input":
        res.why = ("the program waits for input: give it with input: lines "
                   f"({len(run.inputs)} given)")
    elif r.stopped in ("breakpoint", "cursor"):
        res.why = "stopped at a breakpoint"
    else:
        res.why = f"did not finish in {budget} steps (limit: N to allow more)"
    if s.pipeline is not None and r.stopped == "end":
        s.pipeline.drain(s.code_addresses())
    res.summary = _summary(s)
    for c in run.checks:
        res.outcomes.append(evaluate(s, c))
    return res


def _summary(s) -> str:
    p = s.pipeline
    if p is not None:
        stalls = sum(p.stalls.values())
        return (f"{p.cycles} cycles, {p.retired} instructions retired"
                + (f", {stalls} stall cycles" if stalls else ""))
    return f"{s.steps} steps, {s.machine.cycles} cycles"


def evaluate(s, c: Check) -> Outcome:
    """One check against the session's state after the run."""
    if c.cpu is not None:
        if not 0 <= c.cpu < len(s.machines):
            return Outcome(c, False, f"there is no cpu {c.cpu}")
        keep = s.cpu
        s.select(c.cpu)
        try:
            return _evaluate(s, c)
        finally:
            s.select(keep)
    return _evaluate(s, c)


def _evaluate(s, c: Check) -> Outcome:
    kind = c.subject[0]
    if kind == "output":
        return _output(c, s.output)
    try:
        value, bits, shown = _read(s, c.subject)
    except _Unavailable as exc:
        return Outcome(c, False, str(exc))
    want = c.value
    if isinstance(want, tuple):                     # a label (+ N): its address
        sym = s.program.symbols.get(want[1]) if s.program is not None else None
        if sym is None:
            return Outcome(c, False, f"no label {want[1]!r} in the program")
        want = (sym + want[2]) & M32
    if isinstance(value, float):
        target = _as_format(float(want), bits)
        ok = _compare(value, c.op, target)
    elif bits is None:                              # a count
        ok = _compare(value, c.op, want)
    elif c.op in ("==", "!="):
        mask = (1 << bits) - 1
        ok = _compare(value & mask, c.op, want & mask)
    else:
        ok = _compare(_signed(value, bits), c.op, want)
    return Outcome(c, ok, shown)


class _Unavailable(Exception):
    pass


def _output(c: Check, out: str) -> Outcome:
    shown = quote(out)
    if c.op in ("is", "is not"):
        same = out.rstrip() == c.value.rstrip()
        return Outcome(c, same if c.op == "is" else not same, shown)
    if c.op == "contains":
        return Outcome(c, c.value in out, shown)
    try:
        return Outcome(c, re.search(c.value, out) is not None, shown)
    except re.error as exc:
        return Outcome(c, False, f"not a regular expression: {exc}")


def quote(text: str, width: int = 60) -> str:
    """The output as the reports show it: quoted, cut after ``width``."""
    t = text.rstrip()
    q = repr(t if len(t) <= width else t[:width] + "...")
    return '"' + q[1:-1].replace('"', '\\"') + '"' if q[0] == "'" else q


def _read(s, subject: tuple) -> tuple[Any, int | None, str]:
    """(the value, its width in bits — None for a count / a float's format
    bits, how to show it)."""
    kind = subject[0]
    m = s.machine
    p = s.pipeline
    if kind == "cycles":
        n = p.cycles if p is not None else m.cycles
        return n, None, str(n)
    if kind == "instructions":
        n = p.retired if p is not None else s.steps
        return n, None, str(n)
    if kind == "steps":
        return s.steps, None, str(s.steps)
    if kind == "exceptions":
        n = sum(1 for mm in s.machines for e in mm.exception_log if "vector" in e)
        return n, None, str(n)
    if kind == "exit":
        code = s.services.exit_code if s.services is not None else None
        if code is None:
            raise _Unavailable("the program did not exit with a syscall")
        return code, None, str(code)
    if kind == "stalls":
        if p is None:
            raise _Unavailable("stalls: pipeline mode only (eMIPS: --pipeline)")
        n = sum(p.stalls.values()) if subject[1] is None else p.stalls.get(subject[1], 0)
        return n, None, str(n)
    if kind == "reg":
        v = m.read_reg(_reg(m, subject[1]))
        return v, 32, _hex(v, 32)
    if kind == "special" and subject[1] == "msacsr":
        if not m.msa:
            raise _Unavailable("msacsr: this core has no MSA (eMIPS: --msa msa)")
        return m.msacsr, 32, _hex(m.msacsr, 32)
    if kind == "vec":
        if not m.msa:
            raise _Unavailable(f"$w{subject[1]}: this core has no MSA (eMIPS: --msa msa)")
        _, n, fmt, index, floating = subject
        v = m.wh[n] << 64 | m.fpr[n]
        if not fmt:
            return v, 128, f"0x{v:032x}"
        size = _ELEMENT[fmt]
        e = (v >> (index * size)) & ((1 << size) - 1)
        if floating:
            f = struct.unpack(">f" if size == 32 else ">d", e.to_bytes(size // 8, "big"))[0]
            return f, size, repr(f)
        return e, size, _hex(e, size)
    if kind == "special":
        v = {"hi": lambda: m.hi, "lo": lambda: m.lo,
             "pc": lambda: s.current_pc() if s.current_pc() is not None else m.pc}[subject[1]]()
        return v & M32, 32, _hex(v & M32, 32)
    if kind == "fpr":
        from .sim import fpr_get
        n, fmt = subject[1], subject[2]
        if fmt == "d":
            try:
                raw = fpr_get(m, n, "D")
            except Exception:
                raise _Unavailable(f"$f{n} holds no double (an odd register, Status.FR = 0)")
            v = struct.unpack(">d", raw.to_bytes(8, "big"))[0]
            return v, 64, repr(v)
        raw = m.fpr[n] & M32
        if fmt == "s":
            v = struct.unpack(">f", raw.to_bytes(4, "big"))[0]
            return v, 32, repr(v)
        return raw, 32, _hex(raw, 32)
    if kind == "mem":
        width = _MEM[subject[1]]
        addr = _address(s, subject[2])
        try:
            data = bytes(m.read_byte(addr + i) for i in range(width))
        except Exception as exc:
            raise _Unavailable(f"0x{addr:08x}: {exc}") from None
        order = "big" if m.big_endian else "little"
        if subject[1] in ("float", "double"):
            v = struct.unpack((">" if order == "big" else "<") + ("f" if width == 4 else "d"),
                              data)[0]
            return v, width * 8, repr(v)
        v = int.from_bytes(data, order)
        return v, width * 8, _hex(v, width * 8)
    raise _Unavailable(f"cannot read {subject}")


def _address(s, loc: tuple) -> int:
    base, off = loc
    if isinstance(base, tuple):
        if base[0] == "reg":
            base = s.machine.read_reg(_reg(s.machine, base[1]))
        else:
            sym = s.program.symbols.get(base[1]) if s.program is not None else None
            if sym is None:
                raise _Unavailable(f"no label {base[1]!r} in the program")
            base = sym
    return (base + off) & M32


def _hex(v: int, bits: int) -> str:
    signed = _signed(v, bits)
    return f"{signed} (0x{v:0{bits // 4}x})" if signed != v or v > 9 else str(v)


def _signed(v: int, bits: int) -> int:
    return v - (1 << bits) if v >> (bits - 1) & 1 else v


def _as_format(x: float, bits: int) -> float:
    """``x`` rounded to the single / double format the value is in."""
    if bits == 32:
        return struct.unpack(">f", struct.pack(">f", x))[0]
    return x


def _compare(a, op: str, b) -> bool:
    return {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b,
            ">": a > b, ">=": a >= b}[op]


# ------------------------------------------------------------------ reporting

def report(results: list[RunResult], say: Callable[[str], None], *,
           header: bool = True) -> tuple[int, int]:
    """Print every run's outcomes (``header``: a ``--- run`` line before
    each of several); (checks passed, checks) — the "the program
    finishes" check of each run included."""
    passed = total = 0
    for i, res in enumerate(results):
        if header and (len(results) > 1 or res.run.name):
            say(f"--- run {i + 1}" + (f": {res.run.name}" if res.run.name else ""))
        say(f"--- {res.summary}")
        if res.run.after is None:
            total += 1
            if res.finished:
                passed += 1
                say("PASS  the program finishes")
            else:
                say(f"FAIL  the program finishes: {res.why}")
        for o in res.outcomes:
            total += 1
            passed += o.passed
            note = f" ({o.check.note})" if o.check.note else ""
            say(f"{'PASS' if o.passed else 'FAIL'}  {o.check.text}{note}"
                + ("" if o.passed else f": {o.found}"))
    return passed, total
