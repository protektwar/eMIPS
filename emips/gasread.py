"""GNU as's input layer, shared by eMIPS's three assemblers (LF12): what
gas does to the statements before an assembler sees them — macros,
repetitions, conditional assembly, included files, the message
directives and numeric local labels.  Ported from binutils 2.42's
``gas/macro.c`` (``buffer_and_nest``, ``get_token``, ``getstring``,
``get_any_string``, ``do_formals``, ``define_macro``, ``sub_actual``,
``macro_expand_body``, ``macro_expand``, ``check_macro``,
``delete_macro``, ``expand_irp``), ``gas/cond.c`` (``s_if`` ...
``s_endif``, ``ignore_input``, ``cond_finish_check``,
``cond_exit_macro``) and ``gas/read.c`` (``read_a_source_file``'s order:
labels, then a conditional check, then a pseudo-op or a macro, then an
instruction; ``s_macro``, ``s_mexit``, ``s_purgem``, ``s_rept`` /
``do_repeat``, ``s_irp``, ``s_include`` / ``search_and_open``,
``s_err``, ``s_errwarn``, ``s_print``, ``s_end``, ``s_altmacro``,
``s_bad_end``, ``s_fail``); names follow theirs.

:class:`Reader` turns a source into :class:`Stmt` s, one statement each
(``;`` separates statements, as on every eMIPS target), with the
expansions done and numeric local labels named (``N:`` → ``.LN.k``;
``Nb`` the latest definition, ``Nf`` the next — gas's
``fb_label_name``, resolved as the statements are read, so a label in a
``.rept`` body is a new one every time).  An assembler gives it
:class:`Hooks`: the absolute value of an expression (``.if``, ``.rept``,
``%expr`` in alternate macro mode) with the symbols defined so far, whether
a symbol is defined (``.ifdef``), whether a directive is its own (a
macro may not take a pseudo-op's name, read.c), and its comment
stripper.

Choices (``sim.DEVIATIONS["GNU as directives"]``): a relative
``.include`` / ``.incbin`` path is looked up in the program's folder
(gas: the current directory), then in the include folders; the first
error stops the assembly (gas goes on and reports them all); ``.print``
collects its strings (``Reader.printed``) for the caller to show;
MRI mode and its ``.mri`` / ``.common`` / ``\\0`` qualifiers are not
modelled.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol

from .assembler import AsmError

#: a name's first / later characters (gas's is_name_beginner /
#: is_part_of_name with the default lex table: '$' is a name character)
_BEGIN = re.compile(r"[A-Za-z_.$]")
_PART = re.compile(r"[\w.$]")
_LABEL = re.compile(r"\s*([A-Za-z_.$][\w.$]*)\s*:(?!=)")
_FB_DEF = re.compile(r"\s*(\d+)\s*:")
_FB_REF = re.compile(r"(?<![\w.$'])(\d+)([fb])(?![\w.$])")
_QUOTED = re.compile(r"(\"(?:\\.|[^\"\\])*\"?|'(?:\\.|[^'])')")
#: the conditional directives (cond.c; read.c's potable)
_IF_OPS = {".if": "ne", ".ifne": "ne", ".ifeq": "eq", ".iflt": "lt", ".ifle": "le",
           ".ifge": "ge", ".ifgt": "gt"}
_COND = frozenset((*_IF_OPS, ".ifdef", ".ifndef", ".ifnotdef", ".ifb", ".ifnb", ".ifc",
                   ".ifnc", ".ifeqs", ".ifnes", ".elseif", ".else", ".elsec", ".endif",
                   ".endc"))
#: the directives this layer handles
DIRECTIVES = frozenset((*_COND, ".macro", ".endm", ".mexit", ".exitm", ".purgem", ".rept",
                        ".rep", ".irp", ".irep", ".irpc", ".irepc", ".endr", ".include",
                        ".err", ".error", ".warning", ".print", ".end", ".altmacro",
                        ".noaltmacro", ".fail", ".linefile"))
#: read.c: when a conditional ignores input, these are still read
_NEVER_IGNORED = re.compile(r"\.(if|else|endif|endc|linefile)", re.I)


class Hooks(Protocol):
    def absolute(self, text: str) -> int: ...        # an absolute expression (errors raise)
    def defined(self, name: str) -> bool: ...        # .ifdef's test
    def known(self, directive: str) -> bool: ...     # the assembler's own pseudo-op
    def strip(self, line: str) -> str: ...           # its comment stripper
    onechar: bool                                    # app.c's 'c -> its number


@dataclass
class Stmt:
    """One statement for the assembler: ``text`` (comments stripped, local
    labels named), the main source's ``line`` and its ``raw`` text (the
    source map, error messages), ``where`` = (file, line) when the
    statement comes from an included file."""
    text: str
    line: int
    raw: str
    where: tuple[str, int] | None = None
    #: from a macro / .rept / .irp expansion or an included file (``raw``
    #: is then the invoking line, not this statement)
    expanded: bool = False


@dataclass
class _Item:
    text: str
    term: str                   # its terminator: ';' or '\n'
    file: str                   # "" = the main source
    line: int
    expanded: bool = False


@dataclass
class _Buf:
    items: list[_Item]
    kind: str = "file"          # file / include / macro / repeat
    pos: int = 0
    origin: tuple[int, str] | None = None    # an include: the .include's (line, raw)


@dataclass
class _Formal:
    name: str
    index: int = 0
    kind: str = "optional"      # optional / required / vararg
    default: str = ""
    actual: str = ""


@dataclass
class _Macro:
    name: str
    formals: list[_Formal] = field(default_factory=list)
    body: str = ""
    file: str = ""
    line: int = 0

    def formal(self, name: str) -> _Formal | None:
        return next((f for f in self.formals if f.name == name), None)


@dataclass
class _Cframe:
    """cond.c's conditional_frame."""
    line: int
    else_line: int = 0
    else_seen: bool = False
    ignoring: bool = False
    dead_tree: bool = False
    macro_nest: int = 0


def c_comments(source: str) -> str:
    """app.c: ``/* ... */`` comments become a space (their newlines stay);
    strings, ``'c`` constants and ``#`` comments are left alone."""
    out = []
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if c == '"':
            j = i + 1
            while j < n and source[j] != '"' and source[j] != "\n":
                j += 2 if source[j] == "\\" else 1
            out.append(source[i:j + 1])
            i = j + 1
        elif c == "'":
            j = i + 1
            if j < n and source[j] == "\\":
                j += 1
            out.append(source[i:j + 1])
            i = j + 1
        elif c == "#":
            j = source.find("\n", i)
            j = n if j < 0 else j
            out.append(source[i:j])
            i = j
        elif source.startswith("/*", i):
            j = source.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(" " + "\n" * source.count("\n", i, j))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def statements(text: str) -> list[str]:
    """A line's statements (``;`` separates them outside quotes)."""
    out, cur, quote = [], [], None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote and (len(cur) < 2 or cur[-2] != "\\"):
                quote = None
        elif ch == '"':
            quote = ch
            cur.append(ch)
        elif ch == ";":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


_SYMCH = re.compile(r"[\w.$]")
_ESCAPES = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def onechar(line: str) -> str:
    """app.c's LEX_IS_ONECHAR_QUOTE: outside strings ``'x`` (``'\\e``: an
    escape, process_escape) becomes its decimal number, a closing quote
    right after it dropped — everywhere, before anything reads the line
    (so ``'\\c'`` in a macro body is 99 before ``\\c`` could be a parameter)."""
    out = []
    i, n = 0, len(line)
    quote = False
    while i < n:
        c = line[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(line[i + 1])
                i += 2
                continue
            if c == '"':
                quote = False
            i += 1
            continue
        if c == '"':
            quote = True
            out.append(c)
            i += 1
            continue
        if c == "'":
            ch = line[i + 1] if i + 1 < n else "\0"
            i += 2
            if ch == "\\" and i < n:
                ch = _ESCAPES.get(line[i], line[i])
                i += 1
            if i < n and line[i] == "'":
                i += 1
            out.append(str(ord(ch) & 0xFF))
            continue
        out.append(c)
        i += 1
    return "".join(out)


def scrub(rest: str) -> str:
    """app.c's scrubbing of a statement's operands: whitespace kept only
    between two name characters (one space), ``'c`` → its number."""
    out = []
    i = 0
    quote = False
    rest = rest.strip()
    while i < len(rest):
        ch = rest[i]
        if quote:
            out.append(ch)
            if ch == "\\" and i + 1 < len(rest):
                out.append(rest[i + 1])
                i += 2
                continue
            if ch == '"':
                quote = False
            i += 1
            continue
        if ch == '"':
            quote = True
            out.append(ch)
            i += 1
            continue
        if ch == "'":                       # (LEX_IS_ONECHAR_QUOTE: 'x -> its number)
            c = rest[i + 1] if i + 1 < len(rest) else "\0"
            i += 2
            if c == "\\" and i < len(rest):
                c = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}.get(rest[i], rest[i])
                i += 1
            if i < len(rest) and rest[i] == "'":
                i += 1
            out.append(str(ord(c) & 0xFF))
            continue
        if ch in " \t":
            j = i
            while j < len(rest) and rest[j] in " \t":
                j += 1
            if out and j < len(rest) and _SYMCH.match(out[-1]) and (
                    _SYMCH.match(rest[j]) or rest[j] == "'"):
                out.append(" ")
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _skip_white(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t":
        i += 1
    return i


def _skip_comma(s: str, i: int) -> int:
    i = _skip_white(s, i)
    if i < len(s) and s[i] == ",":
        i += 1
    return _skip_white(s, i)


class Reader:
    """gas's statement reader over ``source`` (see the module docstring)."""

    def __init__(self, source: str, hooks: Hooks, *, path: str | None = None,
                 include_dirs: tuple[str, ...] | list[str] = (),
                 warn: Callable[[int, str], None] | None = None) -> None:
        self.hooks = hooks
        self.lines = source.splitlines()
        self.base = os.path.dirname(os.path.abspath(path)) if path else os.getcwd()
        self.include_dirs = list(include_dirs)
        self._warn = warn
        self.printed: list[str] = []
        self.included: list[str] = []
        self.files: dict[str, list[str]] = {}         # an included file's lines
        self.origins: dict[str, tuple[int, str]] = {}  # its first .include (line, raw)
        self.macros: dict[str, _Macro] = {}
        self.macro_number = 0                         # \@
        self.macro_nest = 0
        self.alternate = False                        # .altmacro
        self.cframes: list[_Cframe] = []
        self.fb: dict[str, int] = {}                  # numeric local labels defined
        self.loccnt = 0                               # LOCAL's .LL%04x
        self.stack: list[_Buf] = [_Buf(self._items(source, "", 1))]
        self.current: Stmt = Stmt("", 0, "")
        self.ended = False

    # -- the input ------------------------------------------------------------

    def _items(self, text: str, file: str, first: int) -> list[_Item]:
        """A buffer's statements: app.c's comments removed, each line's
        statements with their terminators."""
        out = []
        conv = getattr(self.hooks, "onechar", True)
        for k, line in enumerate(c_comments(text).split("\n")):
            if conv and "'" in line:
                line = onechar(line)
            parts = statements(self.hooks.strip(line))
            for j, p in enumerate(parts):
                out.append(_Item(p, ";" if j < len(parts) - 1 else "\n", file, first + k))
        return out

    def _place(self, item: _Item) -> Stmt:
        if not item.file:
            n = item.line
            raw = self.lines[n - 1] if 0 < n <= len(self.lines) else ""
            return Stmt(item.text, n, raw, None, item.expanded)
        n, raw = self.origins.get(item.file, (0, ""))
        return Stmt(item.text, n, raw, (item.file, item.line), True)

    def _next_item(self) -> _Item | None:
        """input_scrub_next_buffer: the next statement, popping finished
        buffers (the end of a macro expansion checks its conditionals)."""
        while self.stack:
            b = self.stack[-1]
            if b.pos < len(b.items):
                b.pos += 1
                return b.items[b.pos - 1]
            self.stack.pop()
            if b.kind == "macro":
                self._cond_finish_check(self.macro_nest)
                self.macro_nest -= 1
        return None

    def error(self, msg: str, item: _Item | None = None):
        st = self._place(item) if item is not None else self.current
        if st.where:
            msg = f"{os.path.basename(st.where[0])}:{st.where[1]}: {msg}"
        raise AsmError(st.line, msg, st.raw)

    def warn(self, msg: str) -> None:
        if self._warn is not None:
            st = self.current
            if st.where:
                msg = f"{os.path.basename(st.where[0])}:{st.where[1]}: {msg}"
            self._warn(st.line, msg)

    def _include(self, items: list[_Item], kind: str) -> None:
        """input_scrub_include_sb: read ``items`` next."""
        if kind == "macro":
            self.macro_nest += 1
        self.stack.append(_Buf(items, kind))

    def _expansion(self, texts: list[str], file: str, line: int, kind: str) -> None:
        """An expansion (sb_scrub_and_add_sb: scrubbed again): each copy of
        the body numbered from the body's first line (the .linefile
        buffer_and_nest puts in front of it)."""
        items: list[_Item] = []
        for text in texts:
            items += self._items(text, file, line)
        for it in items:
            it.expanded = True
        self._include(items, kind)

    # -- statements --------------------------------------------------------------

    def __iter__(self) -> Iterator[Stmt]:
        while not self.ended:
            item = self._next_item()
            if item is None:
                break
            self.current = self._place(item)
            for st in self._statement(item):
                yield st
        if not self.ended:
            self.current = Stmt("", len(self.lines), self.lines[-1] if self.lines else "")
            self._cond_finish_check(-1)

    def _statement(self, item: _Item) -> Iterator[Stmt]:
        text = item.text
        if self.cframes and self.cframes[-1].ignoring:
            m = re.match(r"\s*(\.[\w.$]+)", text)
            if not m or not _NEVER_IGNORED.match(m.group(1)):
                return                          # (HANDLE_CONDITIONAL_ASSEMBLY)
        labels = []
        rest = text
        while True:                             # (the labels first)
            m = _LABEL.match(rest) or _FB_DEF.match(rest)
            if not m:
                break
            labels.append(rest[:m.end()])
            rest = rest[m.end():]
        body = rest.strip()
        m = re.match(r"[A-Za-z_.$][\w.$]*", body)
        name = m.group(0) if m else ""
        lname = name.lower()
        args = body[m.end():] if m else ""
        if lname in DIRECTIVES and not self.hooks.known(lname):   # (md tables win: MIPS .end)
            if lname == ".macro" and labels and not labels[-1].strip()[0].isdigit():
                label = _LABEL.match(labels[-1]).group(1)
                pre = "".join(labels[:-1])
                if pre.strip():
                    yield self._out(pre)
                self.s_macro(args, label)
                return
            if labels:
                yield self._out("".join(labels))
            yield from self._directive(lname, args)
            return
        if lname and self.macros and (not lname.startswith(".") or not self.hooks.known(lname)):
            mac = self.macros.get(lname)
            if mac is not None:
                if labels:
                    yield self._out("".join(labels))
                self._expand(mac, scrub(args))
                return
        if body or labels:
            yield self._out(text)

    def ignoring(self) -> bool:
        return bool(self.cframes) and self.cframes[-1].ignoring

    def _out(self, text: str) -> Stmt:
        """The statement as the assembler gets it: local labels named."""
        st = self.current
        return Stmt(self.fb_names(text), st.line, st.raw, st.where, st.expanded)

    def fb_names(self, text: str) -> str:
        """Numeric local labels (read.c's LOCAL_LABELS_FB, symbols.c's
        fb_label_name): the definitions at the start, then the references."""
        if not any(c.isdigit() for c in text):
            return text
        head = []
        rest = text
        while True:
            m = _FB_DEF.match(rest)
            if m:
                num = m.group(1)
                k = self.fb.get(num, 0)
                self.fb[num] = k + 1
                head.append(f"{rest[:m.start(1)]}.L{num}.{k}:")
                rest = rest[m.end():]
                continue
            m = _LABEL.match(rest)
            if m:
                head.append(rest[:m.end()])
                rest = rest[m.end():]
                continue
            break

        def ref(m: re.Match) -> str:
            num, way = m.group(1), m.group(2)
            k = self.fb.get(num, 0)
            if way == "b":
                return f".L{num}.{k - 1}" if k else m.group(0)
            return f".L{num}.{k}"

        parts = _QUOTED.split(rest)
        rest = "".join(p if j % 2 else _FB_REF.sub(ref, p) for j, p in enumerate(parts))
        return "".join(head) + rest

    # -- directives -----------------------------------------------------------------

    def _directive(self, d: str, args: str) -> Iterator[Stmt]:
        args = args.strip()
        if d in _IF_OPS:
            self.s_if(_IF_OPS[d], args)
        elif d in (".ifdef", ".ifndef", ".ifnotdef"):
            self.s_ifdef(d == ".ifdef", args)
        elif d in (".ifb", ".ifnb"):
            self.s_ifb(d == ".ifb", args)
        elif d in (".ifc", ".ifnc"):
            self.s_ifc(d == ".ifnc", args)
        elif d in (".ifeqs", ".ifnes"):
            self.s_ifeqs(d == ".ifnes", args)
        elif d == ".elseif":
            self.s_elseif(args)
        elif d in (".else", ".elsec"):
            self.s_else(args)
        elif d in (".endif", ".endc"):
            self.s_endif(args)
        elif d == ".linefile":
            pass
        elif d == ".macro":
            self.s_macro(args, None)
        elif d in (".endm", ".endr"):
            self.warn(f".end{'r' if d == '.endr' else 'm'} encountered without preceding "
                      + (".rept, .irp, or .irpc" if d == ".endr" else ".macro"))
            self._empty(args)
        elif d in (".mexit", ".exitm"):
            self.s_mexit()
        elif d == ".purgem":
            self.s_purgem(args)
        elif d in (".rept", ".rep"):
            self.s_rept(args)
        elif d in (".irp", ".irep", ".irpc", ".irepc"):
            self.s_irp(d in (".irpc", ".irepc"), args)
        elif d == ".include":
            self.s_include(args)
        elif d == ".err":
            self.error(".err encountered")
        elif d in (".error", ".warning"):
            self.s_errwarn(d == ".error", args)
        elif d == ".print":
            self.printed.append(self._c_string(args, ".print")[0])
        elif d == ".fail":
            v = self.hooks.absolute(args)
            if v >= 500:
                self.warn(f".fail {v} encountered")
            else:
                self.error(f".fail {v} encountered")
        elif d == ".end":
            self.ended = True
        elif d in (".altmacro", ".noaltmacro"):
            self.alternate = d == ".altmacro"
            self._empty(args)
        return iter(())

    def _empty(self, args: str) -> None:
        """demand_empty_rest_of_line."""
        if args.strip():
            self.error("junk at end of line, first unrecognized character is "
                       f"`{args.strip()[0]}'")

    def _c_string(self, s: str, what: str) -> tuple[str, str]:
        """demand_copy_C_string: a "..." string (escapes decoded) and what
        follows it."""
        s = s.lstrip()
        if not s.startswith('"'):
            self.error("expected string")
        out, i = [], 1
        while i < len(s) and s[i] != '"':
            c = s[i]
            if c == "\\" and i + 1 < len(s):
                i += 1
                e = s[i]
                if e in "01234567":
                    j = i
                    while j < len(s) and j < i + 3 and s[j] in "01234567":
                        j += 1
                    out.append(chr(int(s[i:j], 8) & 0xFF))
                    i = j
                    continue
                if e == "x":
                    j = i + 1
                    while j < len(s) and s[j] in "0123456789abcdefABCDEF":
                        j += 1
                    out.append(chr(int(s[i + 1:j] or "0", 16) & 0xFF))
                    i = j
                    continue
                out.append({"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}.get(e, e))
            else:
                out.append(c)
            i += 1
        if i >= len(s):
            self.error("missing close quote; (assumed)")
        return "".join(out), s[i + 1:]

    # -- conditionals (cond.c) ----------------------------------------------------

    def _frame(self) -> _Cframe:
        """initialize_cframe."""
        prev = self.cframes[-1] if self.cframes else None
        return _Cframe(self.current.line, dead_tree=prev is not None and prev.ignoring,
                       macro_nest=self.macro_nest)

    def _test(self, op: str, v: int) -> bool:
        return {"eq": v == 0, "ne": v != 0, "lt": v < 0, "le": v <= 0, "ge": v >= 0,
                "gt": v > 0}[op]

    def _value(self, args: str, what: str) -> int:
        try:
            return self.hooks.absolute(args)
        except AsmError as exc:
            if "absolute" in exc.message or "constant" in exc.message:
                self.error(f"non-constant expression in \"{what}\" statement")
            raise

    def s_if(self, op: str, args: str) -> None:
        cf = self._frame()
        if self.ignoring():
            t = False
        else:
            t = self._test(op, self._value(args, ".if"))
        cf.ignoring = cf.dead_tree or not t
        self.cframes.append(cf)

    def s_ifdef(self, test_defined: bool, args: str) -> None:
        m = re.match(r"\s*([A-Za-z_.$][\w.$]*|\"[^\"]*\")\s*$", args)
        if not m:
            self.error("invalid identifier for \".ifdef\"")
        cf = self._frame()
        if cf.dead_tree:
            cf.ignoring = True
        else:
            name = self.fb_names(m.group(1).strip('"'))
            cf.ignoring = not (test_defined ^ (not self.hooks.defined(name)))
        self.cframes.append(cf)

    def s_ifb(self, test_blank: bool, args: str) -> None:
        cf = self._frame()
        cf.ignoring = True if cf.dead_tree else test_blank == bool(args.strip())
        self.cframes.append(cf)

    @staticmethod
    def _mri_string(s: str, i: int, term: str) -> tuple[str, int]:
        """get_mri_string."""
        i = _skip_white(s, i)
        if i < len(s) and s[i] == "'":
            out = []
            i += 1
            while i < len(s):
                out.append(s[i])
                i += 1
                if out[-1] == "'":
                    if i >= len(s) or s[i] != "'":
                        out.pop()
                        break
                    i += 1
            return "".join(out), _skip_white(s, i)
        j = i
        while j < len(s) and s[j] != term:
            j += 1
        return s[i:j].rstrip(" \t"), j

    def s_ifc(self, negate: bool, args: str) -> None:
        s1, i = self._mri_string(args, 0, ",")
        if i >= len(args) or args[i] != ",":
            self.error("bad format for ifc or ifnc")
        s2, _ = self._mri_string(args, i + 1, ";")
        cf = self._frame()
        cf.ignoring = cf.dead_tree or not ((s1 == s2) ^ negate)
        self.cframes.append(cf)

    def s_ifeqs(self, negate: bool, args: str) -> None:
        s1, rest = self._c_string(args, ".ifeqs")
        rest = rest.lstrip()
        if not rest.startswith(","):
            self.error(".ifeqs syntax error")
        s2, rest = self._c_string(rest[1:], ".ifeqs")
        cf = self._frame()
        cf.ignoring = cf.dead_tree or not ((s1 == s2) ^ negate)
        self.cframes.append(cf)
        self._empty(rest)

    def s_elseif(self, args: str) -> None:
        if not self.cframes:
            self.error("\".elseif\" without matching \".if\"")
        cf = self.cframes[-1]
        if cf.else_seen:
            self.error("\".elseif\" after \".else\"")
        cf.else_line = self.current.line
        cf.dead_tree |= not cf.ignoring
        cf.ignoring = cf.dead_tree
        if not cf.ignoring:
            cf.ignoring = cf.dead_tree or not self._test("ne", self._value(args, ".elseif"))

    def s_else(self, args: str) -> None:
        if not self.cframes:
            self.error("\".else\" without matching \".if\"")
        cf = self.cframes[-1]
        if cf.else_seen:
            self.error("duplicate \".else\"")
        cf.else_line = self.current.line
        cf.ignoring = cf.dead_tree or not cf.ignoring
        cf.else_seen = True
        self._empty(args)

    def s_endif(self, args: str) -> None:
        if not self.cframes:
            self.error("\".endif\" without \".if\"")
        self.cframes.pop()
        self._empty(args)

    def _cond_finish_check(self, nest: int) -> None:
        if self.cframes and self.cframes[-1].macro_nest >= nest:
            self.error("end of macro inside conditional" if nest >= 0
                       else "end of file inside conditional "
                       f"(the conditional starts on line {self.cframes[-1].line})")

    def _cond_exit_macro(self, nest: int) -> None:
        while self.cframes and self.cframes[-1].macro_nest >= nest:
            self.cframes.pop()

    # -- macros (macro.c) --------------------------------------------------------------

    def buffer_and_nest(self, start: str | None, end: str) -> tuple[str, str, int] | None:
        """Statements up to the matching ``.endm`` / ``.endr`` (nested
        ones counted): the body's text, its file and first line; None at
        the end of the input."""
        depth = 1
        out: list[str] = []
        first: _Item | None = None
        starts = ("irepc", "irep", "irpc", "rept", "irp", "rep") if start is None else (start,)
        while True:
            item = self._next_item()
            if item is None:
                return None
            if first is None:
                first = item
            s = item.text
            i = 0
            keep = 0
            while True:                         # (skip the labels)
                m = _LABEL.match(s, i)
                if not m:
                    break
                i = keep = m.end()
            i = _skip_white(s, i)
            if i < len(s) and s[i] == ".":
                word = re.match(r"\.([\w.$]*)", s[i:]).group(1).lower()
                if word in starts:
                    depth += 1
                if word == end:
                    depth -= 1
                    if depth == 0:
                        out.append(s[:keep])
                        return "".join(out), first.file, first.line
            out.append(s + item.term)

    def _get_token(self, s: str, i: int) -> tuple[str, int]:
        j = i
        if i < len(s) and _BEGIN.match(s[i]):
            j = i + 1
            while j < len(s) and _PART.match(s[j]):
                j += 1
        name = s[i:j]
        if self.alternate and j < len(s) and s[j] == "&":
            j += 1
        return name, j

    def _getstring(self, s: str, i: int) -> tuple[str, int]:
        acc = []
        while i < len(s) and (s[i] == '"' or (s[i] == "<" and self.alternate)
                              or (s[i] == "'" and self.alternate)):
            if s[i] == "<":
                nest = 0
                i += 1
                while i < len(s) and (s[i] != ">" or nest):
                    if s[i] == "!":
                        i += 1
                        if i < len(s):
                            acc.append(s[i])
                        i += 1
                    else:
                        if s[i] == ">":
                            nest -= 1
                        if s[i] == "<":
                            nest += 1
                        acc.append(s[i])
                        i += 1
                i += 1
            else:
                tchar = s[i]
                escaped = 0
                i += 1
                while i < len(s):
                    escaped = escaped ^ 1 if s[i - 1] == "\\" else 0
                    if self.alternate and s[i] == "!":
                        i += 1
                        if i < len(s):
                            acc.append(s[i])
                        i += 1
                    elif escaped and s[i] == tchar:
                        acc.append(tchar)
                        i += 1
                    else:
                        if s[i] == tchar:
                            i += 1
                            if i >= len(s) or s[i] != tchar:
                                break
                        acc.append(s[i])
                        i += 1
        return "".join(acc), i

    def _issep(self, c: str) -> bool:
        return c in " \t,\";)(" or (self.alternate and c in "<>")

    def get_any_string(self, s: str, i: int) -> tuple[str, int]:
        i = _skip_white(s, i)
        if i >= len(s):
            return "", i
        if len(s) > i + 2 and s[i + 1] == "'" and s[i] in "bBqQhHdD":
            j = i
            while j < len(s) and not self._issep(s[j]):
                j += 1
            return s[i:j], j
        if s[i] == "%" and self.alternate:
            m = re.match(r"[^,]*", s[i + 1:])
            text = m.group(0)
            try:
                v = self.hooks.absolute(text)
            except AsmError:
                self.error("% operator needs absolute expression")
            return str(v), i + 1 + len(text)
        if s[i] == '"' or (s[i] == "<" and self.alternate) or (self.alternate and s[i] == "'"):
            if self.alternate and s[i] != "<":
                got, j = self._getstring(s, i)
                return '"' + got + '"', j
            return self._getstring(s, i)
        out = []
        brackets: list[str] = []
        while i < len(s) and (brackets or s[i] not in " \t") and s[i] != "," and (
                s[i] != "<" or not self.alternate):
            c = s[i]
            if c in "\"'":
                out.append(c)
                i += 1
                while i < len(s) and s[i] != c:
                    out.append(s[i])
                    i += 1
                if i == len(s):
                    return "".join(out), i
            elif c in "([":
                brackets.append(c)
            elif c == ")" and brackets and brackets[-1] == "(":
                brackets.pop()
            elif c == "]" and brackets and brackets[-1] == "[":
                brackets.pop()
            out.append(c)
            i += 1
        return "".join(out), i

    def _do_formals(self, mac: _Macro, s: str, i: int) -> int:
        i = _skip_white(s, i)
        while i < len(s):
            name, i = self._get_token(s, i)
            if not name:
                if mac.formals:
                    i -= 1
                break
            f = _Formal(name)
            i = _skip_white(s, i)
            if i < len(s) and s[i] == ":":     # (':' begins no name)
                qual, i = self._get_token(s, _skip_white(s, i + 1))
                if not qual:
                    self.error(f"Missing parameter qualifier for `{name}' in macro "
                               f"`{mac.name}'")
                elif qual == "req":
                    f.kind = "required"
                elif qual == "vararg":
                    f.kind = "vararg"
                else:
                    self.error(f"`{qual}' is not a valid parameter qualifier for `{name}' in "
                               f"macro `{mac.name}'")
                i = _skip_white(s, i)
            if i < len(s) and s[i] == "=":
                f.default, i = self.get_any_string(s, i + 1)
                i = _skip_white(s, i)
                if f.kind == "required":
                    f.default = ""
                    self.warn(f"Pointless default value for required parameter `{name}' in "
                              f"macro `{mac.name}'")
            if mac.formal(name) is not None:
                self.error(f"A parameter named `{name}' already exists for macro "
                           f"`{mac.name}'")
            f.index = len(mac.formals)
            mac.formals.append(f)
            if f.kind == "vararg":
                break
            c = i
            i = _skip_comma(s, i)
            if i != c and i >= len(s):
                i = c
                break
        return i

    def s_macro(self, args: str, label: str | None) -> None:
        """s_macro / define_macro."""
        s = scrub(args)
        got = self.buffer_and_nest("macro", "endm")
        if got is None:
            name = label or self._get_token(s, _skip_white(s, 0))[0]
            self.error(f"unexpected end of file in macro `{name}' definition")
        body, file, first = got
        mac = _Macro("", body=body, file=file, line=first)
        i = _skip_white(s, 0)
        if label:
            mac.name = label
            if i < len(s) and s[i] == "(":
                i = self._do_formals(mac, s, i + 1)
                if i < len(s) and s[i] == ")":
                    i = _skip_white(s, i + 1)
                else:
                    self.error(f"missing `)' after formals in macro definition `{label}'")
            else:
                i = self._do_formals(mac, s, i)
        else:
            mac.name, i = self._get_token(s, i)
            if not mac.name:
                self.error("Missing macro name")
            c = _skip_white(s, i)
            i = _skip_comma(s, c)
            i = self._do_formals(mac, s, i) if (i == c or i < len(s)) else c
        if i < len(s):
            self.error(f"Bad parameter list for macro `{mac.name}'")
        name = mac.name.lower()
        if name in self.macros:
            self.error(f"Macro `{mac.name}' was already defined")
        if name.startswith(".") and (name in DIRECTIVES or self.hooks.known(name)):
            self.warn(f"attempt to redefine pseudo-op `{mac.name}' ignored")
            return
        mac.name = name
        self.macros[name] = mac

    def s_purgem(self, args: str) -> None:
        for name in (n.strip() for n in args.split(",")):
            if not name:
                continue
            if self.macros.pop(name.lower(), None) is None:
                self.warn(f"Attempt to purge non-existing macro `{name.lower()}'")

    def s_mexit(self) -> None:
        if self.macro_nest:
            self._cond_exit_macro(self.macro_nest)
            while self.stack and self.stack[-1].kind != "macro":
                self.stack.pop()                # (an .irp / .rept inside the expansion too)
            if self.stack:
                self.stack.pop()
                self.macro_nest -= 1
        else:
            self.warn("ignoring macro exit outside a macro definition.")

    def _sub_actual(self, s: str, i: int, formals: dict[str, _Formal], kind: str,
                    out: list[str], copyifnotthere: bool) -> int:
        start = i
        t, i = self._get_token(s, i)
        if i < len(s) and s[i] == kind:         # (get_apost_token)
            i += 1
        f = formals.get(t)
        if f is not None:
            out.append(f.actual if f.actual else f.default)
        elif kind == "&":
            out.append("&" + t)
            if i != start and s[i - 1] == "&":
                out.append("&")
        elif copyifnotthere:
            out.append(t)
        else:
            out.append("\\" + t)
        return i

    def macro_expand_body(self, body: str, formals: dict[str, _Formal],
                          mac: _Macro | None) -> str:
        out: list[str] = []
        i = 0
        inquote = False
        locals_: list[str] = []
        n = len(body)
        while i < n:
            c = body[i]
            if c == "&":
                i = self._sub_actual(body, i + 1, formals, "&", out, False)
            elif c == "\\":
                i += 1
                if i < n and body[i] == "(":
                    i += 1
                    j = body.find(")", i)
                    if j < 0:
                        out.append(body[i:])
                        i = n
                        self.error("missing `)'")
                    out.append(body[i:j])
                    i = j + 1
                elif i < n and body[i] == "@":
                    out.append(str(self.macro_number))
                    i += 1
                elif i < n and body[i] == "&":
                    out.append("\\&")
                    i += 1
                else:
                    i = self._sub_actual(body, i, formals, "'", out, False)
            elif self.alternate and _BEGIN.match(c) and not inquote:
                if mac is None or not re.match(r"LOCAL[ \t]", body[i:i + 6], re.I):
                    i = self._sub_actual(body, i, formals, "'", out, True)
                else:
                    i = _skip_white(body, i + 5)
                    while i < n and body[i] != "\n":
                        name, i = self._get_token(body, i)
                        if not name:
                            break
                        if name in formals:
                            self.error(f"`{name}' was already used as parameter (or another "
                                       "local) name")
                        self.loccnt += 1
                        formals[name] = _Formal(name, actual=f".LL{self.loccnt:04x}")
                        locals_.append(name)
                        i = _skip_comma(body, i)
            elif c == '"':
                inquote = not inquote
                out.append(c)
                i += 1
            else:
                out.append(c)
                i += 1
        for name in locals_:
            formals.pop(name, None)
        text = "".join(out)
        if not text.endswith("\n"):
            text += "\n"
        return text

    def _expand(self, mac: _Macro, s: str) -> None:
        """macro_expand: the actuals, then the body."""
        for f in mac.formals:
            f.actual = ""
        pending = iter(mac.formals)
        f = next(pending, None)
        keyword = False
        i = _skip_white(s, 0)
        while i < len(s):
            scan = i
            while scan < len(s) and not self._issep(s[scan]) and (
                    self.alternate or s[scan] != "="):
                scan += 1
            if scan < len(s) and not self.alternate and s[scan] == "=":
                keyword = True
                t, i = self._get_token(s, i)
                if i >= len(s) or s[i] != "=":
                    self.error("confusion in formal parameters")
                ptr = mac.formal(t)
                if ptr is None:
                    self.error(f"Parameter named `{t}' does not exist for macro "
                               f"`{mac.name}'")
                if ptr.actual:
                    self.warn(f"Value for parameter `{ptr.name}' of macro `{mac.name}' was "
                              "already specified")
                ptr.actual, i = self.get_any_string(s, i + 1)
            else:
                if keyword:
                    self.error("can't mix positional and keyword arguments")
                if f is None:
                    self.error("too many positional arguments")
                if f.kind != "vararg":
                    f.actual, i = self.get_any_string(s, i)
                elif i < len(s):
                    f.actual, i = s[i:], len(s)
                f = next(pending, None)
            i = _skip_comma(s, i)
        for ptr in mac.formals:
            if ptr.kind == "required" and not ptr.actual:
                self.error(f"Missing value for required parameter `{ptr.name}' of macro "
                           f"`{mac.name}'")
        text = self.macro_expand_body(mac.body, {x.name: x for x in mac.formals}, mac)
        self.macro_number += 1
        self._expansion([text], mac.file, mac.line, "macro")

    # -- repetitions -------------------------------------------------------------------

    def s_rept(self, args: str) -> None:
        count = self.hooks.absolute(args)
        got = self.buffer_and_nest("rept", "endr")
        if got is None:
            self.error("REPT without ENDR")
        body, file, line = got
        if count < 0:
            self.error("negative count for REPT - ignored")
        self._expansion([body] * count, file, line, "repeat")

    def s_irp(self, irpc: bool, args: str) -> None:
        """s_irp / expand_irp."""
        s = scrub(args)
        got = self.buffer_and_nest(None, "endr")
        if got is None:
            self.error("unexpected end of file in irp or irpc")
        body, file, line = got
        i = _skip_white(s, 0)
        name, i = self._get_token(s, i)
        if not name:
            self.error("missing model parameter")
        f = _Formal(name, index=1)
        formals = {name: f}
        out = []
        i = _skip_comma(s, i)
        if i >= len(s):
            out.append(self.macro_expand_body(body, formals, None))
        else:
            in_quotes = False
            if irpc and s[i] == '"':
                in_quotes = True
                i += 1
            while i < len(s):
                if not irpc:
                    f.actual, i = self.get_any_string(s, i)
                else:
                    if s[i] == '"':
                        in_quotes = not in_quotes
                        nxt = _skip_white(s, i + 1)
                        if nxt >= len(s):
                            break
                    f.actual = s[i]
                    i += 1
                out.append(self.macro_expand_body(body, formals, None))
                if not irpc:
                    i = _skip_comma(s, i)
                elif not in_quotes:
                    i = _skip_white(s, i)
        self._expansion(out, file, line, "repeat")

    # -- files and messages ------------------------------------------------------------

    def find(self, name: str) -> str | None:
        """search_and_open: the program's folder, then the include folders."""
        if os.path.isabs(name):
            return name if os.path.isfile(name) else None
        for d in (self.base, *self.include_dirs):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        return None

    def s_include(self, args: str) -> None:
        name, rest = self._c_string(args, ".include")
        self._empty(rest)
        path = self.find(name)
        if path is None:
            self.error(f"can't open {name} for reading: No such file or directory")
        if len([b for b in self.stack if b.kind == "include"]) > 64:
            self.error(f"{name}: included too deeply")
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            self.error(f"can't open {name} for reading: {exc.strerror}")
        self.included.append(path)
        self.files[path] = text.splitlines()
        self.origins.setdefault(path, (self.current.line, self.current.raw))
        self._include(self._items(text, path, 1), "include")

    def s_errwarn(self, err: bool, args: str) -> None:
        msg = (".error directive invoked in source file" if err
               else ".warning directive invoked in source file")
        if args.strip():
            if not args.strip().startswith('"'):
                self.error(f"{'.error' if err else '.warning'} argument must be a string")
            msg, rest = self._c_string(args, "")
            self._empty(rest)
        if err:
            self.error(msg)
        self.warn(msg)
