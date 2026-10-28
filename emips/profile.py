"""A program's own settings: the ``eMIPS:`` header line (Qt-free).

A program — an example, a lab, any ``.asm`` or ``.c`` — may say how it
is meant to run, in a comment line anywhere in it (the guide's labs have
it last, so their line numbers stay), with the command line's own flags::

    # eMIPS: --board malta
    # eMIPS: --cores 2 --caches --pipeline --model 24k
    // eMIPS: --release R6                       (C; /* eMIPS: ... */ too)

:func:`read` turns it into a :class:`Profile`: the release, byte order,
mode and the processor / memory settings it names, every other one of
those at its default.  The GUI applies it when the program is opened
(File > Open, Examples, Labs), the command line uses it for flags not
given there.  Preferences that are not about the machine (assembler
strictness, run limits, latencies, theme) are never touched.
"""

from __future__ import annotations

import argparse
import re
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .compat import RELEASE_ORDER as RELEASES
from .options import BOOT_MODES, Options

#: the header (the first lines of a file)
HEADER = re.compile(r"^\s*(?:#|//|/\*+|\*)\s*eMIPS:\s*(.*?)\s*(?:\(.*\))?\s*(?:\*/)?\s*$")
#: the options a profile owns (the machine): unnamed ones go to their defaults
PROFILE_FIELDS = ("board", "disks", "iocu", "model", "icache", "dcache", "cache_parity", "l2", "l3",
                  "cache_ecc", "write_buffer", "nonblocking", "cache_init", "mmu", "cores",
                  "shadow_sets", "boot", "compressed", "fpu", "dsp", "msa", "msa_wrp", "mt",
                  "mt_pm", "itc", "vps", "itu", "core", "arch", "nms", "crc", "ginv",
                  "rv_extensions", "rv_modes", "rv_relax", "rv_rnmi", "rv_intc", "l2_waits",
                  "dspram")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eMIPS:", add_help=False, exit_on_error=False)
    p.add_argument("--release", choices=RELEASES)
    p.add_argument("--arch", choices=("mips32", "mips64", "nanomips", "rv32", "rv64"))
    p.add_argument("--rv-ext", dest="rv_extensions")
    p.add_argument("--rv-modes", dest="rv_modes", choices=("msu", "mu", "m"))
    p.add_argument("--rv-rnmi", dest="rv_rnmi")
    p.add_argument("--rv-intc", dest="rv_intc", choices=("none", "plic", "aplic", "imsic"))
    p.add_argument("--l2-waits", dest="l2_waits")
    p.add_argument("--dspram", type=int)
    p.add_argument("--no-relax", dest="no_relax", action="store_true")
    p.add_argument("--nms", action="store_true")
    p.add_argument("--crc", choices=("auto", "none", "crc"))
    p.add_argument("--ginv", choices=("auto", "none", "ginvi", "ginvt"))
    p.add_argument("--little", action="store_true")
    p.add_argument("--board")
    p.add_argument("--disk", action="append")
    p.add_argument("--model", choices=("5-stage", "24k", "p8700"))
    p.add_argument("--caches", action="store_true")
    p.add_argument("--icache")
    p.add_argument("--dcache")
    p.add_argument("--cache-parity", action="store_true")
    p.add_argument("--cache-init", choices=("random", "hardware"))
    p.add_argument("--l2")
    p.add_argument("--l3")
    p.add_argument("--cache-ecc", action="store_true")
    p.add_argument("--write-buffer", action="store_true")
    p.add_argument("--nonblocking", action="store_true")
    p.add_argument("--mmu")
    p.add_argument("--cores", type=int)
    p.add_argument("--vps", type=int)
    p.add_argument("--itu")
    p.add_argument("--iocu", action="store_true")
    p.add_argument("--shadow-sets", type=int)
    p.add_argument("--compressed")
    p.add_argument("--core")
    p.add_argument("--no-fpu", action="store_true")
    p.add_argument("--dsp", choices=("auto", "none", "dsp", "dspr2"))
    p.add_argument("--msa", choices=("auto", "none", "msa"))
    p.add_argument("--msa-wrp", type=int)
    p.add_argument("--mt")
    p.add_argument("--mt-pm", choices=("rr", "wrr", "wrr2"))
    p.add_argument("--itc")
    p.add_argument("--boot", choices=BOOT_MODES)
    p.add_argument("--pipeline", action="store_true")
    p.add_argument("--no-forwarding", action="store_true")
    return p


@dataclass
class Profile:
    """What a program's header asks for."""
    args: list[str]
    release: str = "R5"
    endian: str = "big"
    mode: str = "instruction"
    forwarding: bool = True
    #: the profile's options (only the PROFILE_FIELDS it sets)
    options: dict[str, Any] = field(default_factory=dict)
    #: the flags as given (argparse names: what the command line merges)
    given: dict[str, Any] = field(default_factory=dict)

    def options_for(self, base: Options) -> Options:
        """``base`` with the machine settings this profile makes: its own,
        every other profile field at its default."""
        d = Options()
        changes = {f: self.options.get(f, getattr(d, f)) for f in PROFILE_FIELDS}
        return replace(base, **changes).validated()

    def describe(self) -> str:
        return " ".join(self.args) if self.args else "the defaults"

    def differs(self, options: Options, release: str, endian: str, mode: str,
                forwarding: bool) -> bool:
        o = self.options_for(options)
        return ((self.release, self.endian, self.mode, self.forwarding)
                != (release, endian, mode, forwarding)
                or any(getattr(o, f) != getattr(options, f) for f in PROFILE_FIELDS))


def header_args(text: str) -> list[str] | None:
    """The flags of the ``eMIPS:`` header, or None."""
    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            return shlex.split(m.group(1))
    return None


#: the user guide's example markers (tests/test_guide.py) -> the header's flags
_GUIDE_MARKERS = {"release": "--release", "arch": "--arch", "core": "--core",
                  "cores": "--cores", "vps": "--vps", "boot": "--boot", "l2": "--l2",
                  "rv-ext": "--rv-ext", "rv-intc": "--rv-intc"}


def guide_profile(text: str) -> Profile | None:
    """The settings a user-guide example asks for with its first comment
    lines (``# arch: rv32``, ``# rv-ext: gcv``, ``# core: p8700``, ``# caches: 32K`` ...): the
    window's *Open in the editor*.  None: none of them."""
    flags: list[str] = []
    for line in text.splitlines():
        m = re.match(r"#\s*([a-z0-9-]+):\s?(.*)$", line)
        if not m:
            break
        key, value = m.group(1), m.group(2).strip()
        if key in _GUIDE_MARKERS:
            flags += [_GUIDE_MARKERS[key], value]
        elif key == "caches":
            flags += ["--icache", value, "--dcache", value]
        elif key == "write-buffer":
            flags.append("--write-buffer")
    if not flags:
        return None
    return read("# eMIPS: " + shlex.join(flags))


def _resolve_disk(spec: str, base: Path | None) -> str:
    """A ``--disk`` value with its image file relative to the program's
    folder (``base``)."""
    slot, sep, source = spec.partition("=")
    if not sep or slot.strip().lower() not in ("pri0", "pri1", "sec0", "sec1"):
        slot, sep, source = "", "", spec
    source = source.strip()
    if base is not None and not source.lower().startswith("blank:"):
        p = Path(source).expanduser()
        if not p.is_absolute():
            source = str((base / p).resolve())
    return f"{slot.strip()}{sep}{source}"


def read(text: str, base: str | Path | None = None) -> Profile | None:
    """The program's profile (None: no header); ``base``: the program's
    folder (disk image paths are relative to it).  A malformed header is
    a :class:`ValueError` naming what is wrong."""
    args = header_args(text)
    if args is None:
        return None
    try:
        ns, rest = _parser().parse_known_args(args)
    except (argparse.ArgumentError, SystemExit) as exc:
        raise ValueError(f"eMIPS header: {exc}") from None
    if rest:
        raise ValueError(f"eMIPS header: unknown {' '.join(rest)}")
    o: dict[str, Any] = {}
    for name in ("board", "model", "icache", "dcache", "l2", "l3", "mmu", "cores",
                 "shadow_sets", "boot", "cache_init", "compressed", "dsp", "msa", "msa_wrp",
                 "mt", "mt_pm", "itc", "vps", "itu", "core", "arch", "crc", "ginv",
                 "rv_extensions", "rv_modes", "rv_rnmi", "rv_intc", "l2_waits", "dspram"):
        v = getattr(ns, name)
        if v is not None:
            o[name] = v.lower() if isinstance(v, str) and name not in ("boot", "rv_rnmi") \
                else v
    if ns.disk:
        o["disks"] = ",".join(_resolve_disk(d, Path(base) if base is not None else None)
                              for d in ns.disk)
        ns.disk = o["disks"].split(",")
    if ns.caches:
        o.setdefault("icache", "32K")
        o.setdefault("dcache", "32K")
    for name in ("cache_parity", "cache_ecc", "write_buffer", "nonblocking", "iocu", "nms"):
        if getattr(ns, name):
            o[name] = True
    if ns.no_fpu:
        o["fpu"] = False
    if ns.no_relax:
        o["rv_relax"] = False
    check = replace(Options(), **o).validated()      # a bad value: say so
    for name, v in o.items():
        if getattr(check, name) != v:
            raise ValueError(f"eMIPS header: bad {name.replace('_', '-')} {v!r}")
    release = ns.release
    if o.get("core"):                   # its build under the flags given (emips.cores)
        from .cli import explicit_fields
        from .cores import apply
        built, core_release = apply(Options(), o["core"], keep=explicit_fields(ns))
        o.update({f: getattr(built, f) for f in PROFILE_FIELDS
                  if f not in o and getattr(built, f) != getattr(Options(), f)}
                 | {"core": built.core})
        if built.write_buffer:
            o.setdefault("write_buffer", True)
        release = release or core_release
    if o.get("arch") == "nanomips":     # (Release 6 based; another release: refused later)
        release = release or "R6"
    return Profile(args=args, release=release or "R5",
                   endian="little" if ns.little else "big",
                   mode="pipeline" if ns.pipeline else "instruction",
                   forwarding=not ns.no_forwarding, options=o,
                   given={k: v for k, v in vars(ns).items() if v not in (None, False)})
