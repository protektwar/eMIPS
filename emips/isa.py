"""Executable-ISA decoder (M0) built on ``emips/data/isaref.json``.

This is the emulator's decode, as distinct from the legacy table lookup
:meth:`emips.instruction.InstructionSet.find`.  The table is the Ruby dump
plus the cited M2 corrections (``emips/data/corrections.json``), checked
against GNU binutils (every word decoded both ways).  The model
(see ``audit`` in ``isaref.json``):

* the 32-char ``encode`` template is the **sole** decode key, frozen to a
  ``(mask, value)`` pair — ``functionInfo`` only names operand slots;
* among *lead* entries (``decode.lead == True``; since the M2 corrections
  every row is a lead — the identical-key tie-break machinery stays for
  safety) the most specific mask (highest fixed-bit count) wins;
* a word that matches no lead raises :class:`ISADecodeError`, which
  carries :data:`emips.instruction.OPCODE_INSTRUCTION_NOT_FOUND` (201).

Zero runtime dependencies (stdlib only), Qt-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .instruction import OPCODE_INSTRUCTION_NOT_FOUND

_ISAREF_PATH = Path(__file__).resolve().parent / "data" / "isaref.json"


class ISADecodeError(Exception):
    """The word decodes to no lead instruction (the ``201`` condition).

    Carries the offending word and the original error code
    (:data:`emips.instruction.OPCODE_INSTRUCTION_NOT_FOUND`).
    """

    code = OPCODE_INSTRUCTION_NOT_FOUND

    def __init__(self, word: int) -> None:
        self.word = word
        super().__init__(
            f"no instruction decodes 0x{word:08x} "
            f"(code {OPCODE_INSTRUCTION_NOT_FOUND})"
        )


def _as_word(word: int | str) -> int:
    """Normalize *word* to a 32-bit int (accepts int or 32-char binary)."""
    if isinstance(word, bool):
        raise TypeError(f"word must be an int or 32-char binary string, got {word!r}")
    if isinstance(word, str):
        s = word.strip()
        if len(s) != 32 or any(c not in "01" for c in s):
            raise ValueError(f"expected a 32-char binary string, got {word!r}")
        return int(s, 2)
    if isinstance(word, int) and 0 <= word <= 0xFFFFFFFF:
        return word
    raise TypeError(f"word must be an int or 32-char binary string, got {word!r}")


def _extract(word: int, msb: int, length: int) -> int:
    """Extract the field starting at *msb* (0-indexed from the MSB char)
    of *length* bits from a 32-bit word."""
    if not 0 <= msb or not 1 <= length or msb + length > 32:
        raise ValueError(f"bad field span msb={msb} length={length}")
    return (word >> (31 - (msb + length - 1))) & ((1 << length) - 1)


def _lookup(mapping: list[dict[str, Any]], name: object) -> dict[str, Any]:
    for entry in mapping:
        if entry.get("name") == name:
            return entry
    raise KeyError(f"no such operand field/slot: {name!r}")


@dataclass(frozen=True)
class Decode:
    """Result of decoding one 32-bit word.

    ``entry`` is the full ``isaref.json`` instruction record (name,
    opcode, encode template, decode key, fields, slots, family, status,
    pic32 flag) so the M1 executors can read anything without a second
    lookup.
    """

    word: int
    entry: Mapping[str, Any]

    @property
    def name(self) -> str:
        return self.entry["name"]

    @property
    def family(self) -> str:
        return self.entry["family"]

    @property
    def status(self) -> str:
        return self.entry["status"]

    @property
    def pic32(self) -> bool:
        return self.entry["pic32"]

    @property
    def key(self) -> tuple[int, int]:
        d = self.entry["decode"]
        return d["mask"], d["value"]

    def field(self, name: str) -> int:
        """Value of the named operand field (functionInfo slot name)."""
        f = _lookup(self.entry["fields"], name)
        return _extract(self.word, f["msb"], f["length"])

    def field_value(self, name: str) -> int:
        """Value of the named field; fixed fields return their template
        value, so this works for opcode/fct as well."""
        return self.field(name)

    def fields(self, variable_only: bool = False) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.entry["fields"]:
            if variable_only and f["fixed"]:
                continue
            out[f["name"]] = _extract(self.word, f["msb"], f["length"])
        return out

    def slot(self, name: str | int) -> int:
        """Value of an operand slot, by name or positional index."""
        if isinstance(name, int):
            runs = self.entry["slots"]
            if not 0 <= name < len(runs):
                raise IndexError(f"slot index {name} out of range")
            entry = runs[name]
        else:
            entry = _lookup(self.entry["slots"], name)
        return _extract(self.word, entry["msb"], entry["length"])

    def slots(self) -> list[dict[str, Any]]:
        return [
            {**run, "value": _extract(self.word, run["msb"], run["length"])}
            for run in self.entry["slots"]
        ]


_RELATIONS = {"==": lambda x, y: x == y, "!=": lambda x, y: x != y,
              "<": lambda x, y: x < y, ">=": lambda x, y: x >= y}


def _constraint_ok(entry: Mapping[str, Any], w: int) -> bool:
    """Release 6 compact branches share encodings and are told apart by
    relations between the rs and rt fields (``[["rs", "<", "rt"], ...]``)."""
    rules = entry.get("constraint")
    if not rules:
        return True
    f = {"rs": (w >> 21) & 31, "rt": (w >> 16) & 31}
    return all(_RELATIONS[op](f[a], f[b] if isinstance(b, str) else b)
               for a, op, b in rules)


class ISA:
    """The eMIPS executable-ISA definition (313 rows: the Release3 union of
    the Ruby dump plus the M2 corrections).

    Construct once and reuse; decoding is a linear scan of the lead
    entries, most-specific mask wins.

    ``release`` (``"R1"``, ``"R2"``, ``"R3"``, ``"R5"``, ``"R6"``) selects
    the instruction set of one MIPS32 release (PLAN M3): only rows that
    exist in it decode (a word whose only matches belong to other
    releases raises :class:`ISADecodeError`, i.e. Reserved Instruction),
    and the most specific row *of that release* wins — e.g. ``EHB`` is
    ``SLL $0,$0,3`` in Release 1.  The default is ``"R5"``, the full
    pre-Release-6 instruction set.

    ``dsp`` (PLAN M18) adds the DSP Module's rows (``ase`` "dsp" /
    "dspr2", MD00374): 0 = none (their words are Reserved), 1 = the DSP
    Module (Config3.DSPP), 2 = Revision 2 (DSP2P; on Release 6 this is
    Revision 3, with BPOSGE32C).  Without it a DSP word does not decode —
    the Reserved Instruction exception of Table 3.7.

    ``mt`` (PLAN M19) adds the MT Module's rows (``ase`` "mt", MD00378:
    DMT, EMT, DVPE, EVPE, FORK, YIELD, MFTR, MTTR; Releases 2-5).  Without
    it they are Reserved Instructions (MD00378 5.1).

    ``arch`` (PLAN M22) is "mips32" (the default) or "mips64": a MIPS64
    ISA has every row plus the doubleword ones (``arch`` "mips64", MIPS64
    Vol. II), each in the releases of its ``releases64`` (MIPS64 Release
    1 already has what MIPS32 only gained in Release 2: COP1X, paired
    single); a MIPS32 ISA never sees the MIPS64 rows — their words are
    Reserved Instructions there.
    """

    DEFAULT_RELEASE = "R5"
    #: the DSP Module level an ``ase`` value needs
    DSP_LEVEL = {"dsp": 1, "dspr2": 2}

    ARCHES = ("mips32", "mips64")

    def __init__(self, path: str | Path | None = None, *,
                 release: str | None = None, dsp: int = 0, mt: bool = False,
                 arch: str = "mips32") -> None:
        src = Path(path) if path is not None else _ISAREF_PATH
        with open(src, encoding="utf-8") as handle:
            self._data = json.load(handle)
        if arch not in self.ARCHES:
            raise ValueError(f"arch must be one of {self.ARCHES}, not {arch!r}")
        #: "mips32" / "mips64": which rows exist, and which release list
        #: says when (``releases`` / ``releases64``)
        self.arch = arch
        self._key = "releases64" if arch == "mips64" else "releases"
        self._instructions: list[dict[str, Any]] = [
            e for e in self._data["instructions"]
            if arch == "mips64" or e.get("arch") != "mips64"]
        self._by_name = {e["name"]: e for e in self._instructions}
        self.release = release or self.DEFAULT_RELEASE
        order = self._data["meta"].get("release_order", [])
        if order and self.release not in order:
            raise ValueError(f"unknown release {self.release!r}; one of {order}")
        if dsp not in (0, 1, 2):
            raise ValueError(f"dsp must be 0, 1 or 2, not {dsp!r}")
        #: the DSP Module level whose rows decode (0: none)
        self.dsp = dsp
        #: the MT Module's rows decode
        self.mt = bool(mt)
        self._leads = [e for e in self._instructions if e["decode"]["lead"]
                       and self._has(e)]
        self._leads.sort(key=lambda e: -e["decode"]["specificity"])
        #: word -> Decode (decoding is a pure function of the word)
        self._decoded: dict[int, Decode] = {}

    _cache: dict[tuple[str, int, bool, str], "ISA"] = {}
    _DECODED_MAX = 1 << 16

    @classmethod
    def for_release(cls, release: str | None = None, dsp: int = 0,
                    mt: bool = False, arch: str = "mips32") -> "ISA":
        """A shared (cached) ISA of ``release`` (DSP Module level ``dsp``,
        the MT Module ``mt``, ``arch`` MIPS32 / MIPS64) — the tables are
        read once."""
        key = (release or cls.DEFAULT_RELEASE, dsp, bool(mt), arch)
        if key not in cls._cache:
            cls._cache[key] = cls(release=key[0], dsp=dsp, mt=key[2], arch=arch)
        return cls._cache[key]

    def _has(self, e: dict[str, Any]) -> bool:
        """Row ``e`` exists in this release with these modules."""
        if self.release not in e.get(self._key, e.get("releases", [self.release])):
            return False
        ase = e.get("ase")
        if ase == "mt":
            return self.mt
        return self.DSP_LEVEL.get(ase, 0) <= self.dsp

    def in_release(self, name: str) -> bool:
        """Does row ``name`` exist in this ISA's release?"""
        e = self._by_name[name]
        return self.release in e.get(self._key, e.get("releases", [self.release]))

    def implemented(self, name: str) -> bool:
        """Does row ``name`` decode on this ISA (its release, and the
        module its ``ase`` needs)?"""
        return self._has(self._by_name[name])

    # -- decode ------------------------------------------------------------
    def decode(self, word: int | str) -> Decode:
        """Decode a 32-bit word into a :class:`Decode`.

        Raises :class:`ISADecodeError` (code 201) when no lead entry
        matches the fixed-bit pattern.
        """
        if type(word) is int:              # (the usual call: a word decoded before)
            hit = self._decoded.get(word)
            if hit is not None:
                return hit
        w = _as_word(word)
        hit = self._decoded.get(w)
        if hit is not None:
            return hit
        for entry in self._leads:          # most specific first
            d = entry["decode"]
            if (w & d["mask"]) == d["value"] and _constraint_ok(entry, w):
                if len(self._decoded) >= self._DECODED_MAX:
                    self._decoded.clear()
                hit = self._decoded[w] = Decode(word=w, entry=entry)
                return hit
        raise ISADecodeError(w)

    def decode_name(self, word: int | str) -> str:
        return self.decode(word).name

    def can_decode(self, word: int | str) -> bool:
        w = _as_word(word)
        return any(
            (w & e["decode"]["mask"]) == e["decode"]["value"]
            and _constraint_ok(e, w)
            for e in self._leads
        )

    # -- table access -------------------------------------------------------
    def get(self, name: str) -> Mapping[str, Any]:
        """The full instruction record for *name* (KeyError if absent)."""
        return self._by_name[name]

    def names(self) -> list[str]:
        """Instruction names in table order (every row of the architecture,
        all releases)."""
        return [e["name"] for e in self._instructions]

    def status(self, name: str) -> str:
        return self._by_name[name]["status"]

    def family(self, name: str) -> str:
        return self._by_name[name]["family"]

    def by_status(self, status: str) -> list[str]:
        if status not in {"core", "overlay", "unimpl"}:
            raise ValueError(f"unknown status {status!r}")
        return [e["name"] for e in self._instructions if e["status"] == status]

    # -- metadata ------------------------------------------------------------
    @property
    def instructions(self) -> list[dict[str, Any]]:
        return self._instructions

    @property
    def leads(self) -> list[dict[str, Any]]:
        return self._leads

    @property
    def lead_count(self) -> int:
        return len(self._leads)

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._data.get("meta", {})

    @property
    def audit(self) -> Mapping[str, Any]:
        return self._data.get("audit", {})
