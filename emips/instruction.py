"""MIPS instruction tables and lookup.

Port of the original Ruby ``EMIPS::Instruction`` class (``lib/eMIPSInstruction.rb``).

The per-release instruction tables are **data**: at migration time they were
dumped from the original Ruby class into ``data/instructions.json`` (see
``tools/dump_instructions.rb``), so Python does not reimplement the original
table builder.  :meth:`InstructionSet.find` mirrors the lookup semantics of
``Instruction#findInstruction`` one-for-one, and is verified against the
parity goldens captured from the running Ruby code
(``tests/fixtures/instruction_goldens.json``, ``tools/capture_goldens.rb``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: ``EMIPS::Instruction::OPCODE_DEFINITION_NOT_FOUND``
OPCODE_DEFINITION_NOT_FOUND = 200
#: ``EMIPS::Instruction::OPCODE_INSTRUCTION_NOT_FOUND``
OPCODE_INSTRUCTION_NOT_FOUND = 201

#: Releases present in the bundled instruction data, most basic first.
RELEASES = (
    "Release",
    "Release2",
    "Release3",
    "Release4",
    "Release5",
    "Release6",
    "Release7",
    "Release8",
)

#: Default release table (matches the original test harness's choice;
#: Release3/4/5 each carry one extra row, Release6..8 are PIC32 "P6"
#: subsets with ~124 rows).
DEFAULT_RELEASE = "Release2"

#: The generation label is stored but (like in the original) unused by the
#: lookup logic.
DEFAULT_GENERATION = "MIPS32"

_DATA_PATH = Path(__file__).resolve().parent / "data" / "instructions.json"


@dataclass(frozen=True)
class Instruction:
    """One instruction row of a release table.

    Field names mirror the keys of the Ruby ``@instructions`` hash:
    ``opcode``, ``opcodeName``, ``instrType``, ``function``,
    ``functionName``, ``functionInfo``, ``encode``, ``info``, ``desc``.
    """

    name: str
    opcode: str
    opcode_name: str
    instr_type: str
    function: str
    function_name: str
    function_info: dict[str, dict[str, Any]] | None
    encode: str
    info: str
    desc: str

    @property
    def fct(self) -> dict[str, Any] | None:
        """The ``fct`` operand used for matching, or ``None`` when the row
        has no function info (the original Ruby lookup raised
        ``NoMethodError`` on such rows; the port treats them as
        opcode-only matches)."""
        if self.function_info is None:
            return None
        return self.function_info.get("fct")


def _validate_binary32(code: str) -> str:
    if not isinstance(code, str) or len(code) != 32 or any(c not in "01" for c in code):
        raise ValueError(f"expected a 32-character binary string, got {code!r}")
    return code


class InstructionSet:
    """A MIPS instruction table for one release, with lookup by 32-bit
    binary instruction word."""

    def __init__(
        self,
        generation: str = DEFAULT_GENERATION,
        release: str = DEFAULT_RELEASE,
        data_path: str | Path | None = None,
    ) -> None:
        path = Path(data_path) if data_path is not None else _DATA_PATH
        with open(path, encoding="utf-8") as handle:
            blob = json.load(handle)
        if release not in blob:
            raise KeyError(f"unknown release {release!r}; available: {list(blob)}")
        table = blob[release]["instructions"]

        self.generation = generation
        self.release = release
        self._rows: list[Instruction] = []
        self._by_name: dict[str, Instruction] = {}

        for name, info in table.items():
            instr = Instruction(
                name=name,
                opcode=info["opcode"],
                opcode_name=info["opcodeName"],
                instr_type=info["instrType"],
                function=info["function"],
                function_name=info["functionName"],
                function_info=info["functionInfo"] or None,
                encode=info["encode"],
                info=info["info"],
                desc=info["desc"],
            )
            self._rows.append(instr)
            self._by_name[name] = instr  # duplicate names: last wins (as in Ruby)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def count(self) -> int:
        """Number of rows in the table (``@instructionsNumber``)."""
        return len(self._rows)

    def names(self) -> list[str]:
        """Instruction names in original table (insertion) order."""
        return [row.name for row in self._rows]

    def get(self, name: str) -> Instruction | None:
        return self._by_name.get(name)

    def find(self, code: str) -> list[Instruction]:
        """Look up a 32-bit binary instruction word.

        Port of ``Instruction#findInstruction``: rows are scanned in
        original table order and match when both

        * the top 6 bits equal the row's opcode, and
        * the row's own ``fct`` field (``position``/``length``/``value``)
          equals the slice of the word at that position,

        so a single word can match several rows (e.g. ``ABS.S``/``ABS.D``/
        ``ABS.PS`` share opcode + fct).

        Returns the matched rows in table order; an empty list means "not
        found" (the original returned :data:`OPCODE_INSTRUCTION_NOT_FOUND`,
        i.e. ``201``, in that case).
        """
        code = _validate_binary32(code)
        opcode = code[:6]
        matches: list[Instruction] = []
        for row in self._rows:
            if row.opcode != opcode:
                continue
            fct = row.fct
            if fct is None:
                # Documented deviation: the original raised a
                # NoMethodError here; such rows clearly only carry an
                # opcode, so match on that alone.
                matches.append(row)
                continue
            position = int(fct["position"])
            length = int(fct["length"])
            if fct["value"] == code[position : position + length]:
                matches.append(row)
        return matches

    def match(self, code: str) -> bool:
        """True when at least one row matches — the original
        ``findInstruction`` return contract (``true`` vs ``201``)."""
        return bool(self.find(code))
