"""MIPS core catalog — port of ``lib/MIPSinfo.rb`` + ``lib/MIPSlist.dat``.

The catalog file lists one line per core:
``<generation> <name> <introduced> <flag>`` (the last column is ``x`` when
the core is still supported).  The parser is a faithful port of the
original, including its "state carries across lines" behaviour (a generation
word on an already-known generation does not change the active one, and the
active core name persists until a new name word appears).
"""

from __future__ import annotations

from pathlib import Path

#: ``lib/MIPSlist.dat`` was CWD-relative in the original (a known bug — the
#: bot crashed at startup when not run from ``lib/``); the port uses an
#: absolute default anchored to the package.
_DATA_PATH = Path(__file__).resolve().parent / "data" / "MIPSlist.dat"


class MIPSList:
    """Parsed MIPS generation catalog with the original accessor API."""  # noqa: E501


    def __init__(self, path: str | Path | None = None) -> None:
        src = Path(path) if path is not None else _DATA_PATH
        text = src.read_text(encoding="utf-8", errors="replace")

        self._list: dict[str, dict[str, list[str]]] = {}
        generation: str | None = None
        name: str | None = None

        for line in text.splitlines():
            for index, word in enumerate(line.split()):
                if index == 0:
                    # Original: only adopt the word when the generation is
                    # new; otherwise keep the active one.
                    if word not in self._list:
                        self._list[word] = {}
                    generation = word
                elif index == 1:
                    if generation is None:
                        continue  # original would crash; skip malformed line
                    self._list[generation][word] = []
                    name = word
                else:
                    if generation is None or name is None:
                        continue  # original would crash; skip malformed line
                    self._list[generation][name].append(word)

    def is_generation(self, mipsgeneration: str) -> bool:
        return mipsgeneration in self._list

    def list_generations(self) -> list[str]:
        """Generation names in file order (``listGenerations``)."""
        return list(self._list)

    def list_mips_names_in_generation(self, mipsgeneration: str) -> list[str]:
        """Core names of one generation in file order
        (``listMIPSnamesInGeneration``)."""
        if mipsgeneration not in self._list:
            raise KeyError(mipsgeneration)
        return list(self._list[mipsgeneration])

    def list_mips_values(self, mipsgeneration: str, mipsname: str) -> list[str]:
        """Per-core values (year, flag, ...) in file order
        (``listMIPSValues``)."""
        if mipsgeneration not in self._list or mipsname not in self._list[mipsgeneration]:
            raise KeyError(f"{mipsgeneration} {mipsname}")
        return list(self._list[mipsgeneration][mipsname])
