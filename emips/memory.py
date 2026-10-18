"""Physical/virtual memory — port of ``lib/eMIPSmemory.rb`` (``PMemory``,
``VMemory``).

The original stores flat ``address -> 2-char-hex-byte`` hashes.  Keys are
prefix-less lowercase hex strings (as produced by the original's
``address.to_s(16)``); values are 2-character hex strings such as ``"3c"``.
"""

from __future__ import annotations


def _key(address: int | str) -> str:
    if isinstance(address, int):
        return format(address, "x")
    return address


class PMemory:
    """Flat physical memory (``EMIPS::PMemory``)."""

    def __init__(self) -> None:
        self._mem: dict[str, str] = {}

    def write_to_memory(self, address: int | str, value: str) -> None:
        self._mem[_key(address)] = value

    def read_from_memory(self, address: int | str) -> str | None:
        """Byte at *address* (2-char hex string) or ``None`` if never written
        (original returned ``nil``)."""
        return self._mem.get(_key(address))

    def get_memory(self) -> dict[str, str]:
        """The full address→byte mapping (``getMem``)."""
        return self._mem

    def __len__(self) -> int:
        return len(self._mem)

    def __contains__(self, address: int | str) -> bool:
        return _key(address) in self._mem

    def pp(self) -> str:
        """Human-readable dump, same layout as the original ``pp``:
        ``0xADDR: v`` per cell, a `` |`` separator every 4 cells and a line
        break every 16 cells."""
        parts: list[str] = []
        counter = 0
        values = ""
        for index, (mem_addr, mem_value) in enumerate(self._mem.items()):
            if not values:
                values = mem_value
            else:
                values = values + " " + mem_value
            parts.append("0x" + f"{mem_addr}: ")
            if (counter + 1) % 4 == 0:
                values = values + " |"
            counter += 1
            if counter > 15 or (index + 1 == len(self._mem)):
                parts.append(values)
                values = ""
                counter = 0
                parts.append("\n")
        parts.append(values)
        return "".join(parts)


class VMemory:
    """Virtual memory placeholder (``EMIPS::VMemory``) — the original had an
    empty constructor only; kept as a stub so the port mirrors the layout."""
