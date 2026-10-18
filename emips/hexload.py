"""Intel HEX file loader into ``PMemory`` — port of ``HexLoad::DecodeFile``
(``lib/eMIPShexload.rb``).

Record types (from the original): ``00`` data, ``01`` end-of-file, ``02``
extended segment, ``03`` start segment, ``04`` extended linear, ``05`` start
linear.

**Two bugs of the Ruby original are fixed here (documented):**

* **CRC byte loop off-by-one.** The original looped ``0..recLen`` while
  indexing a ``recLen*2``-character data slice, so the final index read out
  of range and raised ``TypeError`` on the first data record of every file.
* **Broken checksum test.** ``((calc + stored).to_s(16)[0]).to_i != 0``
  always flags a *valid* record (for a good record ``calc + stored == 0x100``
  whose leading hex digit is ``"1"`` → ``1 != 0`` → "error").  The standard
  two's-complement check is ``(calc + stored) & 0xFF == 0``.

Also, the Ruby ``loadFile`` returned error codes (100/101/102) that its only
caller silently discarded — errors were invisible.  The port surfaces them as
typed exceptions (same numeric codes) instead.

Additions (not in the Ruby): :meth:`DecodeFile.load_text` decodes HEX text
that is already in memory (the GUI), and :attr:`DecodeFile.has_start` tells
whether a type-05 record was present (``start_address`` stays 0 otherwise).
"""

from __future__ import annotations

from pathlib import Path

from .memory import PMemory

# --- original class constants (HexLoad::DecodeFile) -------------------------
FILE_CONTENT_ERROR = 100
FILE_RECORDTYPE_ERROR = 101
FILE_CRC_ERROR = 102

# --- record types (HexLoad::DecodeFile.recordType) ---------------------------
RECORD_TYPES = {
    "00": "DataRecord",
    "01": "EndofFileRecord",
    "02": "ExtendedSegmentAddressRecord",
    "03": "StartSegmentAddressRecord",
    "04": "ExtendedLinearAddressRecord",
    "05": "StartLinearAddressRecord",
}


class HexLoadError(Exception):
    """Base class; carries the original numeric error code."""

    code = FILE_CONTENT_ERROR


class FileContentError(HexLoadError):
    """A line of the HEX file is not a well-formed record (code 100)."""

    code = FILE_CONTENT_ERROR


class FileRecordTypeError(HexLoadError):
    """Unknown record type byte (code 101)."""

    code = FILE_RECORDTYPE_ERROR


class FileCrcError(HexLoadError):
    """Record checksum does not validate (code 102)."""

    code = FILE_CRC_ERROR


class DecodeFile:
    """Decode an Intel HEX file into a :class:`PMemory` instance."""

    def __init__(self, memory: PMemory) -> None:
        self._memory = memory
        #: Start address of the application (record type ``05``); the Ruby
        #: original computed this into a local that was then discarded.
        self.start_address: int = 0
        #: a start-linear-address record (05) was seen
        self.has_start = False

    def load(self, path: str | Path) -> None:
        self.load_text(Path(path).read_text(encoding="utf-8", errors="replace"))

    def load_text(self, text: str) -> None:
        """Decode Intel HEX ``text`` (the body of :meth:`load`)."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        extended_address = 0
        for lineno, raw in enumerate(text.split("\n"), start=1):
            if not raw:
                continue
            if not raw.startswith(":"):
                raise FileContentError(
                    f"line {lineno}: record does not start with ':' (got {raw[:20]!r})"
                )
            line = raw[1:]

            rec_len = int(line[0:2], 16)
            load_offset = int(line[2:6], 16)
            rec_type = line[6:8]
            data = line[8 : 8 + rec_len * 2]
            rec_crc_raw = line[8 + rec_len * 2 : 10 + rec_len * 2]

            if len(rec_crc_raw) != 2 or not set(rec_crc_raw) <= set("0123456789abcdefABCDEF"):
                raise FileContentError(
                    f"line {lineno}: malformed record (recLen {rec_len:#04x}, "
                    f"length {len(line)} chars)"
                )

            # Two's-complement checksum over recLen + loadOffset + recType + data.
            calculated = (
                int(line[0:2], 16)
                + int(line[2:4], 16)
                + int(line[4:6], 16)
                + int(line[6:8], 16)
                + sum(int(data[i : i + 2], 16) for i in range(0, len(data), 2))
            )
            stored_crc = int(rec_crc_raw, 16)
            if (calculated + stored_crc) & 0xFF:
                raise FileCrcError(
                    f"line {lineno}: checksum mismatch "
                    f"(calculated {calculated:#04x} + stored {stored_crc:#04x})"
                )

            if rec_type == "00":  # DataRecord
                address = load_offset + extended_address
                for i in range(rec_len):
                    byte = data[i * 2 : i * 2 + 2].lower()
                    self._memory.write_to_memory(address, byte)
                    address += 1
            elif rec_type == "01":  # EndofFileRecord
                continue
            elif rec_type in ("02", "03"):  # Extended/Start Segment — kept, not mapped
                continue
            elif rec_type == "04":  # ExtendedLinearAddressRecord
                extended_address = int(data, 16) << 16
            elif rec_type == "05":  # StartLinearAddressRecord
                self.start_address = int(data, 16)
                self.has_start = True
            else:
                raise FileRecordTypeError(
                    f"line {lineno}: unknown record type {rec_type!r}"
                )
