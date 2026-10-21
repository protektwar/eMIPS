"""eMIPS — MIPS instruction decoding, memory, HEX loading and CPU building
blocks.

Python port (2026) of the original Ruby eMIPS bot library (2021–2023, now
removed; only this package remains).  Instruction tables and CP0 register
descriptions are data files under ``emips/data/`` produced from the original
Ruby source at migration time (``tools/``).
"""

from .instruction import (
    DEFAULT_GENERATION,
    DEFAULT_RELEASE,
    Instruction,
    InstructionSet,
    OPCODE_DEFINITION_NOT_FOUND,
    OPCODE_INSTRUCTION_NOT_FOUND,
    RELEASES,
)
from .mipsinfo import MIPSList
from .isa import Decode, ISA, ISADecodeError
from .memory import PMemory, VMemory
from .hexload import (
    DecodeFile,
    FileContentError,
    FileCrcError,
    FileRecordTypeError,
    FILE_CONTENT_ERROR,
    FILE_CRC_ERROR,
    FILE_RECORDTYPE_ERROR,
    HexLoadError,
    RECORD_TYPES,
)
from .assembler import AsmError, Program, assemble
from .disasm import disassemble
from .sim import Machine
from .cpu import (
    CoProcessor0,
    ControlUnit,
    CPU,
    MMU,
    MUX,
    Pipeline,
    Registers,
)

__all__ = [
    "AsmError",
    "Machine",
    "Program",
    "assemble",
    "disassemble",
    "Instruction",
    "InstructionSet",
    "OPCODE_DEFINITION_NOT_FOUND",
    "OPCODE_INSTRUCTION_NOT_FOUND",
    "DEFAULT_GENERATION",
    "DEFAULT_RELEASE",
    "RELEASES",
    "MIPSList",
    "Decode",
    "ISA",
    "ISADecodeError",
    "PMemory",
    "VMemory",
    "DecodeFile",
    "FileContentError",
    "FileCrcError",
    "FileRecordTypeError",
    "FILE_CONTENT_ERROR",
    "FILE_CRC_ERROR",
    "FILE_RECORDTYPE_ERROR",
    "HexLoadError",
    "RECORD_TYPES",
    "ControlUnit",
    "CPU",
    "CoProcessor0",
    "MMU",
    "MUX",
    "Pipeline",
    "Registers",
]

__version__ = "2.5.0"
