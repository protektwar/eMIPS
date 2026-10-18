"""CPU building blocks — port of ``lib/eMIPScpu.rb``.

``ControlUnit`` implements the original five ``andFunctionN`` op-code
decoders.  In the source they are written as bit-inversion + AND chains,
but running the original Ruby class over all 64 six-bit opcodes
(``tools/dump_decoders.rb`` → ``tests/fixtures/decoder_goldens.json``)
shows each decoder asserts for exactly one half of the opcode space, keyed
only on the top bit (bit 5):

==================  =======================================
bit 5               decoders that assert
==================  =======================================
``0`` (opcodes 0–31)   ``andFunction1``, ``andFunction4``, ``andFunction5``
``1`` (opcodes 32–63)  ``andFunction2``, ``andFunction3``
==================  =======================================

``activate`` wires those five decoder outputs to the control signals **exactly**
as the original — including the odd result that ``jump``/``branch`` fire for
every ``bit5==0`` opcode and the memory signals for every ``bit5==1`` opcode
— so signal-level behavior is preserved even where it looks wrong.  (The
original ``getALUOp`` referenced an ivar that was never assigned — a crash;
the port exposes the two real bits ``alu_op1``/``alu_op0`` instead.)
"""

from __future__ import annotations

import json
from pathlib import Path

from .instruction import InstructionSet, OPCODE_INSTRUCTION_NOT_FOUND
from .memory import PMemory

_CP0_PATH = Path(__file__).resolve().parent / "data" / "cp0.json"


class ControlUnit:
    """Six-bit op-code decoders and the control-signal wiring (``EMIPS::ControlUnit``)."""

    # Verified by running the original EMIPS::ControlUnit over all 64
    # opcodes (see module docstring): each decoder fires for exactly one of
    # the two halves of the opcode space, selected by bit 5 alone.
    _LOW_OPS = frozenset(format(i, "06b") for i in range(32))      # bit5 == 0
    _HIGH_OPS = frozenset(format(i, "06b") for i in range(32, 64))  # bit5 == 1
    _DECODERS = {1: _LOW_OPS, 2: _HIGH_OPS, 3: _HIGH_OPS, 4: _LOW_OPS, 5: _LOW_OPS}

    def __init__(self) -> None:
        self.register_destination: int = 0
        self.register_write: int = 0
        self.jump: int = 0
        self.branch: int = 0
        self.memory_read: int = 0
        self.memory_2_register: int = 0
        self.memory_write: int = 0
        self.alu_op1: int = 0
        self.alu_op0: int = 0
        self.alu_source: int = 0

    # --- andFunction1..5 ---------------------------------------------------
    def and_function1(self, op_code: str) -> int:
        """Original ``andFunction1`` — 1 when bit 5 of ``op_code`` is 0."""
        return 1 if op_code in self._DECODERS[1] else 0

    def and_function2(self, op_code: str) -> int:
        """Original ``andFunction2`` — 1 when bit 5 of ``op_code`` is 1."""
        return 1 if op_code in self._DECODERS[2] else 0

    def and_function3(self, op_code: str) -> int:
        """Original ``andFunction3`` — 1 when bit 5 of ``op_code`` is 1."""
        return 1 if op_code in self._DECODERS[3] else 0

    def and_function4(self, op_code: str) -> int:
        """Original ``andFunction4`` — 1 when bit 5 of ``op_code`` is 0."""
        return 1 if op_code in self._DECODERS[4] else 0

    def and_function5(self, op_code: str) -> int:
        """Original ``andFunction5`` — 1 when bit 5 of ``op_code`` is 0."""
        return 1 if op_code in self._DECODERS[5] else 0

    # --- signal wiring ------------------------------------------------------
    def activate(self, op_code: str) -> None:
        """Set the control signals for a six-bit opcode (``activate``).

        Faithful port of the original mapping:

        * registerDestination = and1
        * registerWrite       = and1 | and2   (every opcode)
        * jump                = and5          (bit5==0 — odd, kept on purpose)
        * branch              = and4          (bit5==0 — odd, kept on purpose)
        * memoryRead          = and2          (bit5==1 — odd, kept on purpose)
        * memory2Register     = and2
        * memoryWrite         = and3
        * aluOp1              = and1
        * aluOp0              = and4
        * aluSource           = and2 | and3
        """
        and1 = self.and_function1(op_code)
        and2 = self.and_function2(op_code)
        and3 = self.and_function3(op_code)
        and4 = self.and_function4(op_code)
        and5 = self.and_function5(op_code)
        self.register_destination = and1
        self.register_write = and1 or and2
        self.jump = and5
        self.branch = and4
        self.memory_read = and2
        self.memory_2_register = and2
        self.memory_write = and3
        self.alu_op1 = and1
        self.alu_op0 = and4
        self.alu_source = and2 or and3

    # --- signal getters ------------------------------------------------------
    # The original's get* methods are plain ivar reads; the port exposes
    # attributes directly (see __init__).  ``getALUOp`` in the Ruby source
    # read an unassigned ivar (would raise), so it is not ported.


class Registers:
    """Register-file plumbing (``EMIPS::Registers``) — six named, initially
    empty fields, exposed as attributes in the port (the original had
    get/set accessor pairs)."""

    def __init__(self) -> None:
        self.read_register1 = ""
        self.read_register2 = ""
        self.read_data1 = ""
        self.read_data2 = ""
        self.write_register = ""
        self.write_data = ""


class MUX:
    """Multiplexer (``EMIPS::MUX``).

    The original is proc-based: ``select`` is a callable returning the input
    index, every entry of ``inputs`` is a callable producing a value, and
    ``output`` is a callable receiving the selected value.  Python keeps the
    same convention (pass lambdas).
    """

    def __init__(self, select, output, *inputs) -> None:
        self._select = select
        self._output = output
        self._input = list(inputs)

    def check(self) -> None:
        index = self._select()
        value = self._input[index]()
        self._output(value)


class Pipeline:
    """Five-stage pipeline bookkeeping (``EMIPS::Pipeline``)."""

    PIPELINE_STAGES = 5
    PIPELINE_STEPS = 5

    _STEPS = ("IF", "ID", "EX", "MEM", "WB")

    def __init__(self) -> None:
        self._stages: dict[int, dict[str, object]] = {}
        for stage in range(self.PIPELINE_STAGES):
            self._stages[stage] = {"currentStep": 0}
            for step in range(self.PIPELINE_STEPS):
                self._stages[stage][step] = self._STEPS[step]

    def get_next_step(self, stage: int) -> str:
        """Advance *stage* to its next step (wrapping) and return its name
        (``getNextStep``)."""
        state = self._stages[stage]
        current = int(state["currentStep"]) + 1
        if current > self.PIPELINE_STEPS - 1:
            current = 0
        state["currentStep"] = current
        return str(state[current])


class CoProcessor0:
    """CP0 register description (``EMIPS::CoProcessor0``).

    The original is ~530 lines of hand-written register definitions; the port
    loads the same data from ``data/cp0.json`` (dumped from the Ruby class at
    migration time).
    """

    def __init__(self) -> None:
        with open(_CP0_PATH, encoding="utf-8") as handle:
            blob = json.load(handle)
        self._cp0r = {int(k): {int(k2): dict(v) for k2, v in regs.items()} for k, regs in blob.items()}

    def get_cp0_registers(self) -> dict[int, dict[int, dict]]:
        return self._cp0r


class MMU:
    """``EMIPS::MMU`` — all of the original's methods were empty (returned
    ``None``); the port keeps the same no-op surface."""

    def get_p_memory(self, v_memory_address: int | str) -> None:
        """No-op in the original (virtual → physical was TODO)."""

    def get_v_memory(self, p_memory_address: int | str) -> None:
        """No-op in the original (physical → virtual was TODO)."""


class CPU:
    """Pipeline CPU scaffold (``EMIPS::CPU``).

    ``mapping_cpu=0`` (default) keeps the bootstrap PC ``1fc00000``; pass a
    mapping object with a ``pc`` attribute to override, as in the original.
    ``p_memory`` is a :class:`~emips.memory.PMemory` and ``instructions`` an
    :class:`~emips.instruction.InstructionSet`.
    """

    def __init__(
        self,
        mapping_cpu: int | object = 0,
        p_memory: PMemory | None = None,
        instructions: InstructionSet | None = None,
    ) -> None:
        self._control_unit = ControlUnit()
        self._pipeline = Pipeline()
        self._pc = "1fc00000"
        self.lo = 0
        self.hi = 0
        if mapping_cpu is not None and mapping_cpu != 0:
            self.mapping_cpu = mapping_cpu
            self._init_cpu_program_counter()
        self._r = Registers()
        self._init_cpu_regs()
        self._cp0 = CoProcessor0()
        self._cp0r = self._cp0.get_cp0_registers()
        self._cpu_modes = self._init_cpu_modes()
        self._e_memory = p_memory
        self._e_instructions = instructions
        self._stage = 0
        self._stop: bool | int = True

    # --- state accessors -----------------------------------------------------
    @property
    def control_unit(self) -> ControlUnit:
        return self._control_unit

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def pc(self) -> str:
        return self._pc

    @property
    def registers(self) -> Registers:
        return self._r

    @property
    def cp0_registers(self) -> dict[int, dict[int, dict]]:
        return self._cp0r

    @property
    def memory(self) -> PMemory | None:
        return self._e_memory

    @property
    def instructions(self) -> InstructionSet | None:
        return self._e_instructions

    @property
    def stop(self) -> bool | int:
        """``True`` while running; the error code when a fetch failed."""
        return self._stop

    @property
    def cpu_modes(self) -> dict[str, dict[str, dict[str, str]]]:
        """CPU mode table (``kernelMode`` / ``debugMode``) as initialized from
        the original's constants (``_initCPUModes`` / ``initCPUModes``)."""
        return self._cpu_modes

    # --- pipeline loop -------------------------------------------------------
    def run(self) -> bool | int:
        """Keep fetching/decoding/executing until a fetch fails, then return
        the failure code (``run``).

        Note: in the original the decode/execute/memory/writeback stages are
        empty, so with no overridden hooks this loops forever — callers that
        only want a single fetch should use :meth:`fetch_instruction`.
        """
        while True:
            self.fetch_instruction(self._stage)
            if self._stop is not True:
                return self._stop
            self.decode_instruction()
            self.execute_instruction()
            self.memory_instruction()
            self.writeback_instruction()

    def decode_instruction(self) -> None:
        """No-op stage (empty in the original)."""

    def execute_instruction(self) -> None:
        """No-op stage (empty in the original)."""

    def memory_instruction(self) -> None:
        """No-op stage (empty in the original)."""

    def writeback_instruction(self) -> None:
        """No-op stage (empty in the original)."""

    # --- single-instruction helpers ------------------------------------------
    def fetch_instruction(self, stage: int) -> bool | int:
        """Fetch four little-endian bytes at PC, look the word up in the
        instruction table and advance PC by 4 (``fetchInstruction``).

        Returns ``True`` on a match (PC advanced) or the failure code
        (``201``) when the table has no row for the word — the original
        ``@stop`` contract.
        """
        memory = ""
        for offset in range(4):
            byte = self._e_memory.read_from_memory(format(int(self._pc, 16) + offset, "x"))
            if byte is None:
                raise LookupError(f"no memory byte at 0x{int(self._pc, 16) + offset:x}")
            memory = byte + memory
        bits = format(int(memory, 16), "032b")
        matches = self._e_instructions.find(bits)
        self._stop = True if matches else OPCODE_INSTRUCTION_NOT_FOUND
        if self._stop is not True:
            return self._stop
        self._pc = format(int(self._pc, 16) + 4, "x")
        return self._stop

    def decode_single_instruction(self, instruction: str):
        """Disassemble one little-endian 8-hex-digit word (memory as seen in
        ``decodeSingleInstruction``): returns ``(memory, bytes, binary,
        matches)``.

        The original only printed its progress and returned nothing; the
        port returns the computed values so callers can use them.
        """
        if len(instruction) != 8:
            raise ValueError(f"expected an 8-hex-digit word, got {instruction!r}")
        memory = ""
        for offset in range(4):
            byte = instruction[offset * 2 : offset * 2 + 2].upper()
            memory = byte + memory
        bits = format(int(memory, 16), "032b")
        matches = self._e_instructions.find(bits)
        return memory, [instruction[i : i + 2] for i in range(0, 8, 2)], bits, matches

    # --- register / mode init -------------------------------------------------
    def _init_cpu_program_counter(self) -> None:
        self._pc = self.mapping_cpu.pc

    def _init_cpu_regs(self) -> None:
        """Standard MIPS register file names (original ``initCPURegs``).

        ``@r[index] = {value: 0, name: ...}`` in the original; the port
        exposes it via the plain :class:`Registers` plumbing plus the named
        table on the attribute :attr:`register_file` for lookup convenience.
        """
        self.register_file: dict[int, dict] = {}
        for index in range(32):
            if index == 0:
                name = "zero"
            elif index == 1:
                name = "at"
            elif 2 <= index <= 3:
                name = f"v{index - 2}"
            elif 4 <= index <= 7:
                name = f"a{index - 4}"
            elif 8 <= index <= 15:
                name = f"t{index - 8}"
            elif 16 <= index <= 23:
                name = f"s{index - 16}"
            elif 24 <= index <= 25:
                name = f"t{index - 24}"
            elif 26 <= index <= 27:
                name = f"k{index - 26}"
            elif index == 28:
                name = "gp"
            elif index == 29:
                name = "sp"
            elif index == 30:
                name = "fp"
            else:
                name = "ra"
            self.register_file[index] = {"value": 0, "name": name}

    def get_registers(self) -> Registers:
        """The register-file plumbing (``getRegisters``)."""
        return self._r

    def get_cp0_registers(self) -> dict[int, dict[int, dict]]:
        """The CP0 register table (``getCP0Registers``)."""
        return self._cp0r

    @staticmethod
    def _init_cpu_modes() -> dict:
        """Segment maps for the two CPU modes (original ``initCPUModes``)."""
        kernel = {
            "kuseg": {"start": "00000000", "end": "7FFFFFFF"},
            "kseg0": {"start": "80000000", "end": "9FFFFFFF"},
            "kseg1": {"start": "A0000000", "end": "BFFFFFFF"},
            "kseg2": {"start": "C0000000", "end": "DFFFFFFF"},
            "kseg3": {"start": "E0000000", "end": "FFFFFFFF"},
        }
        debug = {
            "kuseg": {"start": "00000000", "end": "7FFFFFFF"},
            "kseg0": {"start": "80000000", "end": "9FFFFFFF"},
            "kseg1": {"start": "A0000000", "end": "BFFFFFFF"},
            "kseg2": {"start": "C0000000", "end": "DFFFFFFF"},
            "kseg3_1": {"start": "E0000000", "end": "FF1FFFFF"},
            "dseg": {"start": "FF200000", "end": "FF3FFFFF"},
            "kseg3_2": {"start": "FF400000", "end": "FFFFFFFF"},
        }
        return {"KernelMode": kernel, "DebugMode": debug}
