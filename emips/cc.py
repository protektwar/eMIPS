"""Building C programs for eMIPS (PLAN M13): ``python3 -m emips cc``.

eMIPS ships no compiler: this finds one — a MIPS GCC (``mips-mti-elf-gcc``,
``mips-elf-gcc``, ``mips-linux-gnu-gcc`` and the little-endian
``mipsel-*``) or ``clang`` with a MIPS target — and a linker (GNU ``ld``
of a MIPS binutils, or ``ld.lld``), and builds a bare-metal executable:
``emips/c/crt0.S`` (sets ``$gp`` / ``$sp``, clears ``.bss``, calls
``main``, exits with its value), ``emips/c/libemips.c`` (``memcpy`` and
friends) and your sources, linked with ``emips/c/emips.ld`` (code at
0x00400000, data at 0x10010000, the eMIPS / MARS memory map).  Include
``emips.h`` for the system calls (``print_int``, ``print_string``,
``read_int``, ``exit`` ...).

The flags: no PIC / abicalls (a static program: ``-fno-pic
-mno-abicalls``), no small-data section (``-G0``), freestanding, o32,
hard float, DWARF line information (``-g``) so the debugger and the GUI
show the C lines; ``-march`` from the release (R1 ``mips32``, R2 / R3 /
R5 ``mips32r2``, R6 ``mips32r6``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

C_DIR = Path(__file__).parent / "c"
MARCH = {"R1": "mips32", "R2": "mips32r2", "R3": "mips32r2", "R5": "mips32r2",
         "R6": "mips32r6"}
GCC_PREFIXES = ("mips-mti-elf", "mips-img-elf", "mips-elf", "mips-linux-gnu")


#: a link error naming a soft-float routine (__addsf3, __fixdfsi, __ltsf2 ...)
_SOFT_FP = re.compile(r"undefined reference to `__\w*[sd]f\w*'")


class BuildError(RuntimeError):
    """No toolchain, or the compiler / linker failed (the message has its
    output)."""


@dataclass
class Toolchain:
    cc: list[str]                    # the compiler command (with its target)
    ld: list[str]                    # the linker command
    kind: str                        # "gcc" or "clang"
    notes: list[str] = field(default_factory=list)


def find_toolchain(endian: str = "big") -> Toolchain:
    """A MIPS GCC, else clang (any clang has the MIPS back end); GNU ld
    of the same prefix, a MIPS binutils' ld, else ld.lld."""
    el = endian == "little"
    for prefix in GCC_PREFIXES:
        for p in ((prefix.replace("mips-", "mipsel-", 1),) if el else ()) + (prefix,):
            gcc = shutil.which(f"{p}-gcc")
            if gcc:
                ld = shutil.which(f"{p}-ld") or shutil.which("ld.lld")
                if ld:
                    return Toolchain([gcc, "-EL" if el else "-EB"], [ld], "gcc")
    clang = shutil.which("clang")
    if clang:
        target = "mipsel-unknown-elf" if el else "mips-unknown-elf"
        for p in (("mipsel-linux-gnu", "mips-linux-gnu") if el else ("mips-linux-gnu",)) + \
                ("mips-elf", "mips-mti-elf"):
            ld = shutil.which(f"{p}-ld")
            if ld:
                return Toolchain([clang, "-target", target], [ld], "clang")
        lld = shutil.which("ld.lld")
        if lld:
            return Toolchain([clang, "-target", target], [lld], "clang")
    raise BuildError("no MIPS C compiler found: install clang (with ld.lld or a MIPS "
                     "binutils such as binutils-mips-linux-gnu) or a MIPS GCC "
                     "(e.g. gcc-mips-linux-gnu)")


def cflags(release: str = "R2", optimize: str = "-O1", fpu: bool = True) -> list[str]:
    return [f"-march={MARCH.get(release, 'mips32r2')}", "-mabi=32",
            "-mhard-float" if fpu else "-msoft-float",
            "-fno-pic", "-mno-abicalls", "-G0", "-ffreestanding", "-fno-common",
            "-nostdlib", "-g", optimize, "-Wall", f"-I{C_DIR}"]


def build(sources: list[str | Path], output: str | Path, *, release: str = "R2",
          endian: str = "big", optimize: str = "-O1", extra: list[str] | None = None,
          toolchain: Toolchain | None = None, mips16: bool = False,
          micromips: bool = False, fpu: bool = True, dsp: int = 0) -> str:
    """Compile and link ``sources`` (C or assembly .S) into the ELF
    ``output``; returns the commands run (for the user to see).  Raises
    :class:`BuildError`.  ``mips16``: the C files as MIPS16e (``-mips16``;
    crt0 stays MIPS32, the linker turns calls between the ISAs into JALX;
    a MIPS GCC: clang's MIPS16 back end does not work).  ``micromips``:
    the C files as microMIPS (``-mmicromips``, Release 2 / 3 / 5, GCC or
    clang; crt0 MIPS32, JALX as for MIPS16e).  ``fpu=False``: a core
    without an FPU (the 24Kc, the M14K): ``-msoft-float``, the float
    arithmetic from a MIPS GCC's libgcc.  ``dsp``: the C files may use the
    DSP Module (1: ``-mdsp``, 2: ``-mdspr2`` — the builtins
    ``__builtin_mips_*`` and the vector types; the ELF's .MIPS.abiflags
    say so, and a core gets it)."""
    if mips16 and micromips:
        raise BuildError("MIPS16e or microMIPS: a core has one of them")
    if micromips and release == "R1":
        raise BuildError("microMIPS needs Release 2 or later (the M14K is a Release 2 core "
                         "with microMIPS): build with --release R2, R3 or R5")
    if micromips and release == "R6":
        raise BuildError("microMIPS Release 6 C: no compiler agrees with the manual — GNU as "
                         "has no microMIPS R6, clang 18 shifts the compact branch offsets "
                         "(BEQC ...) by 2 where MD00582 6.05's pseudocode and QEMU shift "
                         "them by 1, and its la / li put MIPS32 LUI encodings in microMIPS "
                         "code; write microMIPS R6 in assembly (.set micromips)")
    if dsp and release in ("R1", "R6"):
        raise BuildError("DSP C needs Release 2, 3 or 5: the DSP Module needs Release 2+, "
                         "and the compilers have no -mdspr3 for Release 6")
    tc = toolchain or find_toolchain(endian)
    if mips16:
        if release == "R6":
            raise BuildError("Release 6 has no MIPS16e")
        if tc.kind != "gcc":
            raise BuildError("MIPS16e C needs a MIPS GCC (clang's MIPS16 back end does not "
                             "work): install gcc-mips-linux-gnu")
    flags = cflags(release, optimize, fpu) + list(extra or [])
    log = []
    with tempfile.TemporaryDirectory(prefix="emips-cc-") as tmp:
        objs = []
        for src in [C_DIR / "crt0.S", C_DIR / "libemips.c", *map(Path, sources)]:
            obj = Path(tmp) / (("crt0" if src.name == "crt0.S" else src.stem)
                               + f"_{len(objs)}.o")
            if src.name == "crt0.S":
                obj = Path(tmp) / "crt0.o"           # the linker script names it
            isa = ((["-mips16"] if mips16 else ["-mmicromips"] if micromips else [])
                   + ([] if not dsp else ["-mdsp"] if dsp == 1 else ["-mdspr2"])
                   if src.suffix == ".c" else [])
            cmd = tc.cc + flags + isa + ["-c", str(src), "-o", str(obj)]
            _run(cmd, log)
            objs.append(str(obj))
        # -n: no page alignment of the segments in the file (bare metal)
        cmd = tc.ld + ["-EL" if endian == "little" else "-EB", "-n",
                       "-T", str(C_DIR / "emips.ld"), "-o", str(output)] + objs
        lib = None if fpu else soft_libgcc(endian)
        if lib is not None:                 # the soft-float routines (__addsf3 ...)
            cmd.append(lib)
        try:
            _run(cmd, log)
        except BuildError as exc:
            if not fpu and lib is None and _SOFT_FP.search(str(exc)):
                raise BuildError(f"{exc}\n(float arithmetic on a core without an FPU "
                                 "calls the soft-float routines __addsf3 ...: they come "
                                 "from a MIPS GCC with a soft-float libgcc, e.g. "
                                 "mips-mti-elf-gcc; Debian's mips-linux-gnu has a "
                                 "hard-float one only)") from None
            raise
    return "\n".join(log)


def soft_libgcc(endian: str = "big") -> str | None:
    """A MIPS GCC's soft-float libgcc.a (o32: __addsf3 ... in integer
    code), or None — a hard-float libgcc implements them with the FPU's
    instructions (Debian's mips-linux-gnu has only that one)."""
    for prefix in GCC_PREFIXES:
        p = prefix.replace("mips-", "mipsel-", 1) if endian == "little" else prefix
        gcc = shutil.which(f"{p}-gcc") or (shutil.which(f"{prefix}-gcc")
                                            if endian == "little" else None)
        if not gcc:
            continue
        try:
            flags = [gcc, "-mabi=32", "-msoft-float"] + (["-EL"] if endian == "little" else [])
            d = subprocess.run(flags + ["-print-multi-directory"], capture_output=True,
                               text=True, timeout=30)
            r = subprocess.run(flags + ["-print-libgcc-file-name"],
                               capture_output=True, text=True, timeout=30)
        except OSError:
            continue
        path = r.stdout.strip()
        if ("soft" in d.stdout and r.returncode == 0 and Path(path).is_file()):
            return path
    return None


def _run(cmd: list[str], log: list[str]) -> None:
    log.append(" ".join(cmd))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except OSError as exc:
        raise BuildError(f"cannot run {cmd[0]}: {exc}") from None
    if r.returncode != 0:
        raise BuildError(f"{' '.join(cmd)}\n{r.stdout}{r.stderr}".rstrip())
    if r.stderr.strip():
        log.append(r.stderr.rstrip())
