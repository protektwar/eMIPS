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

MIPS64 (``arch="mips64"``, PLAN M22): the n64 ABI (``-mabi=64``: 64-bit
``long`` and pointers) — ``-march`` ``mips64`` / ``mips64r2`` /
``mips64r6`` — an ELF64 linked with the same memory map (GNU ld's
``elf64btsmip`` / ``elf64ltsmip``).  GCC builds with ``-msym32`` (every
symbol in the 32-bit compatibility space: two-instruction addresses, as
the MIPS64 assembler's programs); clang has no ``-msym32`` and loads
addresses with its 64-bit sequences, the same values.

nanoMIPS (``arch="nanomips"``, PLAN M23): MediaTek's ``nanomips-elf-gcc``
(GCC 11.2, nanoMIPS-2022.03-01: the later releases ship no compiler) and
its gold linker; the p32 ABI (DN00179), ``-march=32r6`` (an NMS core:
``32r6s``, the nanoMIPS subset), soft float (nanoMIPS has no FPU),
``emips/c/crt0_nano.S`` and ``emips/c/emips_nano.ld`` (the nanoMIPS
assembler's layout: code at 0x00400000, at 0x10010000 the data, ``_gp``
and the small data the compiler reaches from ``$gp``); ``emips.h`` as for
MIPS32 (the services keep their register numbers).  The toolchain's
libgcc (soft float, 64-bit division) when it has one for the byte order
(nanoMIPS-2022.03-01: little-endian -march=32r6 only).

RISC-V (``arch="rv32"`` / ``"rv64"``, PLAN M25): a RISC-V GCC
(``riscv64-unknown-elf-gcc`` builds both widths; ``riscv32-*``,
``riscv64-linux-gnu-*``) or clang ``--target=riscv32 / riscv64-unknown-elf``,
with GNU ld (``elf32lriscv`` / ``elf64lriscv``) or ld.lld; ``-march`` = the
hart's extensions (``rv64imafdc_zicsr_zifencei_...``), ``-mabi`` the
psABI's for them (``ilp32`` / ``lp64``, ``f`` / ``d`` with the F / D
registers), medlow addresses, little-endian; ``emips/c/crt0_rv.S`` and
``emips/c/emips_rv.ld`` (RARS's map: code at 0x00400000, data at
0x10010000, ``__global_pointer$`` by GNU ld's formula — the linker relaxes
accesses near it to gp-relative ones); ``emips.h`` makes RARS's ``ecall``
services (a7 = the service).  The GCC's libgcc multilib for the
``-march`` / ``-mabi`` (soft float, division without M).
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
MARCH64 = {"R1": "mips64", "R2": "mips64r2", "R3": "mips64r2", "R5": "mips64r2",
           "R6": "mips64r6"}
GCC_PREFIXES = ("mips-mti-elf", "mips-img-elf", "mips-elf", "mips-linux-gnu")
#: the nanoMIPS toolchain's prefix (MediaTek's nanomips-elf tools)
NANO_PREFIX = "nanomips-elf"


#: RISC-V GCCs (a riscv64-unknown-elf GCC builds rv32 too: its multilibs)
RV_PREFIXES = ("riscv64-unknown-elf", "riscv32-unknown-elf", "riscv64-elf", "riscv32-elf",
               "riscv64-linux-gnu", "riscv64-unknown-linux-gnu")


#: a link error naming a soft-float routine (__addsf3, __fixdfsi, __ltsf2 ...)
_SOFT_FP = re.compile(r"undefined reference to `__\w*[sd]f\w*'")
#: ... any libgcc routine (gold quotes it '__divdi3', GNU ld `__divdi3')
_LIBGCC = re.compile(r"undefined reference to [`']__\w+'")


class BuildError(RuntimeError):
    """No toolchain, or the compiler / linker failed (the message has its
    output)."""


@dataclass
class Toolchain:
    cc: list[str]                    # the compiler command (with its target)
    ld: list[str]                    # the linker command
    kind: str                        # "gcc" or "clang"
    notes: list[str] = field(default_factory=list)


def find_toolchain(endian: str = "big", arch: str = "mips32") -> Toolchain:
    """A MIPS GCC, else clang (any clang has the MIPS back end); GNU ld
    of the same prefix, a MIPS binutils' ld, else ld.lld.  (A MIPS GCC
    and binutils build both ABIs: ``-mabi=64`` and ``ld -m elf64*``.)"""
    el = endian == "little"
    wide = arch == "mips64"
    if arch == "nanomips":                  # (one GCC, both byte orders; its gold)
        gcc, ld = shutil.which(f"{NANO_PREFIX}-gcc"), shutil.which(f"{NANO_PREFIX}-ld")
        if gcc and ld:
            return Toolchain([gcc, "-EL" if el else "-EB"], [ld], "gcc")
        raise BuildError("no nanoMIPS C compiler found: put the bin folder of MediaTek's "
                         "nanoMIPS GNU tools (nanoMIPS-2022.03-01: the later releases ship "
                         "no nanomips-elf-gcc) on PATH")
    if arch in ("rv32", "rv64"):
        return _rv_toolchain(arch)
    for prefix in GCC_PREFIXES:
        for p in ((prefix.replace("mips-", "mipsel-", 1),) if el else ()) + (prefix,):
            gcc = shutil.which(f"{p}-gcc")
            if gcc:
                ld = shutil.which(f"{p}-ld") or shutil.which("ld.lld")
                if ld:
                    return Toolchain([gcc, "-EL" if el else "-EB"], [ld], "gcc")
    clang = shutil.which("clang")
    if clang:
        target = f"mips{'64' if wide else ''}{'el' if el else ''}-unknown-elf"
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


def _rv_toolchain(arch: str) -> Toolchain:
    for prefix in RV_PREFIXES:
        if arch == "rv64" and prefix.startswith("riscv32"):
            continue
        gcc, ld = shutil.which(f"{prefix}-gcc"), shutil.which(f"{prefix}-ld")
        if gcc and (ld or shutil.which("ld.lld")):
            return Toolchain([gcc], [ld or shutil.which("ld.lld")], "gcc")
    clang = shutil.which("clang")
    if clang:
        ld = next((shutil.which(f"{p}-ld") for p in RV_PREFIXES if shutil.which(f"{p}-ld")),
                  None) or shutil.which("ld.lld")
        if ld:
            return Toolchain([clang, f"--target=riscv{arch[2:]}-unknown-elf"], [ld], "clang")
    raise BuildError("no RISC-V C compiler found: install a RISC-V GCC (e.g. "
                     "gcc-riscv64-unknown-elf with binutils-riscv64-unknown-elf) or clang "
                     "with ld.lld")


#: the order -march names the single-letter extensions in (the ISA manual's
#: canonical order) and the multi-letter ones' categories (by their 2nd letter)
_RV_LETTERS = "iemafdqlcbkjtpvh"


def rv_march(arch: str, extensions: str) -> tuple[str, str]:
    """(-march, -mabi) for a hart with ``extensions``: the canonical ISA
    string, the psABI's integer (ilp32 / lp64) or hardware floating-point
    calling convention (``f`` / ``d`` suffix)."""
    from .riscv import parse_extensions
    exts = parse_extensions(extensions) | {"i"}
    letters = "".join(c for c in _RV_LETTERS if c in exts)
    multi = sorted((e for e in exts if len(e) > 1),
                   key=lambda e: (e[0] != "z", _RV_LETTERS.find(e[1]) if e[0] == "z" else 0, e))
    march = f"rv{arch[2:]}{letters}" + "".join("_" + e for e in multi)
    abi = ("ilp32" if arch == "rv32" else "lp64") + ("d" if "d" in exts else
                                                      "f" if "f" in exts else "")
    return march, abi


def cflags(release: str = "R2", optimize: str = "-O1", fpu: bool = True,
           arch: str = "mips32", kind: str = "gcc", nms: bool = False) -> list[str]:
    if arch == "nanomips":                  # p32, soft float (no nanoMIPS FPU)
        return [f"-march={'32r6s' if nms else '32r6'}", "-m32", "-msoft-float",
                "-ffreestanding", "-fno-common", "-nostdlib", "-g", optimize, "-Wall",
                f"-I{C_DIR}"]
    if arch == "mips64":                    # n64 (clang: no -msym32)
        abi = [f"-march={MARCH64.get(release, 'mips64r2')}", "-mabi=64"] + (
            ["-msym32"] if kind == "gcc" else [])
    else:
        abi = [f"-march={MARCH.get(release, 'mips32r2')}", "-mabi=32"]
    return abi + ["-mhard-float" if fpu else "-msoft-float",
            "-fno-pic", "-mno-abicalls", "-G0", "-ffreestanding", "-fno-common",
            "-nostdlib", "-g", optimize, "-Wall", f"-I{C_DIR}"]


def build(sources: list[str | Path], output: str | Path, *, release: str = "R2",
          endian: str = "big", optimize: str = "-O1", extra: list[str] | None = None,
          toolchain: Toolchain | None = None, mips16: bool = False,
          micromips: bool = False, fpu: bool = True, dsp: int = 0,
          arch: str = "mips32", nms: bool = False, msa: bool = False,
          extensions: str = "imafdc_zicsr_zifencei_zicntr_zihpm_zba_zbb") -> str:
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
    say so, and a core gets it).  ``arch="mips64"``: an n64 ELF64 for a
    MIPS64 core; ``micromips`` there = microMIPS64 (Release 3 / 5, a MIPS
    GCC: clang has no microMIPS64), ``mips16`` MIPS16e64 (MD00077; GCC
    builds MIPS16 n64 code soft-float only — "'hard-float' MIPS16 code for
    ABIs other than o32 and o64" is unimplemented —, so the whole program
    is soft-float, as without an FPU), ``dsp`` the DSP Module (MD00375).
    ``msa``: the C files may use MSA (MD00866: ``-mmsa`` — GCC's
    ``msa.h`` / ``__builtin_msa_*``, and the auto-vectorizer at -O3);
    every object gets ``-mfp64`` (MSA needs Status.FR = 1: one FP ABI
    for the link) and ``-mnan=2008`` (an MSA core's FPU is IEEE
    754-2008), Release 5 / 6 (``-march=mips32r5`` / ``mips64r5``).
    ``arch="nanomips"``: a p32 nanoMIPS executable (``nms``: for a core with
    the nanoMIPS subset only; ``dsp``: the DSP ASE, ``-mdsp``); no compressed
    ISAs.  ``arch="rv32"`` / ``"rv64"``: a RISC-V executable for a hart
    with ``extensions`` (-march / -mabi from them: :func:`rv_march`)."""
    if arch in ("rv32", "rv64"):
        if mips16 or micromips or dsp or msa or nms:
            raise BuildError("MIPS16e / microMIPS / the DSP Module / MSA / NMS are MIPS "
                             "options: a RISC-V program takes the hart's extensions")
        return _build_rv(sources, output, arch, extensions, optimize, extra, toolchain)
    if arch == "nanomips":
        return _build_nano(sources, output, endian, optimize, extra, toolchain, nms,
                           mips16 or micromips, dsp)
    if mips16 and release == "R6":
        raise BuildError("Release 6 has no MIPS16e")
    if arch == "mips64" and micromips and release in ("R1", "R2"):
        raise BuildError("microMIPS64 needs Release 3 or later (MD00594 3.1): build with "
                         "--release R3 or R5")
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
    if msa and release not in ("R5", "R6"):
        raise BuildError("MSA C needs Release 5 or 6 (MD00866 1.1): build with --release "
                         "R5 or R6")
    if msa and not fpu:
        raise BuildError("MSA needs the FPU: its vector registers extend the FPU's "
                         "(MD00866 3.1)")
    if msa and (mips16 or micromips):
        raise BuildError("MSA C is MIPS32 / MIPS64 code here (GCC has no microMIPS MSA)")
    if dsp and release in ("R1", "R6"):
        raise BuildError("DSP C needs Release 2, 3 or 5: the DSP Module needs Release 2+, "
                         "and the compilers have no -mdspr3 for Release 6")
    tc = toolchain or find_toolchain(endian, arch)
    if micromips and arch == "mips64" and tc.kind != "gcc":
        raise BuildError("microMIPS64 C needs a MIPS GCC (clang: \"micromips is not "
                         "supported for target CPU 'mips64r2'\"): install gcc-mips-linux-gnu")
    if mips16:
        if tc.kind != "gcc":
            raise BuildError("MIPS16e C needs a MIPS GCC (clang's MIPS16 back end does not "
                             "work): install gcc-mips-linux-gnu")
    #: soft-float code: no FPU, or MIPS16e64 (GCC has no hard-float MIPS16
    #: n64: the FPU stays, the program does not use it)
    soft = not fpu or (mips16 and arch == "mips64")
    flags = cflags(release, optimize, not soft, arch, tc.kind) + list(extra or [])
    if msa:                                 # (every object: one FP ABI, 2008 NaNs)
        flags = [f"-march={'mips64r5' if arch == 'mips64' else 'mips32r5'}"
                 if f.startswith("-march=") and release == "R5" else f for f in flags]
        flags += ["-mfp64", "-mnan=2008"]
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
                   + (["-mmsa"] if msa else [])
                   if src.suffix == ".c" else [])
            cmd = tc.cc + flags + isa + ["-c", str(src), "-o", str(obj)]
            _run(cmd, log)
            objs.append(str(obj))
        # -n: no page alignment of the segments in the file (bare metal)
        emulation = []                      # (GNU ld: the n64 ELF64 emulation)
        if arch == "mips64" and "lld" not in Path(tc.ld[0]).name:
            emulation = ["-m", "elf64ltsmip" if endian == "little" else "elf64btsmip"]
        cmd = tc.ld + emulation + ["-EL" if endian == "little" else "-EB", "-n",
                                   "-T", str(C_DIR / "emips.ld"), "-o", str(output)] + objs
        lib = None if not soft else soft_libgcc(endian, arch)
        if lib is not None:                 # the soft-float routines (__addsf3 ...)
            cmd.append(lib)
        try:
            _run(cmd, log)
        except BuildError as exc:
            if soft and lib is None and _SOFT_FP.search(str(exc)):
                why = ("float arithmetic on a core without an FPU" if not fpu else
                       "float arithmetic in MIPS16e64 C (GCC builds MIPS16 n64 code "
                       "soft-float only)")
                raise BuildError(f"{exc}\n({why} calls the soft-float routines __addsf3 "
                                 "...: they come from a MIPS GCC with a soft-float "
                                 "libgcc, e.g. mips-mti-elf-gcc; Debian's mips-linux-gnu "
                                 "has a hard-float one only)") from None
            raise
    return "\n".join(log)


def _build_nano(sources, output, endian, optimize, extra, toolchain, nms, compressed,
                dsp) -> str:
    if compressed:
        raise BuildError("nanoMIPS has no MIPS16e / microMIPS: its 16-bit instructions are "
                         "part of the one nanoMIPS ISA (MD01247)")
    tc = toolchain or find_toolchain(endian, "nanomips")
    flags = cflags("R6", optimize, False, "nanomips", tc.kind, nms) + list(extra or [])
    if dsp:                         # the DSP ASE (MD01249, Revision 3): GCC's -mdsp
        flags.append("-mdsp")       # (__builtin_mips_*; .nanoMIPS.abiflags: DSPR3)
    log: list[str] = []
    with tempfile.TemporaryDirectory(prefix="emips-cc-") as tmp:
        objs = []
        for src in [C_DIR / "crt0_nano.S", C_DIR / "libemips.c", *map(Path, sources)]:
            obj = Path(tmp) / (("crt0_nano" if src.name == "crt0_nano.S"   # (the linker
                                else f"{src.stem}_{len(objs)}") + ".o")    # script names it)
            _run(tc.cc + flags + ["-c", str(src), "-o", str(obj)], log)
            objs.append(str(obj))
        # -n: no page alignment of the segments in the file (bare metal)
        cmd = tc.ld + ["-EL" if endian == "little" else "-EB", "-n", "-T",
                       str(C_DIR / "emips_nano.ld"), "-o", str(output)] + objs
        lib = _nano_libgcc(tc, flags)
        if lib is not None:                 # (soft float, 64-bit division ...)
            cmd.append(lib)
        try:
            _run(cmd, log)
        except BuildError as exc:
            if lib is None and _LIBGCC.search(str(exc)):
                march = next(f for f in flags if f.startswith("-march="))
                raise BuildError(f"{exc}\n(the code calls libgcc's routines — soft-float "
                                 "arithmetic, 64-bit division ...: this nanoMIPS toolchain "
                                 f"has no libgcc for {endian}-endian {march})") from None
            raise
    return "\n".join(log)


def _build_rv(sources, output, arch, extensions, optimize, extra, toolchain) -> str:
    tc = toolchain or find_toolchain("little", arch)
    march, abi = rv_march(arch, extensions)
    flags = [f"-march={march}", f"-mabi={abi}", "-mcmodel=medlow", "-ffreestanding",
             "-fno-common", "-nostdlib", "-g", optimize, "-Wall", f"-I{C_DIR}"] + list(extra or [])
    log: list[str] = []
    with tempfile.TemporaryDirectory(prefix="emips-cc-") as tmp:
        objs = []
        for src in [C_DIR / "crt0_rv.S", C_DIR / "libemips.c", *map(Path, sources)]:
            obj = Path(tmp) / (("crt0_rv" if src.name == "crt0_rv.S"     # (the linker
                                else f"{src.stem}_{len(objs)}") + ".o")  # script names it)
            _run(tc.cc + flags + ["-c", str(src), "-o", str(obj)], log)
            objs.append(str(obj))
        emulation = [] if "lld" in Path(tc.ld[0]).name else [
            "-m", "elf32lriscv" if arch == "rv32" else "elf64lriscv"]
        # -n: no page alignment of the segments in the file (bare metal)
        cmd = tc.ld + emulation + ["-n", "-T", str(C_DIR / "emips_rv.ld"), "-o",
                                   str(output)] + objs
        lib = _nano_libgcc(tc, flags) if tc.kind == "gcc" else None
        if lib is not None:                 # (soft float, division without M ...)
            cmd.append(lib)
        try:
            _run(cmd, log)
        except BuildError as exc:
            if lib is None and _LIBGCC.search(str(exc)):
                raise BuildError(f"{exc}\n(the code calls libgcc's routines — soft-float "
                                 "arithmetic, division without the M extension ...: they come "
                                 f"from a RISC-V GCC's libgcc for -march={march} "
                                 f"-mabi={abi})") from None
            raise
    return "\n".join(log)


def _nano_libgcc(tc: Toolchain, flags: list[str]) -> str | None:
    """The (nanoMIPS / RISC-V) GCC's libgcc.a for these flags (its multilib), or
    None (nanoMIPS-2022.03-01 lists big-endian and -march=32r6s
    multilibs it does not install: GCC then prints a bare ``libgcc.a``)."""
    try:
        r = subprocess.run(tc.cc + flags + ["-print-libgcc-file-name"], capture_output=True,
                           text=True, timeout=30)
    except OSError:
        return None
    path = r.stdout.strip()
    return path if r.returncode == 0 and Path(path).is_absolute() and Path(path).is_file() \
        else None


def soft_libgcc(endian: str = "big", arch: str = "mips32") -> str | None:
    """A MIPS GCC's soft-float libgcc.a (o32 / n64: __addsf3 ... in
    integer code), or None — a hard-float libgcc implements them with the
    FPU's instructions (Debian's mips-linux-gnu has only that one)."""
    for prefix in GCC_PREFIXES:
        p = prefix.replace("mips-", "mipsel-", 1) if endian == "little" else prefix
        gcc = shutil.which(f"{p}-gcc") or (shutil.which(f"{prefix}-gcc")
                                            if endian == "little" else None)
        if not gcc:
            continue
        try:
            flags = [gcc, "-mabi=64" if arch == "mips64" else "-mabi=32",
                     "-msoft-float"] + (["-EL"] if endian == "little" else [])
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
