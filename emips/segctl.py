"""Segmentation Control and EVA (PLAN M9 phase 7).

Reference: *MIPS32 Architecture For Programmers Volume III* (MD00090,
Rev. 6.02): 4.10 "Segmentation Control" (the ``SegmentLookup``
pseudo code), 4.11 "Enhanced Virtual Addressing" (Tables 4.5, 4.7, 4.8),
9.15-9.17 SegCtl0-2 (Tables 9.27-9.30), 9.50 Config5 (K, CV, EVA).

With Segmentation Control (``Config3.SC = 1``) the six regions of the
32-bit address space are no longer fixed: each has a *Segment
Configuration* (16 bits, two per SegCtl register)::

    15     9 8 7 6  4 3  2  0
    |  PA  |0 0| AM |EU| C |

* **AM** — who may use the region and whether it is mapped (Table 9.28);
* **PA** — the physical address bits when unmapped (bits 35:29, or 35:30
  for the two 1 GB regions);
* **C** — the cacheability attribute when unmapped;
* **EU** — the region becomes unmapped and uncached while Status.ERL = 1.

``CFG0`` covers 0xE0000000-0xFFFFFFFF (kseg3), ``CFG1`` 0xC0000000-
(kseg2), ``CFG2`` 0xA0000000- (kseg1), ``CFG3`` 0x80000000- (kseg0),
``CFG4`` 0x40000000-0x7FFFFFFF and ``CFG5`` 0x00000000-0x3FFFFFFF.  The
reset value is the legacy map of Table 9.29 (eMIPS: the values QEMU's
P5600 model resets to, C = 0 / 3 / 2 where the table leaves it open).

An **EVA** instruction (LBE, SWE, CACHEE ...) is checked and translated as
if the processor were in user mode (Table 4.7), so a kernel reaches user
memory through the user's mapping (``Machine._eva``).

eMIPS choices (``sim.DEVIATIONS``): AM = 6 (undefined) behaves as UK; the
reset / BEV vectors need no overlay segment (Solution 1 of 4.10.1.2: the
reset configuration keeps CFG2 unmapped and uncached); physical addresses
are 32 bits, so PA values above 7 (and bit 6 of the 1 GB regions) wrap.
"""

from __future__ import annotations

from . import cp0 as C

M32 = 0xFFFFFFFF
SEGCTL0, SEGCTL1, SEGCTL2 = (5, 2), (5, 3), (5, 4)
CONFIG5 = (16, 5)
#: Config5 bits (Vol. III 9.50)
CFG5_K, CFG5_CV, CFG5_EVA = 1 << 30, 1 << 29, 1 << 28
#: Config3.SC
CFG3_SC = 1 << 25

#: access modes (Table 9.28)
UK, MK, MSK, MUSK, MUSUK, USK, UUSK = 0, 1, 2, 3, 4, 5, 7
AM_NAMES = {UK: "UK", MK: "MK", MSK: "MSK", MUSK: "MUSK", MUSUK: "MUSUK",
            USK: "USK", 6: "UK (6, undefined)", UUSK: "UUSK"}
KERNEL, SUPER, USER = "kernel", "supervisor", "user"

#: (allowed for user, supervisor, kernel; mapped for user, supervisor, kernel)
_AM = {UK: ((False, False, True), (False, False, False)),
       MK: ((False, False, True), (True, True, True)),
       MSK: ((False, True, True), (True, True, True)),
       MUSK: ((True, True, True), (True, True, True)),
       MUSUK: ((True, True, True), (True, True, False)),
       USK: ((False, True, True), (False, False, False)),
       6: ((False, False, True), (False, False, False)),
       UUSK: ((True, True, True), (False, False, False))}
_LEVEL = {USER: 0, SUPER: 1, KERNEL: 2}

#: SegCtl0-2 after reset: the legacy map (Table 9.29), as QEMU's P5600
#: model resets them: CFG0 kseg3 MK, CFG1 kseg2 MSK, CFG2 kseg1 UK C=2,
#: CFG3 kseg0 UK C=3, CFG4 / CFG5 kuseg MUSK EU=1 (PA 2 / 0), C=2.
RESET = (0x00200010, 0x00030002, 0x003A043A)

#: EVA's 3 GB user configuration (Table 4.5)
EVA_3GB = ((6 << 9 | MK << 4 | 3) << 16 | (7 << 9 | MK << 4 | 3),
           (4 << 9 | MUSUK << 4 | 8 | 3) << 16 | (5 << 9 | MUSUK << 4 | 8 | 3),
           (0 << 9 | MUSUK << 4 | 8 | 3) << 16 | (2 << 9 | MUSUK << 4 | 8 | 3))


def region(va: int) -> int:
    """The Segment Configuration (0..5) of a virtual address."""
    return (5, 5, 4, 4, 3, 2, 1, 0)[(va & M32) >> 29]


def cfg(m, n: int) -> int:
    """Segment Configuration ``n`` (16 bits) of a machine."""
    reg = (SEGCTL0, SEGCTL1, SEGCTL2)[n // 2]
    return (m.cp0[reg]["value"] >> (16 * (n % 2))) & 0xFFFF


def fields(c: int) -> tuple[int, int, int, int]:
    """``(PA, AM, EU, C)`` of a Segment Configuration."""
    return (c >> 9) & 0x7F, (c >> 4) & 7, (c >> 3) & 1, c & 7


def level(m) -> str:
    """The privilege level of an access: user for an EVA instruction
    (Table 4.7), else the processor's mode."""
    if m._eva:
        return USER
    status = m.cp0[C.STATUS]["value"]
    if m._dm or C.kernel_mode(status):       # Debug Mode: kernel rights
        return KERNEL
    return SUPER if C.ksu(status) == 1 else USER


def allowed(am: int, lvl: str) -> bool:
    return _AM[am][0][_LEVEL[lvl]]


def mapped(am: int, lvl: str) -> bool:
    return _AM[am][1][_LEVEL[lvl]]


def unmapped_pa(c: int, n: int, va: int) -> int:
    """The physical address of ``va`` in an unmapped region: PA[35:29]
    (or PA[35:30] = PA >> 1 for the 1 GB regions CFG4 / CFG5) over the
    offset in the region; 32-bit physical addresses keep the low bits."""
    pa = (c >> 9) & 0x7F
    if n >= 4:                               # 1 GB regions
        return (((pa >> 1) << 30) | (va & 0x3FFFFFFF)) & M32
    return ((pa << 29) | (va & 0x1FFFFFFF)) & M32


def describe(n: int, c: int) -> str:
    pa, am, eu, cc = fields(c)
    return f"CFG{n} {AM_NAMES[am]} (PA field 0x{pa:02x}, C{cc}{', EU' if eu else ''})"
