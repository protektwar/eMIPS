"""Release 6 multi-threading: the virtual processors of a core (PLAN M22
phase 6f).

Sources: MIPS32 / MIPS64 Vol. II rev. 6.06 (DVP, EVP), Vol. III rev. 6.03
(9.5 VPControl, 9.8 the Global Number register and Table 9.13's VP
numbering, Config5.VP, EBase.CPUNum = VPNum with multi-threading), the
I6400 Programmer's Guide MD01196 rev. 1.00 (chapter 12: each VP is a
complete processor — GPRs, CP0, FPU, TLB mappings, interrupts and
exceptions, debug, reset —, a VTLB per VP and one FTLB, the caches shared;
11.1.1 / 11.4.2 SAARI per VP, SAAR per core; Table 7.3 four VP numbers a
core), Linux (``smp-cps.c``, ``cps-vec.S``: VPs started through the CPC).

A core's VPs are Machines (``Machine.vpcore`` = their :class:`VPCore`,
``Machine.vp_id`` = the VPId) on the core's caches; they share its FTLB
(``TLBMMU.ftlb_peers``), its SAAR and its ITU (emips.itu).  VPNum =
CoreNum x 4 + VPId (Vol. III 9.8's non-contiguous numbering with Max-VP 4,
the I6400's: GCR_SYS_CONFIG2.MAXVPW, the GIC's MAP_VP bits).  Starting /
stopping VPs is the CPC's (emips.cps: VP_RUN / VP_STOP / VP_RUNNING).

DVP stops the core's other VPs (they keep their state, take no interrupt
and fetch nothing) until the same VP's EVP; the eMIPS choices are in
``sim.DEVIATIONS["Release 6 VPs"]``.
"""

from __future__ import annotations

from . import cp0 as C

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
VPCONTROL, GLOBALNUMBER, SAARI, SAAR = (0, 4), (3, 1), (9, 6), (9, 7)
CONFIG5 = (16, 5)
#: VPControl.DIS (Table 9.4)
VPCTL_DIS = 1
#: Config5.VP (Table 9.72)
CFG5_VP = 1 << 7
#: the VP numbers of a core (Max-VP: the I6400's GIC maps four a core,
#: MD01196 Table 7.3; GCR_SYS_CONFIG2.MAXVPW)
MAX_VP = 4
#: what MD01196 configures (12.1: the micro-TLB sizes for 1, 2 and 4 VPs)
I6400_VPS = (1, 2, 4)
#: the registers the VPCore holds for its VPs (MFC0 / MTC0 go to it)
REGISTERS = frozenset({SAARI, SAAR})
#: SAARI.TARGET values (Table 11.2): 0 = the ITU, the others reserved
SAARI_ITU = 0


def vp_number(core: int, vp: int) -> int:
    """VPNum = CoreNum x Max-VP + VPId (Vol. III Table 9.13)."""
    return core * MAX_VP + vp


class VPCore:
    """The VPs of Release 6 core ``core``: their Machines (VP k =
    ``vps[k]``), the core's caches they share, its ITU and SAAR, and the
    VP whose DVP holds the others (``dis_by``)."""

    def __init__(self, core: int, vps: list, itu=None) -> None:
        from .itu import SAAR_SIZE
        self.core = core
        self.vps: list = []
        #: the shared caches (emips.cache.Caches) of the core, or None
        self.caches = vps[0].caches if vps else None
        #: the core's ITU (emips.itu.ITU) or None
        self.itu = itu
        #: SAAR (Figure 11.2): ADDR 43:13, SIZE 5:1 = 17 (128 KB, fixed), EN 0
        self.saar = SAAR_SIZE
        #: the VPId that executed a DVP the others wait on (None: all enabled)
        self.dis_by: int | None = None
        for k, m in enumerate(vps):
            self.attach(m, k)

    def attach(self, m, k: int) -> None:
        """VP ``k`` of the core = Machine ``m`` (a fresh one: power-up, a
        core reset, a VP started by the CPC): Config5.VP, GlobalNumber,
        VPControl, EBase.CPUNum = VPNum, SAARI / SAAR, the ITU, the
        shared FTLB."""
        if k < len(self.vps):
            old = self.vps[k]
            self.vps[k] = m
            if self.dis_by == k:            # its state is new: its DVP is gone
                self.dis_by = None
        else:
            old = None
            self.vps.append(m)
        m.vpcore, m.vp_id = self, k
        m._ext_regs, m._ext = REGISTERS, self
        cells = m.cp0
        cells[VPCONTROL] = {"name": "VPControl", "value": 0}
        cells[GLOBALNUMBER] = {"name": "GlobalNumber", "value": (self.core << 8) | k}
        cells[SAARI] = {"name": "SAARI", "value": 0}
        cells[SAAR] = {"name": "SAAR", "value": self.saar & (M64 if m.wide else M32)}
        for key in (VPCONTROL, GLOBALNUMBER):
            m._wmask[key] = 0                   # read-only (Tables 9.4 / 9.12)
        cells[CONFIG5]["value"] |= CFG5_VP
        m._wmask[CONFIG5] = m._wmask.get(CONFIG5, M32) & ~CFG5_VP
        ebase = cells[C.EBASE]["value"]
        cells[C.EBASE]["value"] = (ebase & ~0x3FF) | vp_number(self.core, k)
        if m.wide:
            m._wide_cp0 = m._wide_cp0 | {SAAR}
        m.itc = self.itu
        mmu = m.mmu
        if hasattr(mmu, "share_ftlb"):           # one FTLB for the core's VPs
            others = [v.mmu for v in self.vps if v is not m and v is not old]
            mmu.share_ftlb(others)

    @property
    def nvp(self) -> int:
        return len(self.vps)

    def vpnum(self, k: int) -> int:
        return vp_number(self.core, k)

    # -- the registers held here -----------------------------------------------------

    def read(self, m, key: tuple[int, int]) -> int:
        if key == SAAR:                         # SAARI.TARGET = 0: the ITU's SAAR
            return self.saar & (M64 if m.wide else M32)
        return m.cp0[key]["value"]

    def write(self, m, key: tuple[int, int], value: int) -> None:
        from .itu import SAAR_ADDR, SAAR_EN, SAAR_SIZE
        if key == SAARI:                        # writes of reserved targets are dropped
            if value & 0x3F == SAARI_ITU:
                m.cp0[SAARI]["value"] = SAARI_ITU
            return
        # SAAR (11.1.1.2): ADDR R/W, SIZE fixed (the ITU is 128 KB), EN only
        # with an ITU ("If the EN bit is not set, the ITU is not present")
        new = (value & SAAR_ADDR) | SAAR_SIZE
        if self.itu is not None:
            new |= value & SAAR_EN
        self.saar = new & (M64 if m.wide else M32)
        for v in self.vps:
            v.cp0[SAAR]["value"] = self.saar & (M64 if v.wide else M32)
        if self.itu is not None:
            self.itu.place(self.saar)

    # -- DVP / EVP (Vol. II 6.06) ------------------------------------------------------

    def disabled(self, k: int) -> bool:
        """VP ``k`` is held by another VP's DVP."""
        return self.dis_by is not None and self.dis_by != k

    def dvp(self, m) -> int:
        """DVP on VP ``m``: the old VPControl; with DIS = 0 the other VPs
        stop and DIS = 1 (with DIS = 1 nothing else happens: the pseudocode
        still returns VPControl, which the nesting of 6.06's programming
        notes relies on)."""
        old = m.cp0[VPCONTROL]["value"]
        if not old & VPCTL_DIS:
            if self.dis_by is None:
                self.dis_by = m.vp_id
            m.cp0[VPCONTROL]["value"] = old | VPCTL_DIS
        return old

    def evp(self, m) -> int:
        """EVP on VP ``m``: the old VPControl; with DIS = 1 the other VPs
        go on (as if the DVP had not been executed) and DIS = 0."""
        old = m.cp0[VPCONTROL]["value"]
        if old & VPCTL_DIS:
            if self.dis_by == m.vp_id:
                self.dis_by = None
            m.cp0[VPCONTROL]["value"] = old & ~VPCTL_DIS
        return old

    def lines(self) -> list[str]:
        held = (f"VP {self.vpnum(self.dis_by)}'s DVP holds the others"
                if self.dis_by is not None else "no DVP")
        return [f"core {self.core}: {self.nvp} VP(s) "
                f"{', '.join(str(self.vpnum(k)) for k in range(self.nvp))}; {held}; "
                f"SAAR 0x{self.saar:016x}"]
