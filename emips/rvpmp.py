"""RISC-V Physical Memory Protection (PLAN M25 phase 2c; The RISC-V
Instruction Set Manual, Volume II: Privileged Architecture 20250508 3.7).

:class:`PMP` holds a hart's PMP entries (``entries`` of the 64 the
architecture names: 0, 16 or 64; the generic hart has none unless asked
— then every access succeeds — and the Session's boot firmware sets the
entries up, as OpenSBI does) and their CSRs:

- **pmpcfg0-15** (RV32; on RV64 only the even ones exist), one byte per
  entry: R W X, A (OFF / TOR / NA4 / NAPOT), L.  A locked entry ignores
  writes, and a locked TOR entry also locks the previous entry's
  address; R = 0 with W = 1 is reserved — W is dropped (Spike).
  **pmpaddr0-63** hold bits 55:2 (RV64) / 33:2 (RV32) of an address;
  the granularity is 4 bytes (G = 0).  Entries above ``entries`` read 0.
- **The check** (3.7.1): the lowest-numbered entry that matches any byte
  of the access decides — all bytes must match it, else the access fails;
  its R / W / X permit the access for S / U-mode always and for M-mode
  when L is set (M-mode ignores unlocked entries).  No entry matches: an
  M-mode access succeeds, an S / U-mode one fails when any entry is
  implemented.  A failure is an access fault of the access's kind (1
  fetch, 5 load, 7 store / AMO); page-table walks are checked with
  S-mode's privilege (3.7.2).

**Smepmp** (LF8, priv. ch. 6; Spike's ``access_ok`` / ``pmp_ok``):
``mseccfg`` (RLB, MMWP, MML) — RLB lets locked entries be written; with
MML, L means M-mode-only, RW = 01 (and LRWX = 1111) a shared region (the
truth table of 6.2.1), an M-mode-only or locked shared rule with X cannot
be added unless RLB, and with MML or MMWP an M-mode access no entry matches
fails (MML: fetches only — MMWP: every access).

Choices: ``sim.DEVIATIONS["RISC-V"]``.
"""

from __future__ import annotations

M32 = 0xFFFFFFFF

R, W, X, L = 1, 2, 4, 0x80
A_SHIFT = 3
OFF, TOR, NA4, NAPOT = 0, 1, 2, 3


class PMP:
    """See the module docstring."""

    def __init__(self, xlen: int, entries: int = 0) -> None:
        if entries not in range(0, 65):
            raise ValueError(f"PMP entries must be 0-64, not {entries}")
        self.xlen = xlen
        self.entries = entries
        self.cfg = [0] * 64
        self.addr = [0] * 64
        #: counts the writes (what depends on the entries rebuilds: the
        #: P8700's PMAs, emips.p8700)
        self.version = 0
        #: pmpaddr's width: physical addresses of 56 (RV64) / 34 (RV32) bits
        self.addr_mask = (1 << 54) - 1 if xlen == 64 else M32
        #: Smepmp's mseccfg (emips.rvcsr keeps it here too): MML 1, MMWP 2, RLB 4
        self.mseccfg = 0

    # -- the CSRs -----------------------------------------------------------------

    def cfg_exists(self, n: int) -> bool:
        return self.xlen == 32 or n % 2 == 0

    def read_cfg(self, n: int) -> int:
        per = self.xlen // 8
        return sum(self.cfg[n * 4 + k] << (8 * k) for k in range(per))

    def write_cfg(self, n: int, value: int) -> None:
        self.version += 1
        rlb, mml = self.mseccfg & 4, self.mseccfg & 1
        for k in range(self.xlen // 8):
            i = n * 4 + k
            if i >= self.entries or (self.cfg[i] & L and not rlb):
                continue
            c = (value >> (8 * k)) & (R | W | X | 3 << A_SHIFT | L)
            if not mml and not c & R:
                c &= ~W                         # R = 0, W = 1: reserved (Spike) — shared with MML
            if not rlb and mml and c & L and (c & (R | W | X)) != R | W | X and \
                    (c & X or (c & (R | W)) == W):
                continue                        # (6.2.4b: no executable M-only / locked shared)
            self.cfg[i] = c

    def read_addr(self, i: int) -> int:
        return self.addr[i] if i < self.entries else 0

    def write_addr(self, i: int, value: int) -> None:
        self.version += 1
        rlb = self.mseccfg & 4
        if i >= self.entries or (self.cfg[i] & L and not rlb):
            return
        if i + 1 < self.entries and self.cfg[i + 1] & L and not rlb and \
                (self.cfg[i + 1] >> A_SHIFT) & 3 == TOR:
            return
        self.addr[i] = value & self.addr_mask

    # -- the check ------------------------------------------------------------------

    def _range(self, i: int) -> tuple[int, int] | None:
        """[lo, hi) of entry ``i``, None when it is OFF."""
        a = (self.cfg[i] >> A_SHIFT) & 3
        if a == OFF:
            return None
        if a == TOR:
            lo = self.addr[i - 1] << 2 if i else 0
            return lo, self.addr[i] << 2
        if a == NA4:
            return self.addr[i] << 2, (self.addr[i] << 2) + 4
        ones = (self.addr[i] ^ (self.addr[i] + 1)).bit_length() - 1    # trailing ones
        size = 1 << (ones + 3)
        base = (self.addr[i] & ~((1 << ones) - 1)) << 2
        return base & ~(size - 1), (base & ~(size - 1)) + size

    def allows(self, pa: int, size: int, kind: str, priv: int) -> bool:
        """3.7.1 (see the module docstring); ``kind``: fetch / load / store
        / hlvx (HLVX.HU / WU: both R and X, 21.3.1)."""
        if not self.entries:
            return True
        end = pa + size
        for i in range(self.entries):
            rng = self._range(i)
            if rng is None:
                continue
            lo, hi = rng
            if end <= lo or pa >= hi:
                continue                        # no byte matches
            if pa < lo or end > hi:
                return False                    # some but not all bytes
            c = self.cfg[i]
            if self.mseccfg & 1:                # Smepmp's MML: 6.2.1
                return _mml_ok(c, kind, priv == 3)
            if priv == 3 and not c & L:
                return True
            need = X if kind == "fetch" else W if kind == "store" else \
                R | X if kind == "hlvx" else R  # (HLVX: read and execute, priv. 21.3.1)
            return c & need == need
        if priv != 3 or self.mseccfg & 2:       # (MMWP: M-mode denied too)
            return False
        return not self.mseccfg & 1 or kind in ("load", "store")    # (MML: no M fetch)

    def locked(self) -> bool:
        """Some entry has L (mseccfg.RLB stays 0 then: 6.2.2b)."""
        return any(self.cfg[i] & L for i in range(self.entries))

    def lines(self) -> list[str]:
        """The configured entries (the views)."""
        out = []
        names = {OFF: "OFF", TOR: "TOR", NA4: "NA4", NAPOT: "NAPOT"}
        for i in range(self.entries):
            c = self.cfg[i]
            rng = self._range(i)
            if rng is None:
                continue
            perm = "".join(f if c & b else "-" for f, b in (("r", R), ("w", W), ("x", X)))
            out.append(f"pmp{i}: {names[(c >> A_SHIFT) & 3]:5} {perm} {'L' if c & L else ' '} "
                       f"0x{rng[0]:x}-0x{rng[1] - 1:x}")
        return out


def _mml_ok(c: int, kind: str, m_mode: bool) -> bool:
    """6.2.1's table (mseccfg.MML = 1), as Spike's ``access_ok``: L marks an
    M-mode-only rule, RW = 01 a shared region (X: S / U may write it; with L
    shared code, X: M may read it), LRWX = 1111 shared read-only data."""
    load, store, fetch = kind in ("load", "hlvx"), kind == "store", kind == "fetch"
    hlvx = kind == "hlvx"
    r, w, x, lk = c & R, c & W, c & X, c & L
    if r and w and x and lk:
        return load and not hlvx
    if not r and w:                             # a shared region
        return bool((not lk and x and ((load and not hlvx) or store))
                    or (not lk and not x and ((load and not hlvx) or (store and m_mode)))
                    or (lk and fetch) or (lk and load and x and m_mode))
    if m_mode != bool(lk):
        return False
    return bool((load and r and (not hlvx or x)) or (store and w) or (fetch and x))
