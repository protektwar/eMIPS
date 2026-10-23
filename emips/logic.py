"""Two-level logic: programmable logic arrays and their minimization
(Quine-McCluskey).  Qt-free.

A :class:`PLA` is what Patterson & Hennessy draw for the MIPS control
(Appendix C/D, "the PLA implementation of the main control"): an **AND
plane** — each *product term* is the AND of some inputs, true or
inverted — and an **OR plane** — each output is the OR of some product
terms.  :func:`minimize` turns a truth table (the cases that must give 1,
those that must give 0, the rest "don't care") into few product terms:
the prime implicants of Quine-McCluskey and a greedy cover of the ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

#: a product term: (value, mask) — for each input bit i in ``mask`` the
#: term needs input i == bit i of ``value``; inputs outside ``mask`` are
#: not connected (don't care)
Term = tuple[int, int]


def _covers(term: Term, x: int) -> bool:
    value, mask = term
    return x & mask == value


def minimize(ones: set[int], zeros: set[int], n: int) -> list[Term]:
    """Product terms over ``n`` inputs that are 1 on every case in
    ``ones`` and 0 on every case in ``zeros`` (anything else: don't care):
    Quine-McCluskey prime implicants (merging over the don't cares too),
    the essential ones first, then a greedy cover."""
    if not ones:
        return []
    full = (1 << n) - 1
    if not zeros:
        return [(0, 0)]                            # always 1
    # start from the cubes that touch no zero: grow every one maximally
    # (cheaper than enumerating the don't-care space for 10+ inputs)
    zero_list = list(zeros)
    memo: dict[Term, bool] = {}

    def ok(term: Term) -> bool:
        """The cube touches no zero (its own points when it is the
        smaller set to look at)."""
        hit = memo.get(term)
        if hit is not None:
            return hit
        value, mask = term
        free = [1 << i for i in range(n) if not mask >> i & 1]
        if (1 << len(free)) < len(zero_list):
            hit = True
            for k in range(1 << len(free)):
                x = value
                for j, b in enumerate(free):
                    if k >> j & 1:
                        x |= b
                if x in zeros:
                    hit = False
                    break
        else:
            hit = not any(z & mask == value for z in zero_list)
        memo[term] = hit
        return hit

    primes: set[Term] = set()
    for x in ones:
        term = (x, full)
        changed = True
        while changed:                             # drop literals while legal
            changed = False
            for i in range(n - 1, -1, -1):
                bit = 1 << i
                if term[1] & bit:
                    t = (term[0] & ~bit, term[1] & ~bit)
                    if ok(t):
                        term, changed = t, True
        primes.add(term)
    # QM-style merge pass on the found implicants (two cubes differing in
    # one cared bit merge into a larger one when it stays legal)
    grew = True
    while grew:
        grew = False
        for a, b in combinations(list(primes), 2):
            if a[1] != b[1]:
                continue
            diff = a[0] ^ b[0]
            if diff and diff & (diff - 1) == 0:
                t = (a[0] & ~diff, a[1] & ~diff)
                if ok(t) and t not in primes:
                    primes.add(t)
                    grew = True
        # keep only primes not contained in a larger one
        primes = {p for p in primes if not any(
            q != p and q[1] & p[1] == q[1] and p[0] & q[1] == q[0] for q in primes)}
    # cover: essential implicants, then the one covering most of the rest
    left = set(ones)
    chosen: list[Term] = []
    plist = sorted(primes, key=lambda t: (bin(t[1]).count("1"), t))
    for x in sorted(ones):
        cov = [p for p in plist if _covers(p, x)]
        if len(cov) == 1 and cov[0] not in chosen:
            chosen.append(cov[0])
    for p in chosen:
        left -= {x for x in left if _covers(p, x)}
    while left:
        best = max(plist, key=lambda p: (sum(_covers(p, x) for x in left),
                                         -bin(p[1]).count("1")))
        chosen.append(best)
        left -= {x for x in left if _covers(best, x)}
    return chosen


@dataclass
class PLA:
    """``inputs`` (names, input 0 = the most significant: the leftmost
    column), product ``terms`` and, per output, the terms it ORs."""
    name: str
    inputs: list[str]
    outputs: list[str]
    terms: list[Term] = field(default_factory=list)
    #: output name -> indices of its terms
    uses: dict[str, list[int]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.inputs)

    def literal(self, term: Term, i: int) -> int | None:
        """Input ``i``'s literal in a term: 1 (true), 0 (inverted) or None."""
        bit = 1 << (self.n - 1 - i)
        if not term[1] & bit:
            return None
        return 1 if term[0] & bit else 0

    def evaluate(self, x: int) -> tuple[list[bool], dict[str, bool]]:
        """``(product terms, outputs)`` for the input vector ``x`` (input 0 =
        the most significant bit)."""
        hit = [_covers(t, x) for t in self.terms]
        return hit, {o: any(hit[i] for i in self.uses.get(o, [])) for o in self.outputs}

    def describe_term(self, k: int) -> str:
        t = self.terms[k]
        lits = []
        for i, name in enumerate(self.inputs):
            v = self.literal(t, i)
            if v is not None:
                lits.append(name if v else f"~{name}")
        return " AND ".join(lits) or "1"


def build(name: str, inputs: list[str], outputs: list[str],
          table: dict[int, dict[str, bool]]) -> PLA:
    """A PLA from ``table`` (input vector -> the outputs that matter for
    it; vectors not in the table, and outputs missing from a row, are
    don't cares).  Terms common to several outputs are shared."""
    pla = PLA(name, inputs, outputs)
    index: dict[Term, int] = {}
    for o in outputs:
        ones = {x for x, row in table.items() if row.get(o) is True}
        zeros = {x for x, row in table.items() if row.get(o) is False}
        for t in minimize(ones, zeros, len(inputs)):
            if t not in index:
                index[t] = len(pla.terms)
                pla.terms.append(t)
            pla.uses.setdefault(o, []).append(index[t])
    return pla


# -- cubes: truth tables too large to list point by point -------------------------

def intersects(a: Term, b: Term) -> bool:
    """Do two cubes share a point?"""
    return not (a[0] ^ b[0]) & a[1] & b[1]


def contains(a: Term, b: Term) -> bool:
    """Is cube ``b`` inside cube ``a``?"""
    return a[1] & b[1] == a[1] and not (a[0] ^ b[0]) & a[1]


def sharp(a: Term, b: Term) -> list[Term]:
    """Cube ``a`` minus cube ``b``, as disjoint cubes."""
    if not intersects(a, b):
        return [a]
    out: list[Term] = []
    value, mask = a
    rest = b[1] & ~mask
    while rest:
        bit = rest & -rest
        rest &= rest - 1
        out.append(((value & ~bit) | (~b[0] & bit), mask | bit))
        value, mask = (value & ~bit) | (b[0] & bit), mask | bit
    return out                          # (what is left lies inside b)


def minimize_cubes(ones: list[Term], zeros: list[Term], n: int) -> list[Term]:
    """Product terms over ``n`` inputs covering every cube in ``ones`` and
    touching no cube in ``zeros`` (the rest: don't care) — :func:`minimize`
    for a table given as cubes: each one-cube grows (literals dropped, the
    most significant input first) while it touches no zero-cube, then a
    greedy cover of the one-cubes."""
    if not ones:
        return []
    grown: list[Term] = []
    for value, mask in ones:
        value &= mask
        for i in range(n - 1, -1, -1):
            bit = 1 << i
            if mask & bit:
                t = (value & ~bit, mask & ~bit)
                if not any(intersects(t, z) for z in zeros):
                    value, mask = t
        if (value, mask) not in grown:
            grown.append((value, mask))
    left = list(range(len(ones)))
    chosen: list[Term] = []
    while left:
        best = max(grown, key=lambda g: (sum(contains(g, ones[k]) for k in left),
                                         -bin(g[1]).count("1")))
        chosen.append(best)
        left = [k for k in left if not contains(best, ones[k])]
    return chosen


def build_cubes(name: str, inputs: list[str], outputs: list[str],
                table: list[tuple[Term, dict[str, bool]]]) -> PLA:
    """:func:`build` from cubes: ``table`` lists (cube, the outputs that
    matter there); points in no cube are don't cares."""
    pla = PLA(name, inputs, outputs)
    index: dict[Term, int] = {}
    for o in outputs:
        ones = [c for c, row in table if row.get(o) is True]
        zeros = [c for c, row in table if row.get(o) is False]
        for t in minimize_cubes(ones, zeros, len(inputs)):
            if t not in index:
                index[t] = len(pla.terms)
                pla.terms.append(t)
            pla.uses.setdefault(o, []).append(index[t])
    return pla
