"""The TLB pane (PLAN M9 phase 8): the TLB entries (or the BAT entries,
the I7200 MPU's segments and regions),
the entry the last translation used (green: translated, red: TLB
exception), how its virtual address was cut (VPN2 | even/odd | offset)
and became a physical one (PFN | offset), the ITLB / DTLB copies, the
registers a refill handler uses and the statistics.

Everything comes from :mod:`emips.tlbview` (side-effect free reads of the
entries and counters) and :meth:`Session.tlb_accesses` (the translations
recorded in the last step)."""

from __future__ import annotations

import html

from .. import tlbview as tv
from ..session import Session
from . import theme
from .i18n import N_, tr
from .qt import QtCore, QtWidgets
from .widgets import _set, mono_font

Qt = QtCore.Qt
T = theme.colors

#: words from emips.tlbview shown in the pane
WORDS = {"never written": N_("never written"), "invalidated (EHINV)": N_("invalidated (EHINV)"),
         "VPN2": N_("VPN2"), "even/odd": N_("even/odd"), "offset": N_("offset"),
         "even": N_("even"), "odd": N_("odd")}


def word(text: str) -> str:
    return tr(WORDS.get(text, text))


class TLBView(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._session: Session | None = None
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.everything = QtWidgets.QCheckBox(tr("all entries"))
        self.everything.setToolTip(tr("List every entry, also those never written or "
                                      "invalidated; otherwise the entries in use, the wired "
                                      "ones and the one Random points to"))
        self.reset = QtWidgets.QPushButton(tr("Reset statistics"))
        bar.addWidget(self.everything)
        bar.addStretch(1)
        bar.addWidget(self.reset)
        lay.addLayout(bar)
        self.address = QtWidgets.QLabel()
        self.address.setTextFormat(Qt.TextFormat.RichText)
        self.address.setWordWrap(True)
        lay.addWidget(self.address)
        split = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        self.table = self._table()
        self.micro = self._table()
        self.stats = QtWidgets.QPlainTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setFont(mono_font())
        for w in (self.table, self.micro, self.stats):
            split.addWidget(w)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 2)
        split.setStretchFactor(2, 2)
        lay.addWidget(split, 1)
        self.everything.toggled.connect(lambda _: self._again())
        self.reset.clicked.connect(self._reset_stats)

    @staticmethod
    def _table() -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(0, 0)
        t.setFont(mono_font())
        t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        t.verticalHeader().setDefaultSectionSize(20)
        t.verticalHeader().setVisible(False)
        return t

    def _again(self) -> None:
        if self._session is not None:
            self.refresh(self._session)

    def _reset_stats(self) -> None:
        s = self._session
        if s is not None and s.machine is not None:
            tv.reset_stats(s.machine)
            self.refresh(s)

    # -- refresh ------------------------------------------------------------------------

    def refresh(self, s: Session) -> None:
        self._session = s
        m = s.machine
        if m is not None and m.rv:              # a RISC-V hart: satp, its TLB
            self._rv(m)
            return
        if m is None or tv.kind(m) == "fixed":
            for t in (self.table, self.micro):
                t.setRowCount(0)
                t.setColumnCount(0)
            self.address.setText(tr("No TLB: the fixed mapping MMU translates by fixed "
                                    "rules. Pick a TLB in Configure > Settings > Processor "
                                    "> MMU (the program restarts).") if m is not None else "")
            self.stats.setPlainText("")
            self.reset.setEnabled(False)
            return
        self.reset.setEnabled(True)
        if tv.kind(m) == "mpu":                 # the I7200's MPU: segments, regions
            self.address.setText("<b>" + "<br>".join(html.escape(x.strip())
                                                     for x in m.mmu.lines()[:3]) + "</b>")
            self._mpu(m)
            self.micro.setRowCount(0)
            self.micro.setColumnCount(0)
            self.micro.setVisible(False)
            self.stats.setPlainText("\n".join(tv.stats_lines(m)))
            return
        acc = tv.last_access(s.tlb_accesses())
        self.address.setText(self._address_html(m, acc, s.tlb_accesses()))
        if tv.kind(m) == "bat":
            self._bat(m, acc)
            self.micro.setRowCount(0)
            self.micro.setColumnCount(0)
            self.micro.setVisible(False)
        else:
            self.micro.setVisible(True)
            self._entries(m, acc)
            self._micro(m, acc)
        text = tv.stats_lines(m) + [""] + [f"{n:<9} 0x{v:08x}" for n, v in tv.registers(m)]
        self.stats.setPlainText("\n".join(text))

    def _mark(self, acc, entry: int) -> tuple[str | None, bool]:
        """Background and bold of the row of ``entry``."""
        colors = T()
        if acc is not None and acc.entry == entry:
            return (colors.miss_fill if acc.fault else colors.hit_fill), True
        return None, False

    def _entries(self, m, acc) -> None:
        mmu = m.mmu
        heads = [tr("entry"), "", tr("VA (page pair)"), tr("owner"), tr("page"),
                 tr("even page (EntryLo0)"), tr("odd page (EntryLo1)")]
        rows = tv.rows(mmu, m, everything=self.everything.isChecked())
        if acc is not None and acc.entry is not None and acc.entry not in {r.index for r in rows}:
            rows = sorted(rows + [tv.entry_row(mmu, m, acc.entry)], key=lambda r: r.index)
        t = self.table
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setRowCount(len(rows))
        colors = T()
        focus = None
        for i, r in enumerate(rows):
            bg, bold = self._mark(acc, r.index)
            if bg is None and r.used:
                bg = colors.valid_fill
            if bold:
                focus = i
            marks = ("W" if r.wired else "") + ("R" if r.random else "")
            name = str(r.index) if not mmu.ftlb else (
                f"V{r.index}" if r.index < mmu.n else
                "F{s}.{w}".format(s=(r.index - mmu.n) % mmu.fsets, w=(r.index - mmu.n) // mmu.fsets))
            if r.used:
                owner = "G" if r.g else f"ASID {r.asid}"
                end = r.va + r.span - 1             # (MIPS64 with 64-bit addresses: wider)
                cells = [name, marks, f"0x{r.va:08x}-0x{end & (0xFFFFFFFF if r.va >> 32 == 0 else (1 << 64) - 1):08x}",
                         owner, tv.size_name(r.page), tv.half_text(r, 0), tv.half_text(r, 1)]
            else:
                cells = [name, marks, word(r.state), "", "", "", ""]
            for c, text in enumerate(cells):
                half_used = (bold and acc is not None and c == 5 + acc.half)
                _set(t, i, c, text, bg, bold and (c < 5 or half_used))
            t.item(i, 0).setToolTip(tv.format_entry(m, r.index))
            t.item(i, 1).setToolTip(tr("W = wired (TLBWR never replaces it), R = Random "
                                       "(the entry TLBWR writes next)"))
        t.resizeColumnsToContents()
        if focus is not None:
            t.scrollToItem(t.item(focus, 0))

    def _micro(self, m, acc) -> None:
        mmu = m.mmu
        heads = [tr("micro-TLB"), tr("slot"), tr("VA"), tr("size"), tr("owner"), tr("PA"),
                 tr("from entry")]
        t = self.micro
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        slots = [(u, i, e) for u in (mmu.itlb, mmu.dtlb) for i, e in enumerate(u.slots)]
        t.setRowCount(len(slots))
        colors = T()
        for r, (u, i, e) in enumerate(slots):
            victim = tv.micro_victim(u) == i
            bg = None
            bold = False
            if (acc is not None and e is not None and acc.micro == u.name
                    and e.matches(acc.va, acc.asid)):
                bg, bold = colors.hit_fill if acc.micro_hit else colors.miss_fill, True
            elif e is not None:
                bg = colors.valid_fill
            if e is None:
                cells = [u.name, str(i) + (" *" if victim else ""), "-", "", "", "", ""]
            else:
                cells = [u.name, str(i) + (" *" if victim else ""), f"0x{e.vbase:08x}",
                         tv.size_name(e.size), "G" if e.g else f"ASID {e.asid}",
                         f"0x{e.pbase:08x} C{e.cca}", str(e.index)]
            for c, text in enumerate(cells):
                _set(t, r, c, text, bg, bold)
            t.item(r, 1).setToolTip(tr("* = replaced by the next miss of this micro-TLB"))
        t.resizeColumnsToContents()

    def _rv(self, m) -> None:
        """A RISC-V hart (priv. ch. 12): satp's scheme / ASID / root, the
        entries its TLB caches (the leaf PTEs a walk found; SFENCE.VMA
        flushes them) and the counters."""
        mmu = m.mmu
        self.micro.setRowCount(0)
        self.micro.setColumnCount(0)
        self.micro.setVisible(False)
        t = self.table
        if not hasattr(mmu, "tlb"):             # (M-only / M + U: no paging)
            t.setRowCount(0)
            t.setColumnCount(0)
            self.address.setText(tr("No paging: this hart has no S-mode (Settings > "
                                    "Processor > RISC-V privilege modes): PA = VA, "
                                    "PMP checks the accesses."))
            self.stats.setPlainText("")
            self.reset.setEnabled(False)
            return
        self.reset.setEnabled(False)
        scheme, asid, root = mmu._satp(m)
        self.address.setText(html.escape(
            tr("satp: {mode}, ASID {asid}, root page table at PA 0x{root:x} — M-mode "
               "and Bare translate PA = VA").format(
                mode=scheme.name if scheme is not None else "Bare", asid=asid,
                root=root << 12)))
        if hasattr(mmu, "rows"):                # the P8700's VTLB / FTLB (MD01502 ch. 2)
            self._p8700(m, mmu)
            return
        from ..rvmmu import entry_rows
        h = m.csrs.has_h
        if h:                                   # (priv. ch. 21: V, hgatp, vsatp)
            g, vmid, groot = mmu._hgatp(m)
            vs, vasid, vroot = mmu._atp(m, m.csrs.vsatp)
            self.address.setText(self.address.text() + "<br>" + html.escape(
                tr("{mode}-mode, V = {v}; hgatp: {g}, VMID {vmid}; vsatp: {vs}, ASID "
                   "{vasid} — a guest access translates twice (VS-stage, G-stage)").format(
                    mode=m.mode_name, v=m.v, g=g.name + "x4" if g else "Bare", vmid=vmid,
                    vs=vs.name if vs else "Bare", vasid=vasid)))
        heads = [tr("ASID"), tr("page"), tr("size"), tr("PA"), tr("PTE flags")]
        if h:
            heads = [tr("stage"), tr("VMID")] + heads[:3] + [tr("GPA"), tr("PA"),
                                                             tr("VS / PTE flags"),
                                                             tr("G-stage flags")]
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        entries = entry_rows(m)
        t.setRowCount(len(entries))
        for i, r in enumerate(entries):         # (most recent first)
            cells = list(r) if h else [r[2], r[3], r[4], r[6], r[7]]
            for c, text in enumerate(cells):
                _set(t, i, c, text, T().valid_fill if i == 0 else None, False)
        t.resizeColumnsToContents()
        self.stats.setPlainText(
            f"TLB: {len(entries)} of {mmu.entries} entries\n"
            f"hits {mmu.hits}   misses {mmu.misses}   page-table walks {mmu.walks}")

    def _p8700(self, m, mmu) -> None:
        t = self.table
        heads = [tr("entry"), tr("ASID"), tr("VPN"), tr("page"), tr("PPN"), tr("PTE flags")]
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        rows = mmu.rows(m)
        t.setRowCount(len(rows))
        names = "VRWXUGAD"
        for i, (where, asid, vpn, size, ppn, pte, leaf) in enumerate(rows):
            flags = "".join(c if pte >> k & 1 else "-" for k, c in enumerate(names))
            cells = [where, asid, f"0x{vpn:x}", size if leaf else tr("{s} (non-leaf)").format(
                s=size), f"0x{ppn:x}", flags]
            for c, text in enumerate(cells):
                _set(t, i, c, text, None, False)
        t.resizeColumnsToContents()
        self.stats.setPlainText("\n".join([mmu.lines(m)[0], mmu.lines(m)[-1].strip()]))

    def _bat(self, m, acc) -> None:
        mmu = m.mmu
        heads = [tr("entry"), tr("use"), tr("bounds (last VA)"), tr("base"), tr("attributes")]
        t = self.table
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setRowCount(len(mmu.entries))
        colors = T()
        for i, e in enumerate(mmu.entries):
            bg, bold = self._mark(acc, i)
            if bg is None and e.lo & tv.LO_V:
                bg = colors.valid_fill
            cells = [str(i), mmu.entry_how(i), f"0x{e.bounds << 12 | 0xFFF:08x}",
                     f"0x{e.pfn << 12:08x}", f"C{(e.lo >> 3) & 7} {tv.lo_flags(e.lo)}"]
            for c, text in enumerate(cells):
                _set(t, i, c, text, bg, bold)
        t.resizeColumnsToContents()

    def _mpu(self, m) -> None:
        """The MPU build's 16 segments, then every enabled region (a higher
        region wins where they overlap; the FMT build: regions only)."""
        mpu = m.mmu
        heads = [tr("segment / region"), tr("addresses"), tr("CCA"), tr("inhibits")]
        rows = []
        if mpu.segments:
            for k in range(16):
                rows.append((tr("segment {n}").format(n=k),
                             f"0x{k << 28:08x}-0x{(k << 28) | 0x0FFFFFFF:08x}", mpu.seg[k], None))
        for r in range(mpu.n):
            span = mpu.span(r)
            if span is not None:
                rows.append((tr("region {n}").format(n=r), f"0x{span[0]:08x}-0x{span[1] - 1:08x}",
                             mpu.ctl[r] & 0x3F, T().valid_fill))
        t = self.table
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setRowCount(len(rows))
        for i, (name, span, bits, bg) in enumerate(rows):
            for c, text in enumerate((name, span, f"C{bits & 7}", tv.mpu_flags(bits))):
                _set(t, i, c, text, bg, False)
        t.resizeColumnsToContents()

    # -- the last translation -----------------------------------------------------------

    @staticmethod
    def _address_html(m, acc, accesses) -> str:
        e = html.escape
        head = e(f"{m.mmu.name}")
        ops = [a for _w, a in accesses if a.note]
        tail = "".join(f"<br>{e(a.note)}" for a in ops)
        if acc is None:
            return (f"<b>{head}</b><br>" + e(tr("no translation through the TLB in the last "
                                                 "step (unmapped addresses, e.g. kseg0, are "
                                                 "not translated)")) + tail)
        text = f"<b>{head}</b><br>{e(acc.describe())}"
        if acc.page and acc.entry is not None:
            colors = T()
            sp = tv.VASplit(acc.va, acc.page)
            cells = []
            for i, (name, msb, lsb, value) in enumerate(sp.fields()):
                rng = f"VA[{msb}:{lsb}]" if msb != lsb else f"VA[{msb}]"
                cells.append(
                    f"<td bgcolor='{colors.fields[i]}' style='padding:2px 6px'>"
                    f"<span style='color:{colors.highlight_text}'><b>{e(word(name))}</b> "
                    f"{rng}<br><tt>{tv.bits(acc.va, msb, lsb)}</tt><br>0x{value:x}</span></td>")
            text += f"<table cellspacing=2><tr>{''.join(cells)}</tr></table>"
            if acc.pa is not None:
                text += e(tr("{entry}: {half} page -> PFN 0x{pfn:x}; PA = 0x{frame:08x} + "
                             "offset = 0x{pa:08x}").format(
                    entry=acc.name, half=word(tv.HALF[acc.half]), pfn=acc.pfn,
                    frame=acc.frame, pa=acc.pa))
        return text + tail
