"""Colours of everything the GUI draws itself (PLAN M8 step 4): the editor
highlights and legend, the pipeline chart, the disassembly, the field
cells of the instruction panel and the tools, and the datapath canvas.

Two sets, :data:`LIGHT` and :data:`DARK`; :func:`colors` is the one in
use.  The window picks it from Configure > Theme — Light, Dark, or System
(dark when the desktop palette is dark) — with :func:`set_dark`, then
repaints.  Qt's own widgets follow the application palette instead.

Rule: a highlight colour is always paired with its text colour
(``highlight_text``) — light cells with dark text, dark cells with light
text — never the theme's text colour, which may not contrast.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Colors:
    dark: bool
    #: stage -> (fill: bands, chart cells, editor lines; accent: lit parts / wires)
    stage: dict[str, tuple[str, str]]
    current_line: str          # next instruction (editor, disassembly), changed registers
    error_line: str            # assembly error line, stall cell
    breakpoint: str            # breakpoint dot / text
    highlight_text: str        # text on any of the fills above and on field cells
    fields: tuple[str, ...]    # instruction bit-field cells
    red: str                   # forwarding paths, stalls, errors
    red_fill: str              # hazard unit while stalling
    # cache pane / datapath caches
    hit_fill: str              # the line a hit found
    miss_fill: str             # the line a miss filled
    valid_fill: str            # any valid line
    # datapath canvas
    canvas: str
    bubble: str                # stage band without an instruction
    idle_line: str             # unused wires / part outlines
    idle_text: str             # labels of unused parts, port names
    part_fill: str
    text: str                  # labels of used parts, values
    muted_text: str            # events line, pipeline-register names
    bar_fill: str              # pipeline registers
    bar_stroke: str
    value_bg: tuple[int, int, int, int]   # RGBA behind values on wires


LIGHT = Colors(
    dark=False,
    stage={"IF": ("#dbeafe", "#2563eb"), "ID": ("#dcfce7", "#16a34a"),
           "EX": ("#ffedd5", "#ea580c"), "MEM": ("#f3e8ff", "#9333ea"),
           "WB": ("#e5e7eb", "#4b5563")},
    current_line="#fbbf24", error_line="#fca5a5", breakpoint="#dc2626",
    highlight_text="#111827",
    fields=("#fde68a", "#bfdbfe", "#bbf7d0", "#fbcfe8", "#ddd6fe", "#fed7aa",
            "#a7f3d0", "#e5e7eb"),
    red="#dc2626", red_fill="#fee2e2",
    hit_fill="#bbf7d0", miss_fill="#fca5a5", valid_fill="#e2e8f0",
    canvas="#ffffff", bubble="#f8fafc", idle_line="#cbd5e1", idle_text="#64748b",
    part_fill="#ffffff", text="#0f172a", muted_text="#475569",
    bar_fill="#e2e8f0", bar_stroke="#475569", value_bg=(255, 255, 255, 225),
)

DARK = Colors(
    dark=True,
    stage={"IF": ("#1e3a5f", "#60a5fa"), "ID": ("#163d29", "#4ade80"),
           "EX": ("#4a2c14", "#fb923c"), "MEM": ("#3b2759", "#c084fc"),
           "WB": ("#3d434d", "#a3aab5")},
    current_line="#7a5a0c", error_line="#7f1d1d", breakpoint="#f87171",
    highlight_text="#f3f4f6",
    fields=("#5c4b13", "#1e3a5f", "#14532d", "#5b2140", "#3b2f63", "#5a3412",
            "#134e3a", "#374151"),
    red="#f87171", red_fill="#4c1d1d",
    hit_fill="#14532d", miss_fill="#7f1d1d", valid_fill="#323844",
    canvas="#1b1d22", bubble="#202329", idle_line="#3f4651", idle_text="#8b93a1",
    part_fill="#23262d", text="#e5e7eb", muted_text="#a3aab5",
    bar_fill="#2f3540", bar_stroke="#8b93a1", value_bg=(27, 29, 34, 225),
)

#: the stages of the 24K pipeline model in the colours of the textbook
#: stage doing the same work (fetch, decode / register read, execute,
#: memory, completion)
STAGE_ALIASES = {"IS": "IF", "IT": "IF", "RF": "ID", "AG": "EX", "MS": "MEM", "ER": "WB"}
for _set in (LIGHT, DARK):
    for _name, _like in STAGE_ALIASES.items():
        _set.stage.setdefault(_name, _set.stage[_like])

_current = LIGHT


def colors() -> Colors:
    """The colour set in use."""
    return _current


#: the P8700 pipeline's stages and chart cells (emips.pipe8700) in the
#: textbook stage colours
STAGE_ALIAS = {"DR": "ID", "IS": "EX", "co": "MEM", "GR": "WB", "rp": "ID"}


def stage_colours(st: str) -> tuple[str, str] | None:
    """(fill, stroke) of stage / chart cell ``st``, or None."""
    stage = _current.stage
    return stage.get(st) or stage.get(STAGE_ALIAS.get(st, ""))


def set_dark(dark: bool) -> bool:
    """Use the dark (or light) set; True if that changed anything."""
    global _current
    new = DARK if dark else LIGHT
    changed = new is not _current
    _current = new
    return changed


def palette_is_dark(palette) -> bool:
    """Is a ``QPalette`` dark (its window background)?"""
    from .qt import QtGui
    return palette.color(QtGui.QPalette.ColorRole.Window).lightness() < 128
