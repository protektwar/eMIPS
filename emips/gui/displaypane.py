"""The Display pane (PLAN M27 phase S2): the bitmap display — MARS's
Bitmap Display tool — drawn from :meth:`Session.display_frame` (a scan of
physical memory below the caches, no side effects): each memory word a
unit-size block, scaled to fit with its proportions kept; the word under
the mouse (its column / row, address, physical address, value); Save
Image writes the PNG :meth:`emips.display.Frame.png` gives.  Without a
display (Settings > Exceptions & I/O > Bitmap display, ``--display``) it
says how to switch one on."""

from __future__ import annotations

from ..session import Session
from . import theme
from .i18n import tr
from .qt import QtCore, QtGui, QtWidgets, Signal
from .widgets import mono_font

Qt = QtCore.Qt


class _Canvas(QtWidgets.QWidget):
    """The picture: a QImage of the frame (one pixel a word), drawn with
    each word as its unit-size block, scaled to fit (nearest neighbour)."""

    hovered = Signal(int, int)                 # column, row (-1: outside)

    def __init__(self) -> None:
        super().__init__()
        self.image: QtGui.QImage | None = None
        self.units = (1, 1)
        self.setMouseTracking(True)
        self.setMinimumSize(64, 64)

    def set_frame(self, frame) -> None:
        if frame is None:
            self.image = None
        else:
            sp = frame.spec
            self.image = QtGui.QImage(frame.rgb, sp.columns, sp.rows, 3 * sp.columns,
                                      QtGui.QImage.Format.Format_RGB888).copy()
            self.units = (sp.unit_w, sp.unit_h)
        self.update()

    def _target(self) -> QtCore.QRectF | None:
        if self.image is None:
            return None
        w = self.image.width() * self.units[0]
        h = self.image.height() * self.units[1]
        k = min(self.width() / w, self.height() / h)
        if k >= 1:
            k = float(int(k))                    # whole multiples: crisp pixels
        tw, th = w * k, h * k
        return QtCore.QRectF((self.width() - tw) / 2, (self.height() - th) / 2, tw, th)

    def paintEvent(self, event) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(theme.colors().canvas))
        r = self._target()
        if r is not None:
            p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
            p.drawImage(r, self.image)
            p.setPen(QtGui.QColor(theme.colors().idle_line))
            p.drawRect(r.adjusted(-1, -1, 0, 0))
        p.end()

    def cell_at(self, pos) -> tuple[int, int]:
        r = self._target()
        if r is None or not r.contains(QtCore.QPointF(pos)):
            return -1, -1
        col = int((pos.x() - r.x()) / r.width() * self.image.width())
        row = int((pos.y() - r.y()) / r.height() * self.image.height())
        return (min(col, self.image.width() - 1), min(row, self.image.height() - 1))

    def mouseMoveEvent(self, event) -> None:
        pos = event.position() if hasattr(event, "position") else event.pos()
        self.hovered.emit(*self.cell_at(pos.toPoint() if hasattr(pos, "toPoint") else pos))

    def leaveEvent(self, event) -> None:
        self.hovered.emit(-1, -1)


class DisplayView(QtWidgets.QWidget):
    #: Save Image (the window asks for a file name)
    save_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._session: Session | None = None
        lay = QtWidgets.QVBoxLayout(self)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self.canvas = _Canvas()
        self.canvas.hovered.connect(self._hover)
        lay.addWidget(self.canvas, 1)
        row = QtWidgets.QHBoxLayout()
        self.info = QtWidgets.QLabel()
        self.info.setFont(mono_font())
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.info, 1)
        self.save = QtWidgets.QPushButton(tr("Save Image…"))
        self.save.setToolTip(tr("Write the display as a PNG image"))
        self.save.clicked.connect(self.save_requested)
        row.addWidget(self.save)
        lay.addLayout(row)
        self.frame = None

    def refresh(self, s: Session) -> None:
        self._session = s
        frame = s.display_frame() if s.machine is not None else None
        self.frame = frame
        self.canvas.set_frame(frame)
        self.save.setEnabled(frame is not None)
        if frame is None:
            self.note.setText(tr("No bitmap display. Switch one on in Configure > Settings > "
                                 "Exceptions & I/O > Bitmap display (or --display on the "
                                 "command line): memory drawn as pixels, one word "
                                 "0x00RRGGBB a pixel, as MARS's Bitmap Display."))
            self.info.setText("")
            return
        sp = frame.spec
        dev = s.display_device()
        state = (tr("off (CONTROL.ON = 0)") if not frame.on else
                 tr("{n} words written").format(n=frame.written)
                 + (" — " + tr("{n} on pages nothing maps (black)")
                    .format(n=frame.unmapped) if frame.unmapped else ""))
        frames = tr("frame {n}, showing 0x{base:08x}").format(n=dev.frame, base=dev.front) \
            if dev is not None and (dev.frame or dev.front != sp.base) else ""
        self.note.setText(" — ".join(x for x in (sp.describe(), state, frames) if x))
        if not self.info.text():
            self.info.setText(tr("Point at a pixel: its address and value"))

    def _hover(self, col: int, row: int) -> None:
        s = self._session
        if s is None or self.frame is None or col < 0:
            return
        va, pa, v = s.display_word(col, row)
        where = f"PA 0x{pa:08x}" if pa is not None else tr("no physical address")
        self.info.setText(f"({col}, {row})  0x{va:08x}  {where}  0x{v:08x}  "
                          f"RGB {v >> 16 & 0xFF}, {v >> 8 & 0xFF}, {v & 0xFF}")
