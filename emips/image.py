"""Drawing helpers — Pillow port of ``BotImage::ImageMe`` (``lib/eMIPSbotImage.rb``).

The original rendered with RMagick/ImageMagick.  This keeps the same
conceptual API (gravity-anchored text with per-gravity shadow offsets,
grid, watermark, border, ``.jpg`` output) using Pillow.  Rendered with
the same calls as the ``ready/*.py`` scripts it is a *visual* port, not a
pixel-identical one.  Deliberate approximations (all calibrated against
the pre-port JPGs shipped in ``ready/``):

* ImageMagick rasterised 1px grid strokes with heavy antialiasing, so
  ``"grey"`` lines measured ~#E5 and doubled field borders ~#5B.  Pillow
  strokes are crisp, so ``make_grid`` composites its 1px stroke (and any
  solid fill) at a fixed ~25% alpha to look equivalent.
* The original ``fontBold=1`` was a same-colour stroke of width 0.5;
  Pillow's minimum stroke width is 1px.
* Gravities with no shadow delta in the original (Center, East, South,
  SouthEast) left an unset Ruby local and raised; the port uses a (0, 0)
  delta for them instead of crashing.
* Cell backgrounds of ``""`` / ``"transparent"`` are drawn without fill
  (as they appeared in the shipped diagrams).
* The watermark used ImageMagick's ``Overlay`` composite; this uses a
  plain alpha composite at the given opacity.  In the shipped diagrams
  the watermark band is fully covered by content either way.
* ImageMagick rotates clockwise for positive angles; Pillow rotates
  counter-clockwise, so the angle is negated to match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

# ---------------------------------------------------------------------------
# Gravity constants (the original scripts passed ``Magick::*Gravity``)
# ---------------------------------------------------------------------------

FORGET_GRAVITY = "forget"
NORTH_WEST_GRAVITY = "north_west"
NORTH_GRAVITY = "north"
NORTH_EAST_GRAVITY = "north_east"
WEST_GRAVITY = "west"
CENTER_GRAVITY = "center"
EAST_GRAVITY = "east"
SOUTH_WEST_GRAVITY = "south_west"
SOUTH_GRAVITY = "south"
SOUTH_EAST_GRAVITY = "south_east"

# Pillow text anchors / per-line alignment for each gravity.
_ANCHORS = {
    FORGET_GRAVITY: "la",
    NORTH_WEST_GRAVITY: "la",
    NORTH_GRAVITY: "ma",
    NORTH_EAST_GRAVITY: "ra",
    WEST_GRAVITY: "lm",
    CENTER_GRAVITY: "mm",
    EAST_GRAVITY: "rm",
    SOUTH_WEST_GRAVITY: "ld",
    SOUTH_GRAVITY: "md",
    SOUTH_EAST_GRAVITY: "rd",
}
_ALIGNS = {"l": "left", "m": "center", "r": "right"}

# Shadow offsets per gravity, exactly as hard-coded in the original
# ``addTextAt``.  Center/East/South/SouthEast were unset there (a latent
# crash); use (0, 0) instead of raising.
_SHADOW_DELTAS = {
    FORGET_GRAVITY: (1, 1),
    NORTH_WEST_GRAVITY: (1, 1),
    NORTH_GRAVITY: (2, 2),
    NORTH_EAST_GRAVITY: (-2, -2),
    WEST_GRAVITY: (-2, -2),
    SOUTH_WEST_GRAVITY: (-2, -2),
}


def _anchor_point(gravity: str, width: int, height: int):
    """ImageMagick gravity = an anchor point *within the image*; the
    caller's (x, y) are offsets from it (``ForgetGravity`` uses (x, y)
    directly, top-left of the text)."""
    if gravity == FORGET_GRAVITY:
        return (0, 0)
    points = {
        NORTH_WEST_GRAVITY: (0, 0),
        NORTH_GRAVITY: (width / 2, 0),
        NORTH_EAST_GRAVITY: (width, 0),
        WEST_GRAVITY: (0, height / 2),
        CENTER_GRAVITY: (width / 2, height / 2),
        EAST_GRAVITY: (width, height / 2),
        SOUTH_WEST_GRAVITY: (0, height),
        SOUTH_GRAVITY: (width / 2, height),
        SOUTH_EAST_GRAVITY: (width, height),
    }
    return points[gravity]

# Effective alpha for grid strokes/fills — calibrated against the shipped
# JPGs' antialiased ImageMagick strokes (see module docstring).
_GRID_ALPHA = 64  # ~= 25% of 255

Color = Union[str, int, tuple]


def _resolve_font(font: str, size: int) -> ImageFont.FreeTypeFont:
    """Load ``font`` (a TTF path) at ``size``.

    Tried verbatim, then relative to the project root (the parent of the
    ``emips/`` package), which is what made ``../font/...`` work when the
    original scripts ran from ``ready/``.
    """
    if not font:
        raise ValueError("font is required")
    candidates = [Path(font)]
    if not Path(font).is_absolute():
        root = Path(__file__).resolve().parent.parent
        candidates.append(root / font)
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    raise FileNotFoundError(
        f"could not load font {font!r} (tried: {', '.join(str(c) for c in candidates)})"
    )


# X11 names the scripts use that Pillow's ImageColor does not ship.
_EXTRA_COLORS = {
    "maroon3": (127, 16, 43),
}


def _rgb(color: Color) -> tuple:
    text = str(color).strip()
    if text.lower() in _EXTRA_COLORS:
        return _EXTRA_COLORS[text.lower()]
    return ImageColor.getcolor(text, "RGB")


class ImageMe:
    """Pillow equivalent of ``BotImage::ImageMe``.

    ``width``/``height`` are the drawing canvas (the ``add_border`` margin
    added later is *not* included), matching the original where
    ``border!`` enlarged the canvas.
    """

    def __init__(self, width: int, height: int, bgcolor: Color = "transparent"):
        if str(bgcolor).lower() == "transparent":
            bg = (0, 0, 0)  # ImageMagick flattens transparent to black in JPEG
        else:
            bg = _rgb(bgcolor)
        self._image = Image.new("RGB", (width, height), bg)

    # -- introspection -----------------------------------------------------

    @property
    def image(self) -> Image.Image:
        return self._image

    @property
    def width(self) -> int:
        return self._image.width

    @property
    def height(self) -> int:
        return self._image.height

    # -- primitives --------------------------------------------------------

    def add_text_at(
        self,
        text: str,
        x: int,
        y: int,
        font: str,
        size: int,
        color: Color,
        gravity: str,
        bold: int,
        drop_shadow: int,
    ) -> None:
        """Draw ``text`` at (``x``, ``y``) anchored by ``gravity``.

        Mirrors the original parameter-for-parameter (including the legacy
        ``bold``/``drop_shadow`` int flags).  ``gravity`` is one of the
        ``*_GRAVITY`` constants above.
        """
        anchor = _ANCHORS[gravity]
        align = _ALIGNS[anchor[0]]
        ax, ay = _anchor_point(gravity, self._image.width, self._image.height)
        x, y = ax + x, ay + y
        rgb = _rgb(color)
        f = _resolve_font(font, size)
        draw = ImageDraw.Draw(self._image)

        if drop_shadow == 1:
            dx, dy = _SHADOW_DELTAS.get(gravity, (0, 0))
            draw.multiline_text(
                (x + dx, y + dy),
                text,
                font=f,
                fill=(128, 128, 128),  # the original shadow colour: "grey"
                anchor=anchor,
                align=align,
                spacing=0,
            )

        if bold == 1:
            draw.multiline_text(
                (x, y),
                text,
                font=f,
                fill=rgb,
                stroke_width=1,
                stroke_fill=rgb,
                anchor=anchor,
                align=align,
                spacing=0,
            )
        else:
            draw.multiline_text(
                (x, y), text, font=f, fill=rgb, anchor=anchor, align=align, spacing=0
            )

    def put_text2_width(
        self,
        text: str,
        x: int,
        y: int,
        width: int,
        font: str,
        size: int,
        color: Color,
        gravity: str,
        bold: int,
        drop_shadow: int,
    ) -> None:
        """Lay out one character every ``width`` px starting at (``x``, ``y``).

        Faithfully keeps the original quirk: the +1,+1 shadow is drawn
        *after* the main character (so it ends up on top).
        """
        anchor = _ANCHORS[gravity]
        align = _ALIGNS[anchor[0]]
        ax, ay = _anchor_point(gravity, self._image.width, self._image.height)
        x, y = ax + x, ay + y
        rgb = _rgb(color)
        f = _resolve_font(font, size)
        draw = ImageDraw.Draw(self._image)

        for index, letter in enumerate(text):
            pos = (x + width * index, y)
            if bold == 1:
                draw.multiline_text(
                    pos, letter, font=f, fill=rgb, stroke_width=1,
                    stroke_fill=rgb, anchor=anchor, align=align, spacing=0,
                )
            else:
                draw.multiline_text(
                    pos, letter, font=f, fill=rgb, anchor=anchor, align=align, spacing=0
                )
            if drop_shadow == 1:
                draw.multiline_text(
                    (pos[0] + 1, pos[1] + 1),
                    letter,
                    font=f,
                    fill=(128, 128, 128),
                    anchor=anchor,
                    align=align,
                    spacing=0,
                )

    def make_grid(
        self,
        x: int,
        y: int,
        cell_width: int,
        cell_height: int,
        number_of_rows: int,
        number_of_cols: int,
        color_cell_background: Color,
        color_cell_border: Color,
        opacity: Union[int, float, str],
    ) -> None:
        """Draw a ``number_of_rows`` x ``number_of_cols`` grid of cells.

        The ``opacity`` argument (e.g. ``"75%"``, ``1``, ``0.60``) is kept
        for API parity; the rendered strength is the calibrated
        :data:`_GRID_ALPHA` instead, matching the shipped JPGs.
        """
        stroke = _rgb(color_cell_border) + (_GRID_ALPHA,)
        fill = None
        bg = str(color_cell_background).strip()
        if bg and bg.lower() != "transparent":
            fill = _rgb(bg) + (_GRID_ALPHA,)
        draw = ImageDraw.Draw(self._image, "RGBA")
        for col in range(number_of_cols):
            for row in range(number_of_rows):
                x0 = col * cell_width + x
                y0 = row * cell_height + y
                draw.rectangle(
                    [x0, y0, x0 + cell_width, y0 + cell_height],
                    fill=fill,
                    outline=stroke,
                    width=1,
                )

    def add_border(self, border_width: int, border_height: int, border_color: Color) -> None:
        """Expand the canvas by a uniform ``border_width``x``border_height`` frame."""
        self._image = ImageOps.expand(
            self._image, border=(border_width, border_width, border_height, border_height),
            fill=_rgb(border_color),
        )

    def add_water_mark(
        self,
        text: str,
        font: str,
        color: Color,
        size: int,
        angle: int,
        opacity: float,
    ) -> None:
        """White-filled, ``color``-stroked centred text, rotated and composited.

        Angle sign matches ImageMagick (positive = clockwise), so the
        original ``-45`` renders as 45° counter-clockwise here.
        """
        canvas = self._image
        layer = Image.new("RGBA", (canvas.width, canvas.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        rgba = (*_rgb(color), 255)
        draw.multiline_text(
            (canvas.width / 2, canvas.height / 2 - 5),  # original annotate y offset of -5
            text,
            font=_resolve_font(font, size),
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=rgba,
            anchor="mm",
            align="center",
            spacing=0,
        )
        rotated = layer.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)

        overlay = Image.new("RGBA", (canvas.width, canvas.height), (0, 0, 0, 0))
        px = (canvas.width - rotated.width) // 2
        py = (canvas.height - rotated.height) // 2
        src_x, src_y = max(0, -px), max(0, -py)
        dst_x, dst_y = max(0, px), max(0, py)
        w = min(rotated.width - src_x, canvas.width - dst_x)
        h = min(rotated.height - src_y, canvas.height - dst_y)
        region = rotated.crop((src_x, src_y, src_x + w, src_y + h))
        overlay.paste(region, (dst_x, dst_y), region)

        # apply the watermark opacity, then composite over the base image
        alpha = overlay.getchannel("A").point(lambda v: int(v * opacity))
        overlay.putalpha(alpha)
        self._image = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    def write_file(self, filename: str) -> None:
        """Save as JPEG (ImageMagick's default JPEG quality is 92)."""
        self._image.save(filename, format="JPEG", quality=92)
