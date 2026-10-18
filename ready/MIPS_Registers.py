#!/usr/bin/env python3
"""Renders ``MIPS_Registers.jpg`` — Python port of ``MIPS_Registers.rb``.

Runs from any working directory and rewrites ``ready/MIPS_Registers.jpg``
in place, exactly like the original Ruby script did (1400x1400 canvas +
40px DarkViolet border = 1480x1480 output).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from emips.image import (
    ImageMe,
    NORTH_GRAVITY,
    NORTH_WEST_GRAVITY,
)

FONT = str(HERE.parent / "font" / "Gruppo-Regular.ttf")  # original: ../font/Gruppo-Regular.ttf

# (assembler name, register number/name, description) — one entry per grid row.
ROWS = [
    ("$zero", "r0", "Constant 0 value\nHardwired to ZERO"),
    ("$at", "r1", "Assembler temporary"),
    ("$v0 - $v1", "r2 - r3", "Procedure return values or\nexpression evaluation"),
    ("$a0 - $a3", "r4 - r7", "Arguments / Parameters"),
    ("$t0 - $t7", "r8 - r15", "Temporaries"),
    ("$s0 - $s7", "r16-r23", "Saved Temporaries"),
    ("$t8 - $t9", "r24 - r25", "Temporaries"),
    ("$k0 - $k1", "r26 - r27", "Reserved for OS kernel"),
    ("$gp", "r28", "Global Pointer\n(Global and static variables/data)"),
    ("$sp", "r29", "Stack Pointer"),
    ("$fp", "r30", "Frame Pointer"),
    ("$ra", "r31", "Return address\nfor current procedure"),
    ("not visible", "PC", "Program Counter\n(hidden register)"),
    ("not visible", "IR", "Instruction Register\n(hidden register)"),
    ("not visible", "HI", "results of mul/div\n(hidden register)"),
    ("not visible", "LO", "results of mul/div\n(hidden register)"),
]


def main() -> None:
    image_border_color = "DarkViolet"
    img = ImageMe(1400, 1400, "white")
    img.add_water_mark("eMIPS Technologies", FONT, "maroon3", 120, -45, 0.25)

    x_title = 0
    y_title = 120
    img.add_text_at(
        "MIPS Registers", x_title, y_title, FONT, 40, "black", NORTH_GRAVITY, 1, 1
    )

    x_paragraph = 90
    y_paragraph = y_title + 50  # yStartFormat
    width = 400
    height = 60
    assembler_name_color = "blue"
    register_number_color = "green"
    description_color = "black"

    img.make_grid(x_paragraph, y_paragraph + 70, width, height, 17, 3, "white", "grey", "75%")

    header_table_font_size = 30
    img.add_text_at(
        "Assembler Name", x_paragraph + 20, y_paragraph + 90,
        FONT, header_table_font_size, assembler_name_color, NORTH_WEST_GRAVITY, 0, 1,
    )
    img.add_text_at(
        "Register Number/Name", x_paragraph + 20 + width, y_paragraph + 90,
        FONT, header_table_font_size, register_number_color, NORTH_WEST_GRAVITY, 0, 1,
    )
    img.add_text_at(
        "Description", x_paragraph + 20 + 2 * width, y_paragraph + 90,
        FONT, header_table_font_size, description_color, NORTH_WEST_GRAVITY, 1, 1,
    )

    content_table_font_size = 23
    y_paragraph += 60
    for assembler_name, register_number, description in ROWS:
        # the original offset multi-line descriptions by 80 instead of 90
        desc_offset = 80 if "\n" in description else 90
        img.add_text_at(
            assembler_name, x_paragraph + 20, y_paragraph + 90,
            FONT, content_table_font_size, assembler_name_color, NORTH_WEST_GRAVITY, 0, 1,
        )
        img.add_text_at(
            register_number, x_paragraph + 20 + width, y_paragraph + 90,
            FONT, content_table_font_size, register_number_color, NORTH_WEST_GRAVITY, 0, 1,
        )
        img.add_text_at(
            description, x_paragraph + 20 + 2 * width, y_paragraph + desc_offset,
            FONT, content_table_font_size, description_color, NORTH_WEST_GRAVITY, 1, 1,
        )
        y_paragraph += 60

    img.add_border(40, 40, image_border_color)
    img.write_file(str(HERE / "MIPS_Registers.jpg"))


if __name__ == "__main__":
    main()
