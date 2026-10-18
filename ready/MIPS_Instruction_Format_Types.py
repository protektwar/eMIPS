#!/usr/bin/env python3
"""Renders ``MIPS_Instruction_Format_Types.jpg`` — Python port of
``MIPS_Instruction_Format_Types.rb``.

Runs from any working directory and rewrites the output in place (1400x900
canvas + 40px DarkViolet border = 1480x980 output), like the original.
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

OPCODE_COLOR = "blue"
SOURCE_COLOR = "green"
DESTINATION_COLOR = "maroon3"
SHIFT_AMOUNT_COLOR = "black"
FUNCTION_COLOR = "DarkTurquoise"
BASE_COLOR = "orange"
ADDRESS_COLOR = "lime"
GRID_BG_COLOR = "transparent"
GRID_COLOR = "grey"


def _width_numbers(img: ImageMe, x_paragraph: int, y_paragraph: int, width: int) -> None:
    for number in range(31, -1, -1):  # 31.downto(0)
        img.add_text_at(
            str(number), x_paragraph + 10 + width * (31 - number), y_paragraph + 45,
            FONT, 20, "black", NORTH_WEST_GRAVITY, 0, 1,
        )


def main() -> None:
    img = ImageMe(1400, 900, "white")
    img.add_water_mark("eMIPS Technologies", FONT, "maroon3", 120, -45, 0.25)

    x_title = 0
    y_title = 120
    y_between = 80
    y_start_format = 100
    img.add_text_at(
        "MIPS Instruction Format", x_title, y_title, FONT, 40, "black", NORTH_GRAVITY, 1, 1
    )

    x_paragraph = 90
    y_paragraph = y_title + y_start_format

    # ---------------------- R-Type Format ----------------------
    img.add_text_at(
        "R-Type Format  (Register Type Format)", x_paragraph, y_paragraph,
        FONT, 30, "red", NORTH_WEST_GRAVITY, 0, 1,
    )
    width = 38
    height = 60
    img.make_grid(x_paragraph, y_paragraph + 70, width, height, 1, 32, GRID_BG_COLOR, GRID_COLOR, 1)

    img.make_grid(x_paragraph, y_paragraph + 70, width * 6, height, 1, 1, "", "black", 0.60)
    img.make_grid(x_paragraph + width * 6, y_paragraph + 70, width * 5, height, 1, 1, "", "black", 0.60)
    img.make_grid(x_paragraph + width + 2 * width * 5, y_paragraph + 70, width * 5, height, 1, 1, "", "black", 0.60)
    img.make_grid(x_paragraph + width + 3 * width * 5, y_paragraph + 70, width * 5, height, 1, 1, "", "black", 0.60)
    img.make_grid(x_paragraph + width + 4 * width * 5, y_paragraph + 70, width * 5, height, 1, 1, "", "black", 0.60)
    img.make_grid(x_paragraph + width + 5 * width * 5, y_paragraph + 70, width * 6, height, 1, 1, "", "black", 0.60)

    _width_numbers(img, x_paragraph, y_paragraph, width)

    img.add_text_at("Opcode\n6 bits", x_paragraph + 20, y_paragraph + 80, FONT, 20, OPCODE_COLOR, NORTH_WEST_GRAVITY, 0, 1)
    img.add_text_at("Source\n5 bits", x_paragraph + 20 + width * 6, y_paragraph + 80, FONT, 20, SOURCE_COLOR, NORTH_WEST_GRAVITY, 0, 1)
    img.add_text_at("Target\n5 bits", x_paragraph + 20 + 2 * width * 5 + width, y_paragraph + 80, FONT, 20, SOURCE_COLOR, NORTH_WEST_GRAVITY, 0, 1)
    img.add_text_at("Destination\n5 bits", x_paragraph + 20 + width + 3 * width * 5, y_paragraph + 80, FONT, 20, DESTINATION_COLOR, NORTH_WEST_GRAVITY, 0, 1)
    img.add_text_at("Shift  amount\n5 bits", x_paragraph + 20 + width + 4 * width * 5, y_paragraph + 80, FONT, 20, SHIFT_AMOUNT_COLOR, NORTH_WEST_GRAVITY, 0, 1)
    img.add_text_at("Function\n6 bits", x_paragraph + 20 + width + 5 * width * 5, y_paragraph + 80, FONT, 20, FUNCTION_COLOR, NORTH_WEST_GRAVITY, 0, 1)

    # ------------------- Immediate Type Format -------------------
    y_paragraph = y_paragraph + y_between + y_start_format
    img.add_text_at(
        "I-Type Format  (Immediate Type Format)", x_paragraph, y_paragraph,
        FONT, 30, "red", NORTH_WEST_GRAVITY, 0, 1,
    )
    width = 38
    height = 60
    img.make_grid(x_paragraph, y_paragraph + 70, width, height, 1, 32, GRID_BG_COLOR, GRID_COLOR, 1)

    img.make_grid(x_paragraph, y_paragraph + 70, width * 6, height, 1, 1, "", "black", 0.60)
    img.make_grid(x_paragraph + width * 6, y_paragraph + 70, width * 5, height, 1, 1, "", "black", 0.60)
    img.make_grid(x_paragraph + width + 2 * width * 5, y_paragraph + 70, width * 5, height, 1, 1, "", "black", 0.60)
    img.make_grid(x_paragraph + width + 3 * width * 5, y_paragraph + 70, width * 16, height, 1, 1, "", "black", 0.60)

    _width_numbers(img, x_paragraph, y_paragraph, width)

    img.add_text_at("Opcode\n6 bits", x_paragraph + 20, y_paragraph + 80, FONT, 20, OPCODE_COLOR, NORTH_WEST_GRAVITY, 0, 1)
    img.add_text_at("Base\n5 bits", x_paragraph + 20 + width * 6, y_paragraph + 80, FONT, 20, BASE_COLOR, NORTH_WEST_GRAVITY, 0, 1)
    img.add_text_at("Destination\n5 bits", x_paragraph + 20 + width + 2 * width * 5, y_paragraph + 80, FONT, 20, DESTINATION_COLOR, NORTH_WEST_GRAVITY, 0, 1)
    img.add_text_at("Address\n16 bits", x_paragraph + 20 + width + 3 * width * 5, y_paragraph + 80, FONT, 20, ADDRESS_COLOR, NORTH_WEST_GRAVITY, 0, 1)

    # -------------------- Jump Type Format -----------------------
    y_paragraph = y_paragraph + y_between + y_start_format
    img.add_text_at(
        "J-Type Format  (Jump Type Format)", x_paragraph, y_paragraph,
        FONT, 30, "red", NORTH_WEST_GRAVITY, 0, 1,
    )
    width = 38
    height = 60
    img.make_grid(x_paragraph, y_paragraph + 70, width, height, 1, 32, GRID_BG_COLOR, GRID_COLOR, 1)

    img.make_grid(x_paragraph, y_paragraph + 70, width * 6, height, 1, 1, "", "black", 0.60)
    img.make_grid(x_paragraph + width * 6, y_paragraph + 70, width * 26, height, 1, 1, "", "black", 0.60)

    _width_numbers(img, x_paragraph, y_paragraph, width)

    img.add_text_at("Opcode\n6 bits", x_paragraph + 20, y_paragraph + 80, FONT, 20, OPCODE_COLOR, NORTH_WEST_GRAVITY, 0, 1)
    img.add_text_at("Address\n26 bits", x_paragraph + 20 + width * 6, y_paragraph + 80, FONT, 20, ADDRESS_COLOR, NORTH_WEST_GRAVITY, 0, 1)

    img.add_border(40, 40, "DarkViolet")
    img.write_file(str(HERE / "MIPS_Instruction_Format_Types.jpg"))


if __name__ == "__main__":
    main()
