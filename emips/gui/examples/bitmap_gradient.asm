# The bitmap display: a 16 x 16-word gradient, red down and blue across.
# One word 0x00RRGGBB a pixel, row by row from 0x10010000 (MARS's default
# base, "static"); each word is drawn as 16 x 16 screen pixels.
# Open View > Display (it opens by itself with the display on).
        .text
main:   li    $t0, 0x10010000
        li    $t1, 0                  # row
row:    li    $t2, 0                  # column
col:    sll   $t3, $t1, 20            # red   = row * 16
        sll   $t4, $t2, 4             # blue  = column * 16
        or    $t3, $t3, $t4
        sw    $t3, 0($t0)
        addiu $t0, $t0, 4
        addiu $t2, $t2, 1
        blt   $t2, 16, col
        addiu $t1, $t1, 1
        blt   $t1, 16, row
        li    $v0, 10
        syscall
# eMIPS: --display 256x256:16@static
