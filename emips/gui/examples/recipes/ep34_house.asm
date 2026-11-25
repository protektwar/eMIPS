# MIPS by Example in eMIPS, episode 34 — Drawing a house on the bitmap
# display.  The display shows memory as pixels: 64 x 64 words from
# 0x10008000 ($gp's address in MARS), one word 0x00RRGGBB a pixel, row
# after row.  Pixel (x, y) is the word at 0x10008000 + (y * 64 + x) * 4.
# Everything is drawn from horizontal lines; later shapes cover earlier
# ones, as a painter's layers.
        .data
        .eqv  SCREEN, 0x10008000      # the display's first word
pixels: .word 0                       # how many pixels were written
done:   .asciiz "pixels written: "
        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 16 the fifth argument,
        sw    $ra, 28($sp)            # 0-15 the home area
        sw    $fp, 24($sp)
        move  $fp, $sp
        li    $t0, 0x0080c0ff         # the sky: the whole screen
        sw    $t0, 16($sp)
        li    $a0, 0
        li    $a1, 0
        li    $a2, 63
        li    $a3, 63
        jal   rect
        li    $t0, 0x0040a040         # the grass
        sw    $t0, 16($sp)
        li    $a0, 0
        li    $a1, 52
        li    $a2, 63
        li    $a3, 63
        jal   rect
        li    $t0, 0x00a04040         # the chimney, before the roof covers its foot
        sw    $t0, 16($sp)
        li    $a0, 40
        li    $a1, 14
        li    $a2, 44
        li    $a3, 24
        jal   rect
        li    $a0, 31                 # the roof: apex (31, 11), 19 rows
        li    $a1, 11
        li    $a2, 19
        li    $a3, 0x00c03020
        jal   roof
        li    $t0, 0x00e0c080         # the walls
        sw    $t0, 16($sp)
        li    $a0, 14
        li    $a1, 30
        li    $a2, 48
        li    $a3, 51
        jal   rect
        li    $t0, 0x00804020         # the door
        sw    $t0, 16($sp)
        li    $a0, 27
        li    $a1, 40
        li    $a2, 35
        li    $a3, 51
        jal   rect
        li    $t0, 0x00ffff80         # two windows
        sw    $t0, 16($sp)
        li    $a0, 18
        li    $a1, 34
        li    $a2, 24
        li    $a3, 40
        jal   rect
        li    $a0, 38
        li    $a1, 34
        li    $a2, 44
        li    $a3, 40
        jal   rect
        li    $v0, 4                  # "pixels written: n"
        la    $a0, done
        syscall
        li    $v0, 1
        lw    $a0, pixels
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# rect(x0, y0, x1, y1, colour): a filled rectangle, corners included; the
# colour is the fifth argument, at 16 in the caller's frame
rect:   addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28-16 $s0-$s3
        sw    $ra, 36($sp)
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        sw    $s3, 16($sp)
        move  $fp, $sp
        move  $s0, $a0                # x0, x1, the next row, the last row
        move  $s1, $a2
        move  $s2, $a1
        move  $s3, $a3
        lw    $t9, 56($fp)            # the colour: 40 + 16, the caller's frame
rnext:  bgt   $s2, $s3, rdone
        move  $a0, $s0                # hline(x0, x1, y, colour)
        move  $a1, $s1
        move  $a2, $s2
        move  $a3, $t9
        jal   hline
        lw    $t9, 56($fp)            # ($t9 is not kept across a call)
        addiu $s2, $s2, 1
        b     rnext
rdone:  move  $sp, $fp
        lw    $s3, 16($sp)
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra

# roof(xc, ytop, rows, colour): a triangle, one row longer on each side
# every row down
roof:   addiu $sp, $sp, -48           # 44 $ra, 40 $fp, 36-20 $s0-$s4
        sw    $ra, 44($sp)
        sw    $fp, 40($sp)
        sw    $s0, 36($sp)
        sw    $s1, 32($sp)
        sw    $s2, 28($sp)
        sw    $s3, 24($sp)
        sw    $s4, 20($sp)
        move  $fp, $sp
        move  $s0, $a0                # xc
        move  $s1, $a1                # the row's y
        move  $s2, $a2                # rows left
        move  $s3, $a3                # the colour
        li    $s4, 0                  # k: the half-width
onext:  beqz  $s2, odone
        subu  $a0, $s0, $s4           # hline(xc - k, xc + k, y, colour)
        addu  $a1, $s0, $s4
        move  $a2, $s1
        move  $a3, $s3
        jal   hline
        addiu $s4, $s4, 1             # one wider each side
        addiu $s1, $s1, 1             # one row down
        addiu $s2, $s2, -1
        b     onext
odone:  move  $sp, $fp
        lw    $s4, 20($sp)
        lw    $s3, 24($sp)
        lw    $s2, 28($sp)
        lw    $s1, 32($sp)
        lw    $s0, 36($sp)
        lw    $fp, 40($sp)
        lw    $ra, 44($sp)
        addiu $sp, $sp, 48
        jr    $ra

# hline(x0, x1, y, colour): the pixels (x0, y) ... (x1, y) (a leaf)
hline:  sll   $t0, $a2, 6             # y * 64: the words before row y
        addu  $t0, $t0, $a0           # + x0
        sll   $t0, $t0, 2             # * 4: bytes
        li    $t1, SCREEN
        addu  $t0, $t0, $t1           # the first pixel's address
        lw    $t2, pixels
hnext:  bgt   $a0, $a1, hdone
        sw    $a3, 0($t0)
        addiu $t0, $t0, 4             # the next pixel: the next word
        addiu $a0, $a0, 1
        addiu $t2, $t2, 1
        b     hnext
hdone:  sw    $t2, pixels
        jr    $ra
# eMIPS: --display 256x256:4@gp
