# Lab 94 — Frames: a 4 x 4 square crosses a 32 x 32-word display, one
# position a frame.  Two buffers: the program draws in the one not shown,
# asks the display to show it from the next frame (BASE), and sleeps (wait)
# until the vertical-blank interrupt says the swap has happened.
# The display's registers are at 0xffff0020 (see the guide, 2.35).
# Run:  python3 -m emips run docs/guide/labs/display_vblank.asm
        .eqv  DISPLAY, 0xffff0020
        .eqv  FRAMES, 28
        .text
main:   li    $s7, DISPLAY
        li    $t0, 3000               # PERIOD: a frame every 3000 cycles
        sw    $t0, 0x14($s7)          # (333 Hz: a short run; 60 Hz is 16667)
        li    $t0, 3                  # CONTROL: ON | IE
        sw    $t0, 0($s7)
        mfc0  $t0, $12                # Status: IM4 (the display's IP4) and IE
        ori   $t0, $t0, 0x1001
        mtc0  $t0, $12
        li    $v0, 30                 # the time now
        syscall
        move  $s6, $a0
        move  $s0, $gp                # the buffer shown (A)
        addiu $s1, $gp, 0x1000        # the buffer to draw (B)
        li    $s2, 0                  # x of the square
frame:  addiu $a0, $s2, -2            # this buffer had the square at x - 2:
        move  $a1, $s1
        move  $a2, $zero              # erase it
        jal   square
        move  $a0, $s2                # and draw it at x
        move  $a1, $s1
        li    $a2, 0x00ffc000
        jal   square
        sw    $s1, 8($s7)             # BASE: show this buffer from the next frame
        move  $s5, $s4                # the blanks counted so far
sleep:  wait                          # (the interrupt wakes it)
        beq   $s4, $s5, sleep
        move  $t0, $s0                # swap the roles
        move  $s0, $s1
        move  $s1, $t0
        addiu $s2, $s2, 1
        blt   $s2, FRAMES, frame
        mtc0  $zero, $12              # interrupts off
        li    $v0, 30
        syscall
        subu  $a0, $a0, $s6           # the milliseconds it took
        li    $v0, 1
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        lw    $a0, 0x10($s7)          # FRAME
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall

# square: a 4 x 4 square at column $a0 (if >= 0), row 14, colour $a2, in
# the buffer at $a1 (32 words a row)
square: bltz  $a0, 2f
        sll   $t0, $a0, 2
        addu  $t0, $t0, $a1
        addiu $t0, $t0, 14 * 128      # row 14
        li    $t1, 4
1:      sw    $a2, 0($t0)
        sw    $a2, 4($t0)
        sw    $a2, 8($t0)
        sw    $a2, 12($t0)
        addiu $t0, $t0, 128           # the next row
        addiu $t1, $t1, -1
        bnez  $t1, 1b
2:      jr    $ra

        .ktext 0x80000180             # the vertical blank
        li    $k0, DISPLAY
        li    $k1, 1
        sw    $k1, 4($k0)             # clear STATUS.VBLANK
        addiu $s4, $s4, 1             # count it
        eret
# eMIPS: --display 256x256:8@gp
