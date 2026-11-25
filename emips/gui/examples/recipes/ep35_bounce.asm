# MIPS by Example in eMIPS, episode 35 — Animation without flicker: double
# buffering and the vertical blank.  A ball bounces on a 32 x 32-word
# display.  Two buffers: the program draws the next picture in the one not
# shown, asks the display to show it (BASE, taken at the next vertical
# blank), waits for that blank (STATUS.VBLANK), then swaps their roles.
        .data
        .eqv  DISPLAY, 0xffff0020     # the display's registers (kernel mode)
        .eqv  FRAMES, 60              # pictures to show
done:   .asciiz "frames: "
        .text
main:   addiu $sp, $sp, -56           # 52 $ra, 48 $fp, 44-16 $s0-$s7
        sw    $ra, 52($sp)
        sw    $fp, 48($sp)
        sw    $s0, 44($sp)
        sw    $s1, 40($sp)
        sw    $s2, 36($sp)
        sw    $s3, 32($sp)
        sw    $s4, 28($sp)
        sw    $s5, 24($sp)
        sw    $s6, 20($sp)
        sw    $s7, 16($sp)
        move  $fp, $sp
        li    $s7, DISPLAY
        li    $t0, 4800               # PERIOD: a frame every 4800 cycles
        sw    $t0, 0x14($s7)
        li    $s0, 0x10009000         # the buffer to draw (B) ...
        li    $s1, 0x10008000         # ... and the one shown (A, at reset)
        li    $s2, 3                  # the ball: x, y, dx, dy
        li    $s3, 10
        li    $s4, 1
        li    $s5, 1
        li    $s6, FRAMES
frame:  move  $a0, $s0                # the next picture, in the hidden buffer
        jal   clear
        move  $a0, $s0
        move  $a1, $s2
        move  $a2, $s3
        jal   ball
        sw    $s0, 8($s7)             # BASE: show it from the next blank
        li    $t0, 1
        sw    $t0, 4($s7)             # STATUS: clear VBLANK ...
vwait:  lw    $t0, 4($s7)             # ... and wait until it is set again
        beqz  $t0, vwait
        move  $t0, $s0                # the shown one is drawn next
        move  $s0, $s1
        move  $s1, $t0
        addu  $s2, $s2, $s4           # move the ball
        bltz  $s2, xbounce
        ble   $s2, 28, xok
xbounce:subu  $s4, $zero, $s4         # off a side: turn back
        addu  $s2, $s2, $s4
        addu  $s2, $s2, $s4
xok:    addu  $s3, $s3, $s5
        bltz  $s3, ybounce
        ble   $s3, 28, yok
ybounce:subu  $s5, $zero, $s5
        addu  $s3, $s3, $s5
        addu  $s3, $s3, $s5
yok:    addiu $s6, $s6, -1
        bnez  $s6, frame
        li    $v0, 4                  # "frames: n" (the display counted them)
        la    $a0, done
        syscall
        li    $v0, 1
        lw    $a0, 0x10($s7)          # FRAME
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s7, 16($sp)
        lw    $s6, 20($sp)
        lw    $s5, 24($sp)
        lw    $s4, 28($sp)
        lw    $s3, 32($sp)
        lw    $s2, 36($sp)
        lw    $s1, 40($sp)
        lw    $s0, 44($sp)
        lw    $fp, 48($sp)
        lw    $ra, 52($sp)
        addiu $sp, $sp, 56
        li    $v0, 10
        syscall

# clear(buffer): all 1024 words the background colour (a leaf)
clear:  li    $t0, 0x00102040         # night blue
        addiu $t1, $a0, 4096          # the end of the buffer
cnext:  sw    $t0, 0($a0)
        addiu $a0, $a0, 4
        bne   $a0, $t1, cnext
        jr    $ra

# ball(buffer, x, y): a 4 x 4 ball at (x, y) (a leaf)
ball:   sll   $t0, $a2, 5             # y * 32 + x: the word
        addu  $t0, $t0, $a1
        sll   $t0, $t0, 2
        addu  $t0, $t0, $a0           # its address
        li    $t1, 0x00ffa020         # orange
        li    $t2, 4                  # four rows
bnext:  sw    $t1, 0($t0)
        sw    $t1, 4($t0)
        sw    $t1, 8($t0)
        sw    $t1, 12($t0)
        addiu $t0, $t0, 128           # the next row: 32 words on
        addiu $t2, $t2, -1
        bnez  $t2, bnext
        jr    $ra
# eMIPS: --display 256x256:8@gp
