# Lab 90 — Lines on a bitmap display: Bresenham's algorithm draws a
# triangle on a 32 x 32-word display (MARS's Bitmap Display: one word
# 0x00RRGGBB a pixel, row by row from $gp = 0x10008000).
# Run:  python3 -m emips run docs/guide/labs/display_lines.asm --display-png tri.png
# (the window: the Display pane shows it)
        .text
main:   li    $s7, 0x00ff4040        # red
        li    $a0, 2                 # (2, 29) to (16, 2)
        li    $a1, 29
        li    $a2, 16
        li    $a3, 2
        jal   line
        li    $s7, 0x0040ff40        # green
        li    $a0, 16                # (16, 2) to (29, 29)
        li    $a1, 2
        li    $a2, 29
        li    $a3, 29
        jal   line
        li    $s7, 0x004080ff        # blue
        li    $a0, 29                # (29, 29) to (2, 29)
        li    $a1, 29
        li    $a2, 2
        li    $a3, 29
        jal   line
        move  $a0, $s6               # how many pixels were plotted
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall

# line: from ($a0, $a1) to ($a2, $a3) in colour $s7, any direction.
# dx = |x1 - x0|, dy = -|y1 - y0|, err = dx + dy; each step moves x, y or
# both, whichever keeps the error smallest.
line:   subu  $t0, $a2, $a0          # dx
        li    $t2, 1                 # sx
        bgez  $t0, 1f
        negu  $t0, $t0
        li    $t2, -1
1:      subu  $t1, $a3, $a1          # dy
        li    $t3, 1                 # sy
        bgez  $t1, 2f
        negu  $t1, $t1
        li    $t3, -1
2:      negu  $t1, $t1               # dy = -|dy|
        addu  $t4, $t0, $t1          # err
plot:   sll   $t5, $a1, 5            # address = $gp + (y * 32 + x) * 4
        addu  $t5, $t5, $a0
        sll   $t5, $t5, 2
        addu  $t5, $t5, $gp
        sw    $s7, 0($t5)
        addiu $s6, $s6, 1
        bne   $a0, $a2, 3f
        beq   $a1, $a3, 5f           # the end point: done
3:      sll   $t6, $t4, 1            # e2 = 2 * err
        blt   $t6, $t1, 4f           # e2 >= dy: step x
        addu  $t4, $t4, $t1
        addu  $a0, $a0, $t2
4:      bgt   $t6, $t0, plot         # e2 <= dx: step y
        addu  $t4, $t4, $t0
        addu  $a1, $a1, $t3
        b     plot
5:      jr    $ra
# eMIPS: --display 256x256:8@gp
