# Lab: non-blocking loads on the MIPS 24K (run with --model 24k --caches).
# Four loads from four different cache lines, then their sum.  The 24K
# does not wait for a missing load until an instruction needs its data,
# so the four misses overlap.  Set BLOCK to 1 (Config7.SL = 1: load
# misses block, as in the textbook) and compare the cycles.
        .eqv  BLOCK, 0
        .data
a:      .word 1
        .space 60                    # next line (and the next ...)
b:      .word 2
        .space 60
c:      .word 3
        .space 60
d:      .word 4
        .text
main:   li    $t0, BLOCK
        mtc0  $t0, $16, 7            # Config7.SL (bit 0): blocking load misses
        la    $t1, a
        lw    $t2, 0($t1)            # miss
        lw    $t3, 64($t1)           # miss, while the first is on its way
        lw    $t4, 128($t1)          # miss
        lw    $t5, 192($t1)          # miss
        addu  $a0, $t2, $t3          # waits for the data
        addu  $a0, $a0, $t4
        addu  $a0, $a0, $t5
        li    $v0, 1
        syscall                      # 10
        li    $v0, 10
        syscall
# eMIPS: --model 24k --caches --pipeline    (the settings this program runs with)
