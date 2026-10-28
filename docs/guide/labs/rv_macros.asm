# Lab 86 — Code that writes code (guide 2.33).
# .rept with .set builds a table of the Fibonacci numbers F0 .. F15; main
# adds them up, unrolled (a macro used 16 times) or in a loop — .if UNROLL
# chooses.  The listing (python3 -m emips asm) shows what was assembled.
        .eqv    UNROLL, 1
        .eqv    N, 16
        .data
        .set    a, 0
        .set    b, 1
fib:    .rept   N                   # F0, F1, F2 ...
        .word   a
        .set    t, a + b
        .set    a, b
        .set    b, t
        .endr

        .macro  add_word reg:req, off=0
        lw      t0, \off(s0)
        add     \reg, \reg, t0
        .endm

        .text
main:   la      s0, fib
        li      a0, 0
        .if     UNROLL
        .set    k, 0
        .rept   N                   # 16 copies of add_word
        add_word a0, k
        .set    k, k + 4
        .endr
        .else
        li      t1, N
1:      add_word a0                 # (off: its default, 0)
        addi    s0, s0, 4
        addi    t1, t1, -1
        bnez    t1, 1b
        .endif
        li      a7, 1               # print the sum
        ecall
        li      a7, 10
        ecall
# eMIPS: --arch rv32
