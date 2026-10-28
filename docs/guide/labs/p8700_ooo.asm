# Lab 87 — Out of order on the P8700 (guide 5.14, "The P8700's own pipeline")
# Each trip loads a word, adds it up and multiplies twice.  CHAINS 1: both
# multiplies feed a0, one after the other; CHAINS 2: one feeds a0, the
# other a1.  PREDICT 0 turns branch prediction off (mipsconfig7.DBP).
        .eqv    CHAINS, 1
        .eqv    PREDICT, 1
        .data
buf:    .space  512
        .text
main:
        .if PREDICT == 0
        li      t0, 2                   # mipsconfig7.DBP (MD01502 Table 7.22)
        csrw    0x7d7, t0
        .endif
        la      s0, buf
        li      s1, 64                  # trips
        li      a0, 1
        li      a1, 1
        li      t1, 3
loop:   ld      t2, 0(s0)
        add     t3, t3, t2
        mul     a0, a0, t1
        .if CHAINS == 2
        mul     a1, a1, t1
        .else
        mul     a0, a0, t1
        .endif
        addi    s0, s0, 8
        addi    s1, s1, -1
        bnez    s1, loop
        add     a0, a0, a1
        li      a7, 1                   # print a0
        ecall
        li      a7, 10
        ecall
# eMIPS: --arch rv64 --core p8700 --pipeline
