# Lab 85 — Two trace protocols, one program (guide 5.13).
# `main` sums the digits of the numbers 1 .. 300 (a loop of branches and
# calls), then computes fib(12) with a recursion; it prints both results.
# Trace it with the debugger:  trace on [etrace|ntrace] [OPTIONS], run,
# trace off, trace  (the packets and bytes), trace show / packets.
# Two triggers (Debug spec 5.7) give the encoder a window: trigger 0 at
# fib's first instruction asks for trace on (action 2), trigger 1 where
# the call returns asks for trace off (action 3) — used by `trace on wait`.
        .text
main:   csrw    tselect, zero       # trigger 0: execute at fib -> trace on
        la      t0, fib
        csrw    tdata2, t0
        li      t0, 0x60002044      # mcontrol6, action 2, M-mode, execute
        csrw    tdata1, t0
        li      t0, 1               # trigger 1: execute at back -> trace off
        csrw    tselect, t0
        la      t0, back
        csrw    tdata2, t0
        li      t0, 0x60003044      # mcontrol6, action 3, M-mode, execute
        csrw    tdata1, t0
        li      s0, 1               # the number
        li      s1, 0               # the sum of all digits
next:   mv      a0, s0
        call    digits              # a0 = the sum of a0's digits
        add     s1, s1, a0
        addi    s0, s0, 1
        li      t0, 301
        blt     s0, t0, next
        mv      a0, s1
        li      a7, 1               # print 2001... the digit sum
        ecall
        li      a0, 10
        li      a7, 11
        ecall
        li      a0, 12
        call    fib
back:   li      a7, 1               # print fib(12) = 144
        ecall
        li      a7, 10
        ecall

digits: li      t1, 0               # a0: the number; returns its digit sum
        li      t2, 10
d1:     remu    t3, a0, t2
        add     t1, t1, t3
        divu    a0, a0, t2
        bnez    a0, d1
        mv      a0, t1
        ret

fib:    li      t0, 2               # a0 = fib(a0), recursively
        blt     a0, t0, f2
        addi    sp, sp, -16
        sw      ra, 12(sp)
        sw      a0, 8(sp)
        addi    a0, a0, -1
        call    fib
        sw      a0, 4(sp)
        lw      a0, 8(sp)
        addi    a0, a0, -2
        call    fib
        lw      t0, 4(sp)
        add     a0, a0, t0
        lw      ra, 12(sp)
        addi    sp, sp, 16
f2:     ret
# eMIPS: --arch rv32
