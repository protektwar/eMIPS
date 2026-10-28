# Lab 78 — strip mining: a vector loop that runs on any VLEN.
# The vector unit chooses how many elements one iteration does (vl);
# change m1 to m2, m4, m8 and --rv-ext to ..._zvl256b and count the loops.
        .data
n:      .dword 100
a:      .space 400              # 100 words, filled with 1 ... 100
        .text
main:   la      t0, a
        li      t1, 1
        li      t2, 101
fill:   sw      t1, 0(t0)
        addi    t0, t0, 4
        addi    t1, t1, 1
        bne     t1, t2, fill
        la      a1, a
        ld      a2, n           # elements left
        li      s0, 0           # loops
        vsetvli t0, zero, e32, m1, ta, ma
        vmv.v.i v8, 0           # v8[0]: the sum so far
loop:   vsetvli t0, a2, e32, m1, ta, ma   # t0 = vl: this iteration's elements
        vle32.v v4, (a1)
        vredsum.vs v8, v4, v8   # v8[0] += v4[0] + ... + v4[vl-1]
        slli    t1, t0, 2
        add     a1, a1, t1      # the next vl words
        sub     a2, a2, t0
        addi    s0, s0, 1
        bnez    a2, loop
        vmv.x.s a0, v8
        li      a7, 1
        ecall                   # the sum
        li      a0, ' '
        li      a7, 11
        ecall
        mv      a0, s0
        li      a7, 1
        ecall                   # the loops
        li      a7, 10
        ecall
# eMIPS: --arch rv64 --rv-ext imafdcv_zicsr_zicntr
