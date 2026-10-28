# Lab 79 — masks: make every element of nums non-negative.
# As handed out a scalar loop with a branch per element; do it with
# vector instructions: a compare that writes a mask into v0, then an
# instruction that acts only where the mask is set (v0.t).
# (python3 -m emips check docs/guide/labs/rv_vabs.asm checks a solution)
        .data
count:  .dword 12
nums:   .word 5, -3, 0, -128, 77, -1, 2147483647, -2147483647, 9, -9, 64, -65
        .text
main:   la      a1, nums
        ld      a2, count
loop:   lw      t0, 0(a1)
        bgez    t0, next
        neg     t0, t0          # the negative ones: 0 - x
        sw      t0, 0(a1)
next:   addi    a1, a1, 4
        addi    a2, a2, -1
        bnez    a2, loop
        li      a7, 10
        ecall
# eMIPS: --arch rv64 --rv-ext imafdcv_zicsr_zicntr
