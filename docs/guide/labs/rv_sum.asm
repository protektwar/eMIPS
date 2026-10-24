# Lab 73 — Lab 3's loop on a RISC-V hart: where do its cycles go?
# Run:  python3 -m emips pipe docs/guide/labs/rv_sum.asm
        .data
nums:   .word 1, 2, 3, 4, 5, 6, 7, 8
        .text
main:   la    t0, nums
        li    t1, 8
        li    t2, 0
loop:   lw    t3, 0(t0)
        add   t2, t2, t3
        addi  t0, t0, 4
        addi  t1, t1, -1
        bnez  t1, loop
        mv    a0, t2
        li    a7, 1             # prints 36
        ecall
        li    a7, 10
        ecall
# eMIPS: --arch rv32
