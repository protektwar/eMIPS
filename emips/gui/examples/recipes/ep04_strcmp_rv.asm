# MIPS by Example in eMIPS, episode 4, the RISC-V ending: the same strcmp
# on RV32 — no delay slots, compare-and-branch on two registers (bne, beqz),
# the result in a0.
# eMIPS: --arch rv32
        .data
apple:  .asciz "apple"
apply:  .asciz "apply"
pear:   .asciz "pear"
peach:  .asciz "peach"
mips:   .asciz "MIPS"
zebra:  .asciz "Zebra"
pairs:  .word apple, apply, pear, peach, mips, mips, zebra, apple, 0
less:   .asciz " < "
same:   .asciz " = "
more:   .asciz " > "
        .text
main:   addi  sp, sp, -32             # ra and the s registers main uses
        sw    ra, 28(sp)
        sw    s0, 24(sp)
        sw    s1, 20(sp)
        sw    s2, 16(sp)
        sw    s3, 12(sp)
        la    s0, pairs               # the next pair
next:   lw    s1, 0(s0)               # a (0: no more pairs)
        beqz  s1, end
        lw    s2, 4(s0)               # b
        mv    a0, s1                  # strcmp(a, b)
        mv    a1, s2
        call  strcmp
        la    s3, same                # which sign to print
        beqz  a0, show
        la    s3, less
        bltz  a0, show
        la    s3, more
show:   li    a7, 4                   # a, the sign, b
        mv    a0, s1
        ecall
        mv    a0, s3
        ecall
        mv    a0, s2
        ecall
        li    a7, 11
        li    a0, '\n'
        ecall
        addi  s0, s0, 8               # the next pair
        j     next
end:    lw    s3, 12(sp)
        lw    s2, 16(sp)
        lw    s1, 20(sp)
        lw    s0, 24(sp)
        lw    ra, 28(sp)
        addi  sp, sp, 32
        li    a7, 10
        ecall

# strcmp: a0, a1 = two strings -> a0 < 0, 0 or > 0 (a leaf: no frame)
strcmp: lbu   t0, 0(a0)               # a character of each
        lbu   t1, 0(a1)
        bne   t0, t1, differ          # different: they decide
        beqz  t0, differ              # both ended: equal (0 - 0)
        addi  a0, a0, 1
        addi  a1, a1, 1
        j     strcmp
differ: sub   a0, t0, t1
        ret
