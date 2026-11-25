# MIPS by Example in eMIPS, episode 2, the RISC-V ending: the same program
# on RV32 — eight argument registers (a0-a7: the fifth argument in a4, no
# home area), the result in a0, ra the way back, s0 the frame pointer, sp
# kept a multiple of 16; the callee saves ra, s0 and the s registers it uses.
# eMIPS: --arch rv32
        .data
squares:.asciz "sum of squares = "
fact5:  .asciz "fact(5) = "
fact10: .asciz "fact(10) = "
        .text
main:   addi  sp, sp, -16
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        addi  s0, sp, 16              # s0 = the frame pointer (the frame's top)
        li    a0, 1                   # sum_of_squares(1, 2, 3, 4, 5)
        li    a1, 2
        li    a2, 3
        li    a3, 4
        li    a4, 5                   # the fifth argument: a register too
        call  sum_of_squares
        mv    a1, a0
        la    a0, squares
        call  print_result
        li    a0, 5                   # fact(5)
        call  fact
        mv    a1, a0
        la    a0, fact5
        call  print_result
        li    a0, 10                  # fact(10)
        call  fact
        mv    a1, a0
        la    a0, fact10
        call  print_result
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        li    a7, 10
        ecall

# sum_of_squares: a, b, c, d, e in a0-a4 -> a0
sum_of_squares:
        addi  sp, sp, -32             # ra, s0 (fp), s1, and b-e while square runs
        sw    ra, 28(sp)
        sw    s0, 24(sp)
        sw    s1, 20(sp)
        addi  s0, sp, 32
        sw    a1, -32(s0)
        sw    a2, -28(s0)
        sw    a3, -24(s0)
        sw    a4, -20(s0)
        call  square                  # a * a
        mv    s1, a0
        lw    a0, -32(s0)
        call  square                  # b * b
        add   s1, s1, a0
        lw    a0, -28(s0)
        call  square                  # c * c
        add   s1, s1, a0
        lw    a0, -24(s0)
        call  square                  # d * d
        add   s1, s1, a0
        lw    a0, -20(s0)
        call  square                  # e * e
        add   a0, s1, a0
        lw    s1, 20(sp)
        lw    s0, 24(sp)
        lw    ra, 28(sp)
        addi  sp, sp, 32
        ret

# square: a0 -> a0 * a0 (a leaf)
square: mul   a0, a0, a0
        ret

# fact: a0 = n -> a0 = n!
fact:   addi  sp, sp, -16
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        sw    s1, 4(sp)
        addi  s0, sp, 16
        li    t0, 1
        mv    s1, a0                  # this call's n
        li    a0, 1                   # fact(n <= 1) = 1
        ble   s1, t0, factdone
        addi  a0, s1, -1
        call  fact                    # fact(n - 1)
        mul   a0, s1, a0
factdone:
        lw    s1, 4(sp)
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        ret

# print_result: a0 = a text, a1 = a number (a leaf)
print_result:
        li    a7, 4
        ecall
        li    a7, 1
        mv    a0, a1
        ecall
        li    a7, 11
        li    a0, '\n'
        ecall
        ret
