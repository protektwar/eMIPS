# MIPS by Example in eMIPS, episode 10, the RISC-V ending: the same utoa on
# RV32 — no HI / LO: divu gives the quotient and remu the remainder, each
# into an ordinary register; and neg never traps (no overflow exceptions on
# RISC-V), the result in a0.
# eMIPS: --arch rv32
        .data
numbers:.word 2026, -42, 0
digits: .ascii "0123456789abcdef"
buf:    .space 34                     # 32 binary digits, a sign, the 0
equals: .asciz " = 0b"
hex:    .asciz " = 0x"
        .text
main:   addi  sp, sp, -16             # ra and the s registers main uses
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        sw    s1, 4(sp)
        sw    s2, 0(sp)
        la    s0, numbers
        li    s1, 2                   # two numbers
next:   lw    s2, 0(s0)
        mv    a0, s2
        bgez  s2, positive            # decimal: the sign, then |n|
        li    a7, 11
        li    a0, '-'
        ecall
        neg   a0, s2
positive:
        li    a1, 10
        call  print_base
        li    a7, 4
        la    a0, equals
        ecall
        mv    a0, s2                  # binary: the bits as they are
        li    a1, 2
        call  print_base
        li    a7, 4
        la    a0, hex
        ecall
        mv    a0, s2                  # hex
        li    a1, 16
        call  print_base
        li    a7, 11
        li    a0, '\n'
        ecall
        addi  s0, s0, 4
        addi  s1, s1, -1
        bnez  s1, next
        lw    s2, 0(sp)
        lw    s1, 4(sp)
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        li    a7, 10
        ecall

# print_base: print a0 (unsigned) in base a1 (it calls utoa: ra saved)
print_base:
        addi  sp, sp, -16
        sw    ra, 12(sp)
        la    a2, buf + 33            # the end of the buffer
        call  utoa                    # a0 = the first digit
        li    a7, 4
        ecall
        lw    ra, 12(sp)
        addi  sp, sp, 16
        ret

# utoa: a0 = a number (unsigned), a1 = the base, a2 = the end of a buffer
# -> the digits, ending with a 0 at a2; a0 = the first digit (a leaf)
utoa:   sb    zero, 0(a2)             # the 0 at the end
        la    t1, digits
digit:  remu  t0, a0, a1              # the remainder: the next digit
        divu  a0, a0, a1              # the quotient: two instructions, no HI / LO
        add   t0, t1, t0
        lbu   t0, 0(t0)               # its character
        addi  a2, a2, -1
        sb    t0, 0(a2)
        bnez  a0, digit               # until the quotient is 0
        mv    a0, a2
        ret
