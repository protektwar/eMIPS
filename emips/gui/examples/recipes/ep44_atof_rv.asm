# MIPS by Example in eMIPS, episode 44, the RISC-V ending: text to a float on
# RV32 with F.  fcvt.s.w converts an integer register straight into a float
# register (MIPS needs mtc1, then cvt.s.w); the float compares and branches
# are not needed here; print_float is RARS's service 2, from fa0.
# eMIPS: --arch rv32
        .data
inputs: .word in1, in2, in3, in4, in5, 0   # the texts, 0 at the end
in1:    .asciz "3.14159"
in2:    .asciz "-0.5"
in3:    .asciz "100"
in4:    .asciz "2.75"
in5:    .asciz "0.1"
arrow:  .asciz " -> "
equals: .asciz " = "

        .text
main:   addi  sp, sp, -16             # 12 ra, 8 s0, 4 s1, 0 s2
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        sw    s1, 4(sp)
        sw    s2, 0(sp)
        addi  s0, sp, 16              # s0 = the frame pointer
        la    s1, inputs              # s1: the next text's address in the list
next:   lw    s2, 0(s1)               # s2: the text
        beqz  s2, done
        mv    a0, s2                  # "3.14159 -> "
        li    a7, 4
        ecall
        la    a0, arrow
        li    a7, 4
        ecall
        mv    a0, s2                  # fa0 = parse_float(text)
        call  parse_float
        li    a7, 2                   # the float, as a float
        ecall
        la    a0, equals
        li    a7, 4
        ecall
        fmv.x.w a0, fa0               # its bits, into an integer register
        li    a7, 34
        ecall
        li    a0, '\n'
        li    a7, 11
        ecall
        addi  s1, s1, 4
        j     next
done:   lw    s2, 0(sp)
        lw    s1, 4(sp)
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        li    a7, 10
        ecall

# parse_float(text) -> fa0: [-]digits[.digits] (a leaf)
#   t0 the digits as an integer, t1 the digits after the point,
#   t2 1 after the point, t3 1 for a minus sign
parse_float:
        li    t0, 0
        li    t1, 0
        li    t2, 0
        li    t3, 0
        lbu   t4, 0(a0)
        li    t5, '-'
        bne   t4, t5, digits          # a sign?
        li    t3, 1
        addi  a0, a0, 1
digits: lbu   t4, 0(a0)
        li    t5, '.'
        bne   t4, t5, digit
        li    t2, 1                   # the point: count the digits after it
        addi  a0, a0, 1
        j     digits
digit:  addi  t4, t4, -48             # '0' .. '9' → 0 .. 9
        li    t5, 10
        bgeu  t4, t5, end             # not a digit: the end
        mul   t0, t0, t5              # the integer: × 10 + the digit
        add   t0, t0, t4
        add   t1, t1, t2              # one more after the point (if past it)
        addi  a0, a0, 1
        j     digits
end:    fcvt.s.w fa0, t0              # the integer, made a float: one instruction
        li    t4, 1
        fcvt.s.w ft0, t4              # 10 ^ (digits after the point)
        li    t4, 10
        fcvt.s.w ft1, t4
power:  beqz  t1, scale
        fmul.s ft0, ft0, ft1          # exact: 10.0 × 10.0 ... up to 10^10
        addi  t1, t1, -1
        j     power
scale:  fdiv.s fa0, fa0, ft0          # one rounding: the nearest float
        beqz  t3, pdone
        fneg.s fa0, fa0               # the sign
pdone:  ret
