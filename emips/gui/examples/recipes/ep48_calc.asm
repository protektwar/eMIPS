# MIPS by Example in eMIPS, episode 48 — A whole program in RISC-V: a
# calculator.  "2+3*4" is 14, not 20: * binds tighter than +.  Recursive
# descent follows the grammar, one function a rule:
#   expr   = term { (+ | -) term }
#   term   = factor { (* | /) factor }
#   factor = number | ( expr ) | - factor
# expr calls term, term calls factor, and factor calls expr again for a
# parenthesis: the recursion is the nesting.  pos, in memory, is where the
# text is read next.  RV32 with the M extension: mul and div are
# instructions; a0-a7 carry the arguments, ra the return address, s0 is the
# frame pointer, frames are multiples of 16, and ecall's services are RARS's.
# eMIPS: --arch rv32
        .data
inputs: .word e1, e2, e3, e4, e5, 0   # the expressions, 0 at the end
e1:     .asciz "2+3*4"
e2:     .asciz "(2+3)*4"
e3:     .asciz "100/7-2*3"
e4:     .asciz "-5+20"
e5:     .asciz "(1+2)*(3+4)-(5-6)"
equals: .asciz " = "
        .align 2
pos:    .word 0                       # the next character to read

        .text
main:   addi  sp, sp, -16             # 12 ra, 8 s0, 4 s1
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        sw    s1, 4(sp)
        addi  s0, sp, 16              # s0 = the frame pointer
        la    s1, inputs              # s1: the next expression's address
next:   lw    a0, 0(s1)
        beqz  a0, done
        sw    a0, pos, t0             # read it from its start
        li    a7, 4                   # "2+3*4 = "
        ecall
        la    a0, equals
        li    a7, 4
        ecall
        call  expr                    # a0 = its value
        li    a7, 1
        ecall
        li    a0, '\n'
        li    a7, 11
        ecall
        addi  s1, s1, 4
        j     next
done:   lw    s1, 4(sp)
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        li    a7, 10
        ecall

# peek() -> a0: the character at pos (a leaf)
peek:   lw    t0, pos
        lbu   a0, 0(t0)
        ret

# skip(): pos one character on (a leaf)
skip:   lw    t0, pos
        addi  t0, t0, 1
        sw    t0, pos, t1
        ret

# expr() -> a0: term { (+ | -) term }
expr:   addi  sp, sp, -16             # 12 ra, 8 s0, 4 s1 (the value), 0 s2 (the operator)
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        sw    s1, 4(sp)
        sw    s2, 0(sp)
        addi  s0, sp, 16
        call  term
        mv    s1, a0
eloop:  call  peek
        li    t0, '+'
        beq   a0, t0, eop
        li    t0, '-'
        bne   a0, t0, eend
eop:    mv    s2, a0                  # + or -: the next term
        call  skip
        call  term
        li    t0, '+'
        bne   s2, t0, esub
        add   s1, s1, a0
        j     eloop
esub:   sub   s1, s1, a0
        j     eloop
eend:   mv    a0, s1
        lw    s2, 0(sp)
        lw    s1, 4(sp)
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        ret

# term() -> a0: factor { (* | /) factor }
term:   addi  sp, sp, -16             # 12 ra, 8 s0, 4 s1 (the value), 0 s2 (the operator)
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        sw    s1, 4(sp)
        sw    s2, 0(sp)
        addi  s0, sp, 16
        call  factor
        mv    s1, a0
tloop:  call  peek
        li    t0, '*'
        beq   a0, t0, top
        li    t0, '/'
        bne   a0, t0, tend
top:    mv    s2, a0                  # * or /: the next factor
        call  skip
        call  factor
        li    t0, '*'
        bne   s2, t0, tdiv
        mul   s1, s1, a0              # M: a multiply instruction
        j     tloop
tdiv:   div   s1, s1, a0              # and a divide one
        j     tloop
tend:   mv    a0, s1
        lw    s2, 0(sp)
        lw    s1, 4(sp)
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        ret

# factor() -> a0: number | ( expr ) | - factor
factor: addi  sp, sp, -16             # 12 ra, 8 s0
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        addi  s0, sp, 16
        call  peek
        li    t0, '('
        bne   a0, t0, fneg
        call  skip                    # ( expr ): the recursion
        call  expr
        sw    a0, 4(sp)               # (kept while skip runs)
        call  skip                    # the )
        lw    a0, 4(sp)
        j     fend
fneg:   li    t0, '-'
        bne   a0, t0, fnum
        call  skip                    # - factor
        call  factor
        neg   a0, a0
        j     fend
fnum:   li    a0, 0                   # digits: a0 = a0 × 10 + the digit
        lw    t0, pos
fdig:   lbu   t1, 0(t0)
        addi  t1, t1, -48             # '0' .. '9' → 0 .. 9; anything else is out
        li    t2, 10
        bgeu  t1, t2, fdone           # (unsigned: below '0' wraps round, too big)
        mul   a0, a0, t2
        add   a0, a0, t1
        addi  t0, t0, 1
        j     fdig
fdone:  sw    t0, pos, t1
fend:   lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        ret
