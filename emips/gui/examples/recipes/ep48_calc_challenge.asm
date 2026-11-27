# MIPS by Example in eMIPS, episode 48 — the challenge, solved: % and a
# division by zero.  % is RISC-V's rem, at term's level, as / is.  RISC-V's
# div never traps: x / 0 is -1 (all ones) and x % 0 is x, by the
# specification — so the calculator tests the divisor itself, sets err, and
# main prints an error instead of a value.  (The episode: a whole program in
# RISC-V, a calculator.)
# eMIPS: --arch rv32
        .data
inputs: .word e1, e2, e3, e4, e5, 0   # the expressions, 0 at the end
e1:     .asciz "17%5+1"
e2:     .asciz "2*(10%4)"
e3:     .asciz "7/(3-3)"
e4:     .asciz "100-9%0"
e5:     .asciz "(1+2)*(3+4)%5"
equals: .asciz " = "
divzero: .asciz "error: division by zero"
        .align 2
pos:    .word 0                       # the next character to read
err:    .word 0                       # 1: a division by zero happened

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
        sw    zero, err, t0
        li    a7, 4                   # "2+3*4 = "
        ecall
        la    a0, equals
        li    a7, 4
        ecall
        call  expr                    # a0 = its value
        lw    t0, err
        bnez  t0, bad
        li    a7, 1
        ecall
        j     nl
bad:    la    a0, divzero
        li    a7, 4
        ecall
nl:
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

# term() -> a0: factor { (* | / | %) factor }
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
        beq   a0, t0, top
        li    t0, '%'
        bne   a0, t0, tend
top:    mv    s2, a0                  # * / or %: the next factor
        call  skip
        call  factor
        li    t0, '*'
        bne   s2, t0, tdiv
        mul   s1, s1, a0              # M: a multiply instruction
        j     tloop
tdiv:   bnez  a0, tok                 # / or % by 0: RISC-V would not complain
        li    t0, 1
        sw    t0, err, t1
        j     tloop
tok:    li    t0, '/'
        bne   s2, t0, trem
        div   s1, s1, a0              # a divide instruction
        j     tloop
trem:   rem   s1, s1, a0              # and the remainder
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
