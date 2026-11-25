# MIPS by Example in eMIPS, episode 26, the challenge solved.  postfix
# evaluates an expression written after its operands — "3 4 + 2 *" — with
# the stack: a digit is pushed, an operator pops two and pushes the result.
# The last value left is the answer.  Single digits, + - *.
        .data
e1:     .asciiz "3 4 + 2 *"
e2:     .asciiz "5 1 2 + 4 * + 3 -"
e3:     .asciiz "9 2 - 3 *"
exprs:  .word e1, e2, e3
equals: .asciiz " = "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $s0, exprs
        li    $s1, 3
enext:  lw    $a0, 0($s0)
        jal   postfix
        move  $t0, $v0
        li    $v0, 4
        lw    $a0, 0($s0)
        syscall
        li    $v0, 4
        la    $a0, equals
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, enext
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# postfix: $a0 = a postfix expression -> $v0 = its value; the operands on
# the stack below the frame
postfix:
        addiu $sp, $sp, -24           # the usual frame: 20 $ra, 16 $fp
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
pnext:  lbu   $t0, 0($a0)
        beqz  $t0, pend
        addiu $a0, $a0, 1
        beq   $t0, ' ', pnext         # spaces separate
        addiu $t1, $t0, -'0'
        sltiu $t2, $t1, 10            # a digit?
        beqz  $t2, pop2
        addiu $sp, $sp, -4            # push the number
        sw    $t1, 0($sp)
        b     pnext
pop2:   lw    $t3, 0($sp)             # the right operand: pushed last
        lw    $t2, 4($sp)             # the left one
        addiu $sp, $sp, 4             # two off, one back on: one word up
        bne   $t0, '+', pminus
        addu  $t2, $t2, $t3
        b     ppush
pminus: bne   $t0, '-', ptimes
        subu  $t2, $t2, $t3
        b     ppush
ptimes: mul   $t2, $t2, $t3
ppush:  sw    $t2, 0($sp)             # the result replaces them
        b     pnext
pend:   lw    $v0, 0($sp)             # the last value: the answer
        move  $sp, $fp
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        jr    $ra
