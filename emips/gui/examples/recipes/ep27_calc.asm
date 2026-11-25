# MIPS by Example in eMIPS, episode 27 — A calculator: jump tables and
# function pointers.  A switch on the operator becomes a table of addresses
# indexed by the character: '%' (37) is entry 0, '/' (47) entry 10.  The
# index is checked first (sltiu), then jr jumps to the case.  The cases
# between are filled with the default.
        .data
exprs:  .word 12, '+', 30
        .word 100, '-', 58
        .word 6, '*', 7
        .word 84, '/', 2
        .word 47, '%', 5
        .word 3, '^', 4
cases:  .word c_mod, c_bad, c_bad, c_bad, c_bad        # '%'  &  '  (  )
        .word c_mul, c_add, c_bad, c_sub, c_bad, c_div # '*'  +  ,  -  .  /
equals: .asciiz " = "
unknown:.asciiz ": unknown operator\n"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $s0, exprs
        li    $s1, 6
enext:  lw    $a0, 0($s0)             # calc(a, op, b)
        lw    $a1, 4($s0)
        lw    $a2, 8($s0)
        jal   calc
        move  $t0, $v0                # the result ...
        move  $t1, $v1                # ... and whether op was known
        li    $v0, 1                  # "a op b"
        lw    $a0, 0($s0)
        syscall
        li    $v0, 11
        li    $a0, ' '
        syscall
        lw    $a0, 4($s0)
        syscall
        li    $a0, ' '
        syscall
        li    $v0, 1
        lw    $a0, 8($s0)
        syscall
        beqz  $t1, ebad
        li    $v0, 4                  # " = result"
        la    $a0, equals
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        b     estep
ebad:   li    $v0, 4
        la    $a0, unknown
        syscall
estep:  addiu $s0, $s0, 12
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

# calc: $a0 = a, $a1 = an operator character, $a2 = b -> $v0 = a op b,
# $v1 = 1, or $v1 = 0 for an operator it does not know (a leaf)
calc:   li    $v1, 1
        addiu $t0, $a1, -'%'          # the index: '%' is 0
        sltiu $t1, $t0, 11            # 0 ... 10? (unsigned: below '%' is huge)
        beqz  $t1, c_bad
        sll   $t0, $t0, 2             # a word each
        la    $t1, cases
        addu  $t0, $t1, $t0
        lw    $t0, 0($t0)             # the case's address
        jr    $t0                     # the switch: one jump
c_add:  addu  $v0, $a0, $a2
        jr    $ra
c_sub:  subu  $v0, $a0, $a2
        jr    $ra
c_mul:  mul   $v0, $a0, $a2
        jr    $ra
c_div:  div   $a0, $a2
        mflo  $v0
        jr    $ra
c_mod:  div   $a0, $a2
        mfhi  $v0
        jr    $ra
c_bad:  li    $v1, 0                  # the default
        li    $v0, 0
        jr    $ra
