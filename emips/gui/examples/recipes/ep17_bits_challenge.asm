# MIPS by Example in eMIPS, episode 17, the challenge solved.  umul
# multiplies without mul: for each 1 bit of b, add a shifted copy of a — the
# school method in binary.  pow2 says if n is a power of 2: exactly one 1
# bit, so n & (n - 1) is 0 (and n is not 0).
        .data
pairs:  .word 123, 45, 1000, 1000
nums:   .word 64, 96, 1, 0
times:  .asciiz " * "
equals: .asciiz " = "
yes:    .asciiz " is a power of 2\n"
no:     .asciiz " is not a power of 2\n"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $s0, pairs              # two products without mul
        li    $s1, 2
mnext:  lw    $a0, 0($s0)
        lw    $a1, 4($s0)
        jal   umul
        move  $t2, $v0
        li    $v0, 1
        lw    $a0, 0($s0)
        syscall
        li    $v0, 4
        la    $a0, times
        syscall
        li    $v0, 1
        lw    $a0, 4($s0)
        syscall
        li    $v0, 4
        la    $a0, equals
        syscall
        li    $v0, 1
        move  $a0, $t2
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        addiu $s0, $s0, 8
        addiu $s1, $s1, -1
        bnez  $s1, mnext
        la    $s0, nums               # powers of 2?
        li    $s1, 4
pnext:  lw    $a0, 0($s0)
        jal   pow2
        move  $t2, $v0
        li    $v0, 1
        lw    $a0, 0($s0)
        syscall
        li    $v0, 4
        la    $a0, yes
        bnez  $t2, psay
        la    $a0, no
psay:   syscall
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, pnext
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# umul: $a0 * $a1 -> $v0 (the low 32 bits): shift and add (a leaf)
umul:   li    $v0, 0
mloop:  beqz  $a1, mdone              # no 1 bits of b left
        andi  $t0, $a1, 1
        beqz  $t0, mskip
        addu  $v0, $v0, $a0           # this bit of b is 1: add a, shifted
mskip:  sll   $a0, $a0, 1             # a for the next bit: twice as much
        srl   $a1, $a1, 1
        b     mloop
mdone:  jr    $ra

# pow2: $a0 -> $v0 = 1 if it is a power of 2 (a leaf)
pow2:   li    $v0, 0
        beqz  $a0, pdone              # 0 has no 1 bit
        addiu $t0, $a0, -1
        and   $t0, $a0, $t0           # the lowest 1 cleared: 0 if it was the only one
        sltiu $v0, $t0, 1             # $v0 = ($t0 == 0)
pdone:  jr    $ra
