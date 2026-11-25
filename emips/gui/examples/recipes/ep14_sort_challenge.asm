# MIPS by Example in eMIPS, episode 14, the challenge solved: isort takes
# the comparison as a third argument, a function's address in $a2, and
# calls it with jalr.  strcasecmp (episode 4's challenge) sorts ignoring
# case; bylen compares lengths.  An insertion sort is stable: names that
# compare equal keep their order (apple before Apple; pear before MIPS).
        .data
pear:   .asciiz "pear"
apple:  .asciiz "apple"
mips:   .asciiz "MIPS"
zebra:  .asciiz "zebra"
banana: .asciiz "banana"
apple2: .asciiz "Apple"
        .align 2
names:  .word pear, apple, mips, zebra, banana, apple2
copy:   .word pear, apple, mips, zebra, banana, apple2
nocase: .asciiz "ignoring case:"
bysize: .asciiz "by length:    "
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        la    $a0, names              # isort(names, 6, strcasecmp)
        li    $a1, 6
        la    $a2, strcasecmp
        jal   isort
        li    $v0, 4
        la    $a0, nocase
        syscall
        la    $a0, names
        li    $a1, 6
        jal   show
        la    $a0, copy               # isort(copy, 6, bylen)
        li    $a1, 6
        la    $a2, bylen
        jal   isort
        li    $v0, 4
        la    $a0, bysize
        syscall
        la    $a0, copy
        li    $a1, 6
        jal   show
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# show: episode 14's (a leaf)
show:   move  $t0, $a0
shnext: beqz  $a1, shdone
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 4
        lw    $a0, 0($t0)
        syscall
        addiu $t0, $t0, 4
        addiu $a1, $a1, -1
        b     shnext
shdone: li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra

# isort: episode 14's, comparing with the function at $a2 (it calls it: a
# frame; $s0 = the array, $s1 = n, $s2 = i, $s3 = the key, $s4 = j, $s5 =
# the comparison)
isort:  addiu $sp, $sp, -48           # 44 $ra, 40 $fp, 36-16 $s0-$s5,
        sw    $ra, 44($sp)            # 0-15 the home area
        sw    $fp, 40($sp)
        sw    $s0, 36($sp)
        sw    $s1, 32($sp)
        sw    $s2, 28($sp)
        sw    $s3, 24($sp)
        sw    $s4, 20($sp)
        sw    $s5, 16($sp)
        move  $fp, $sp
        move  $s0, $a0
        move  $s1, $a1
        move  $s5, $a2                # the comparison, kept across the calls
        li    $s2, 1
outer:  bge   $s2, $s1, idone
        sll   $t0, $s2, 2
        addu  $t0, $s0, $t0
        lw    $s3, 0($t0)
        addiu $s4, $s2, -1
inner:  bltz  $s4, place
        sll   $t0, $s4, 2
        addu  $t0, $s0, $t0
        lw    $a0, 0($t0)             # compare(a[j], key)
        move  $a1, $s3
        jalr  $s5                     # a call through a register
        blez  $v0, place
        sll   $t0, $s4, 2
        addu  $t0, $s0, $t0
        lw    $t1, 0($t0)
        sw    $t1, 4($t0)
        addiu $s4, $s4, -1
        b     inner
place:  addiu $t0, $s4, 1
        sll   $t0, $t0, 2
        addu  $t0, $s0, $t0
        sw    $s3, 0($t0)
        addiu $s2, $s2, 1
        b     outer
idone:  move  $sp, $fp
        lw    $s5, 16($sp)
        lw    $s4, 20($sp)
        lw    $s3, 24($sp)
        lw    $s2, 28($sp)
        lw    $s1, 32($sp)
        lw    $s0, 36($sp)
        lw    $fp, 40($sp)
        lw    $ra, 44($sp)
        addiu $sp, $sp, 48
        jr    $ra

# strcasecmp: episode 4's challenge (a leaf)
strcasecmp:
        lbu   $t0, 0($a0)
        lbu   $t1, 0($a1)
        addiu $t2, $t0, -65
        sltiu $t2, $t2, 26
        sll   $t2, $t2, 5
        addu  $t0, $t0, $t2
        addiu $t3, $t1, -65
        sltiu $t3, $t3, 26
        sll   $t3, $t3, 5
        addu  $t1, $t1, $t3
        bne   $t0, $t1, scdiff
        beqz  $t0, scdiff
        addiu $a0, $a0, 1
        addiu $a1, $a1, 1
        b     strcasecmp
scdiff: subu  $v0, $t0, $t1
        jr    $ra

# bylen: $a0, $a1 = two strings -> $v0 = length(a) - length(b) (a leaf)
bylen:  move  $v0, $zero
blena:  lbu   $t0, 0($a0)             # + the length of a
        beqz  $t0, blenb
        addiu $v0, $v0, 1
        addiu $a0, $a0, 1
        b     blena
blenb:  lbu   $t0, 0($a1)             # - the length of b
        beqz  $t0, bldone
        addiu $v0, $v0, -1
        addiu $a1, $a1, 1
        b     blenb
bldone: jr    $ra
