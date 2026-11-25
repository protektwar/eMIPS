# MIPS by Example in eMIPS, episode 4 — Comparing two strings: strcmp
# walks both strings while their characters are equal; the first
# difference (or the end) decides: < 0, 0 or > 0, as C's strcmp.
        .data
apple:  .asciiz "apple"
apply:  .asciiz "apply"
pear:   .asciiz "pear"
peach:  .asciiz "peach"
mips:   .asciiz "MIPS"
zebra:  .asciiz "Zebra"
pairs:  .word apple, apply, pear, peach, mips, mips, zebra, apple, 0
less:   .asciiz " < "
same:   .asciiz " = "
more:   .asciiz " > "
        .text
main:   addiu $sp, $sp, -40           # main's frame: 36 $ra, 32 $fp,
        sw    $ra, 36($sp)            # 28-16 $s0-$s3, 0-15 the home area
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        sw    $s3, 16($sp)
        move  $fp, $sp
        la    $s0, pairs              # the next pair
next:   lw    $s1, 0($s0)             # a (0: no more pairs)
        beqz  $s1, end
        lw    $s2, 4($s0)             # b
        move  $a0, $s1                # strcmp(a, b)
        move  $a1, $s2
        jal   strcmp
        la    $s3, same               # which sign to print
        beqz  $v0, show
        la    $s3, less
        bltz  $v0, show
        la    $s3, more
show:   li    $v0, 4                  # a, the sign, b
        move  $a0, $s1
        syscall
        move  $a0, $s3
        syscall
        move  $a0, $s2
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        addiu $s0, $s0, 8             # the next pair
        b     next
end:    move  $sp, $fp                # main's epilogue
        lw    $s3, 16($sp)
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        li    $v0, 10
        syscall

# strcmp: $a0, $a1 = two strings -> $v0 < 0, 0 or > 0 (a leaf: no frame)
strcmp: lbu   $t0, 0($a0)             # a character of each
        lbu   $t1, 0($a1)
        bne   $t0, $t1, differ        # different: they decide
        beqz  $t0, differ             # both ended: equal (0 - 0)
        addiu $a0, $a0, 1
        addiu $a1, $a1, 1
        b     strcmp
differ: subu  $v0, $t0, $t1
        jr    $ra
