# MIPS by Example in eMIPS, episode 17 — Division without div; counting 1
# bits three ways.  udiv is long division in binary: bring down one bit of n
# at a time; if the remainder reaches d, subtract it and write a 1 in the
# quotient.  The ones are counted bit by bit, by clearing the lowest 1
# (n & (n - 1)), and with a table of the 16 four-bit numbers.
        .data
pairs:  .word 1000, 7, 2026, 10
nums:   .word 2026, -1
bits4:  .byte 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
over:   .asciiz " / "
equals: .asciiz " = "
rem:    .asciiz " remainder "
ones:   .asciiz "ones in "
colon:  .asciiz ":"
        .text
main:   addiu $sp, $sp, -40           # main's frame: 36 $ra, 32 $fp,
        sw    $ra, 36($sp)            # 28-20 $s0-$s2, 0-15 the home area
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        move  $fp, $sp
        la    $s0, pairs              # two divisions without div
        li    $s1, 2
dnext:  lw    $a0, 0($s0)
        lw    $a1, 4($s0)
        jal   udiv
        lw    $a0, 0($s0)             # print_div(n, d, q, r)
        lw    $a1, 4($s0)
        move  $a2, $v0
        move  $a3, $v1
        jal   print_div
        addiu $s0, $s0, 8
        addiu $s1, $s1, -1
        bnez  $s1, dnext
        la    $s0, nums               # the ones, three ways
        li    $s1, 2
onext:  li    $v0, 4
        la    $a0, ones
        syscall
        li    $v0, 1
        lw    $a0, 0($s0)
        syscall
        li    $v0, 4
        la    $a0, colon
        syscall
        lw    $a0, 0($s0)
        jal   ones_loop
        move  $a0, $v0
        jal   print_count
        lw    $a0, 0($s0)
        jal   ones_clear
        move  $a0, $v0
        jal   print_count
        lw    $a0, 0($s0)
        jal   ones_table
        move  $a0, $v0
        jal   print_count
        li    $v0, 11
        li    $a0, '\n'
        syscall
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, onext
        move  $sp, $fp                # main's epilogue
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        li    $v0, 10
        syscall

# udiv: $a0 = n, $a1 = d (unsigned) -> $v0 = n / d, $v1 = n % d: long
# division, one bit a step, 32 steps (a leaf)
udiv:   li    $v0, 0                  # the quotient
        li    $v1, 0                  # the remainder
        li    $t0, 32
ustep:  sll   $v1, $v1, 1             # bring down the next bit of n
        srl   $t1, $a0, 31
        or    $v1, $v1, $t1
        sll   $a0, $a0, 1
        sll   $v0, $v0, 1             # make room for the next quotient bit
        sltu  $t1, $v1, $a1           # remainder < d: that bit is 0
        bnez  $t1, unext
        subu  $v1, $v1, $a1           # else subtract d: that bit is 1
        ori   $v0, $v0, 1
unext:  addiu $t0, $t0, -1
        bnez  $t0, ustep
        jr    $ra

# ones_loop: $a0 -> $v0 = its 1 bits, one bit a step until no 1 is left
# (a leaf)
ones_loop:
        li    $v0, 0
oloop:  beqz  $a0, odone
        andi  $t0, $a0, 1             # the lowest bit
        addu  $v0, $v0, $t0
        srl   $a0, $a0, 1             # the next one down: shift in a 0
        b     oloop
odone:  jr    $ra

# ones_clear: $a0 -> $v0 = its 1 bits; n & (n - 1) clears the lowest 1,
# so the loop turns once per 1 (a leaf)
ones_clear:
        li    $v0, 0
cloop:  beqz  $a0, cdone
        addiu $t0, $a0, -1
        and   $a0, $a0, $t0           # the lowest 1 cleared
        addiu $v0, $v0, 1
        b     cloop
cdone:  jr    $ra

# ones_table: $a0 -> $v0 = its 1 bits, four at a time from the table of
# the 16 four-bit numbers: 8 steps always (a leaf)
ones_table:
        li    $v0, 0
        la    $t1, bits4
        li    $t2, 8
tloop:  andi  $t0, $a0, 15            # the low four bits
        addu  $t0, $t1, $t0
        lbu   $t0, 0($t0)             # their ones, from the table
        addu  $v0, $v0, $t0
        srl   $a0, $a0, 4
        addiu $t2, $t2, -1
        bnez  $t2, tloop
        jr    $ra

# print_div: print "n / d = q remainder r" from $a0-$a3 (a leaf)
print_div:
        li    $v0, 1
        syscall
        li    $v0, 4
        la    $a0, over
        syscall
        li    $v0, 1
        move  $a0, $a1
        syscall
        li    $v0, 4
        la    $a0, equals
        syscall
        li    $v0, 1
        move  $a0, $a2
        syscall
        li    $v0, 4
        la    $a0, rem
        syscall
        li    $v0, 1
        move  $a0, $a3
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra

# print_count: print a space and $a0 (a leaf)
print_count:
        move  $t0, $a0
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        jr    $ra
