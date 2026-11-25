# MIPS by Example in eMIPS, episode 15 — First decisions: the largest of
# three, even and odd.  max3 decides with two signed comparisons (slt);
# odd looks at bit 0 alone (andi 1), which works for negative numbers too.
        .data
ask:    .asciiz "Three numbers:\n"
large:  .asciiz "largest: "
is:     .asciiz " is "
even:   .asciiz "even\n"
oddw:   .asciiz "odd\n"
        .text
main:   addiu $sp, $sp, -40           # main's frame: 36 $ra, 32 $fp,
        sw    $ra, 36($sp)            # 28-20 $s0-$s2, 0-15 the home area
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        move  $fp, $sp
        li    $v0, 4
        la    $a0, ask
        syscall
        li    $v0, 5                  # read_int, three times
        syscall
        move  $s0, $v0
        li    $v0, 5
        syscall
        move  $s1, $v0
        li    $v0, 5
        syscall
        move  $s2, $v0
        move  $a0, $s0                # max3(a, b, c)
        move  $a1, $s1
        move  $a2, $s2
        jal   max3
        move  $t0, $v0
        li    $v0, 4
        la    $a0, large
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $a0, $s0                # even or odd, each
        jal   parity
        move  $a0, $s1
        jal   parity
        move  $a0, $s2
        jal   parity
        move  $sp, $fp                # main's epilogue
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        li    $v0, 10
        syscall

# max3: $a0, $a1, $a2 -> $v0 = the largest (a leaf: two decisions)
max3:   move  $v0, $a0                # the best so far: a
        slt   $t0, $v0, $a1           # b larger?
        beqz  $t0, check_c
        move  $v0, $a1
check_c:
        slt   $t0, $v0, $a2           # c larger?
        beqz  $t0, m3done
        move  $v0, $a2
m3done: jr    $ra

# parity: print "n is even" or "n is odd" for $a0 (it calls odd: a frame;
# n kept in the caller's home area)
parity: addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        sw    $a0, 24($fp)            # n, in the caller's home area
        jal   odd
        move  $t1, $v0
        li    $v0, 1                  # n ...
        lw    $a0, 24($fp)
        syscall
        li    $v0, 4                  # ... is ...
        la    $a0, is
        syscall
        la    $a0, even               # ... even or odd
        beqz  $t1, psay
        la    $a0, oddw
psay:   syscall
        move  $sp, $fp
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        jr    $ra

# odd: $a0 -> $v0 = 1 if it is odd, 0 if even: bit 0 (a leaf)
odd:    andi  $v0, $a0, 1
        jr    $ra
