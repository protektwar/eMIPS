# MIPS by Example in eMIPS, episode 18, the challenge solved.  gcd written
# recursively: gcd(a, b) = a when b is 0, else gcd(b, a % b) — each call a
# frame of its own.  reduce prints a fraction in lowest terms: numerator and
# denominator both divided by their gcd.
        .data
fracs:  .word 84, 126, 1071, 462, 13, 17
slash:  .asciiz "/"
equals: .asciiz " = "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $s0, fracs              # each fraction, reduced
        li    $s1, 3
fnext:  lw    $a0, 0($s0)
        lw    $a1, 4($s0)
        jal   reduce
        addiu $s0, $s0, 8
        addiu $s1, $s1, -1
        bnez  $s1, fnext
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# gcd: $a0, $a1 -> $v0, recursively (it calls itself: a frame each time)
gcd:    addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        move  $v0, $a0                # gcd(a, 0) = a
        beqz  $a1, gdone
        divu  $a0, $a1
        move  $a0, $a1                # gcd(b, a % b)
        mfhi  $a1
        jal   gcd
gdone:  move  $sp, $fp
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        jr    $ra

# reduce: print "n/d = n'/d'" in lowest terms (it calls gcd: a frame; n and
# d kept in $s0 and $s1)
reduce: addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 16 $s1
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        move  $s0, $a0
        move  $s1, $a1
        jal   print_frac              # n/d
        li    $v0, 4
        la    $a0, equals
        syscall
        move  $a0, $s0
        move  $a1, $s1
        jal   gcd
        divu  $s0, $v0                # n / g
        mflo  $a0
        divu  $s1, $v0                # d / g
        mflo  $a1
        jal   print_frac
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# print_frac: print "$a0/$a1" (a leaf)
print_frac:
        li    $v0, 1
        syscall
        li    $v0, 4
        la    $a0, slash
        syscall
        li    $v0, 1
        move  $a0, $a1
        syscall
        jr    $ra
