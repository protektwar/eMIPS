# MIPS by Example in eMIPS, episode 18 — GCD and the Sieve of Eratosthenes.
# gcd is Euclid's: replace (a, b) by (b, a % b) until b is 0.  lcm divides
# before it multiplies.  sieve crosses out the multiples of each prime in a
# byte array: what is never crossed out is prime.
        .data
flags:  .space 100                    # flags[i] = 1: i is not prime
gcdw:   .asciiz "gcd("
lcmw:   .asciiz "lcm("
comma:  .asciiz ", "
equals: .asciiz ") = "
primes: .asciiz "primes below 100:"
count:  .asciiz " primes\n"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        li    $a0, 1071               # gcd(1071, 462)
        li    $a1, 462
        jal   gcd
        la    $a0, gcdw
        li    $a1, 1071
        li    $a2, 462
        move  $a3, $v0
        jal   print_call
        li    $a0, 12                 # lcm(12, 18)
        li    $a1, 18
        jal   lcm
        la    $a0, lcmw
        li    $a1, 12
        li    $a2, 18
        move  $a3, $v0
        jal   print_call
        la    $a0, flags              # the sieve
        li    $a1, 100
        jal   sieve
        li    $v0, 4
        la    $a0, primes
        syscall
        li    $s0, 2                  # print every i not crossed out
        li    $s1, 0                  # and count them
pnext:  la    $t0, flags
        addu  $t0, $t0, $s0
        lbu   $t0, 0($t0)
        bnez  $t0, pskip
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        addiu $s1, $s1, 1
pskip:  addiu $s0, $s0, 1
        blt   $s0, 100, pnext
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $v0, 1
        move  $a0, $s1
        syscall
        li    $v0, 4
        la    $a0, count
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# gcd: $a0, $a1 -> $v0 = their greatest common divisor (a leaf)
gcd:    beqz  $a1, gdone              # gcd(a, 0) = a
        divu  $a0, $a1                # a % b in HI
        move  $a0, $a1                # (a, b) <- (b, a % b)
        mfhi  $a1
        b     gcd
gdone:  move  $v0, $a0
        jr    $ra

# lcm: $a0, $a1 -> $v0 = a / gcd(a, b) * b (it calls gcd: a frame; a and b
# wait in the caller's home area)
lcm:    addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        sw    $a0, 24($fp)            # a and b, in the caller's home area
        sw    $a1, 28($fp)
        jal   gcd
        lw    $t0, 24($fp)
        divu  $t0, $v0                # a / gcd first: no overflow
        mflo  $t0
        lw    $t1, 28($fp)
        mul   $v0, $t0, $t1           # ... then times b
        move  $sp, $fp
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        jr    $ra

# sieve: $a0 = flags (bytes, 0 at the start), $a1 = n: cross out every
# i < n that is not prime (a leaf)
sieve:  li    $t0, 1
        sb    $t0, 0($a0)             # 0 and 1 are not prime
        sb    $t0, 1($a0)
        li    $t1, 2                  # i
inext:  mul   $t2, $t1, $t1           # j starts at i * i
        bge   $t2, $a1, sdone         # i * i >= n: nothing left to cross out
        addu  $t3, $a0, $t1
        lbu   $t3, 0($t3)
        bnez  $t3, iskip              # i is crossed out: its multiples are too
jnext:  addu  $t3, $a0, $t2
        sb    $t0, 0($t3)             # cross out j
        addu  $t2, $t2, $t1           # the next multiple of i
        blt   $t2, $a1, jnext
iskip:  addiu $t1, $t1, 1
        b     inext
sdone:  jr    $ra

# print_call: print "NAME(a, b) = r": $a0 the text "NAME(", $a1 a, $a2 b,
# $a3 r (a leaf)
print_call:
        li    $v0, 4
        syscall
        li    $v0, 1
        move  $a0, $a1
        syscall
        li    $v0, 4
        la    $a0, comma
        syscall
        li    $v0, 1
        move  $a0, $a2
        syscall
        li    $v0, 4
        la    $a0, equals
        syscall
        li    $v0, 1
        move  $a0, $a3
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra
