# MIPS by Example in eMIPS, episode 19, the challenge solved.  sum_mult
# without a loop: the multiples of k below the limit are k × (1 + 2 + ... +
# m) with m = (limit - 1) / k, and 1 + ... + m = m (m + 1) / 2.  fizzbuzz:
# Fizz for multiples of 3, Buzz for 5, FizzBuzz for both — the both first.
        .data
limits: .word 10, 1000
below:  .asciiz "below "
colon:  .asciiz ": "
fizz:   .asciiz "Fizz"
buzz:   .asciiz "Buzz"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $s0, limits             # each limit: the sum, by the formula
        li    $s1, 2
lnext:  lw    $a0, 0($s0)
        jal   sum35
        move  $t0, $v0
        li    $v0, 4
        la    $a0, below
        syscall
        li    $v0, 1
        lw    $a0, 0($s0)
        syscall
        li    $v0, 4
        la    $a0, colon
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, lnext
        li    $s0, 1                  # FizzBuzz from 1 to 15
fnext:  move  $a0, $s0
        jal   fizzbuzz
        li    $v0, 11
        li    $a0, ' '
        beq   $s0, 15, fend           # no space after the last
        syscall
        addiu $s0, $s0, 1
        b     fnext
fend:   li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# sum35: $a0 = limit -> $v0 = the multiples of 3 or 5 below it (it calls
# sum_mult: a frame; the limit in $s0, the sum so far in $s1)
sum35:  addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 16 $s1
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        move  $s0, $a0
        li    $a1, 3
        jal   sum_mult
        move  $s1, $v0
        move  $a0, $s0
        li    $a1, 5
        jal   sum_mult
        addu  $s1, $s1, $v0
        move  $a0, $s0
        li    $a1, 15
        jal   sum_mult
        subu  $v0, $s1, $v0
        move  $sp, $fp
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# sum_mult: $a0 = limit, $a1 = k -> $v0 = k × m (m + 1) / 2, m = (limit -
# 1) / k: no loop (a leaf)
sum_mult:
        addiu $t0, $a0, -1
        divu  $t0, $a1
        mflo  $t0                     # m: how many multiples
        addiu $t1, $t0, 1
        mul   $t0, $t0, $t1           # m (m + 1): even, so ...
        srl   $t0, $t0, 1             # ... / 2 is exact
        mul   $v0, $t0, $a1           # times k
        jr    $ra

# fizzbuzz: print Fizz, Buzz, FizzBuzz or $a0 itself (a leaf)
fizzbuzz:
        move  $t0, $a0
        li    $t2, 0                  # printed a word yet?
        li    $t1, 3
        divu  $t0, $t1
        mfhi  $t1
        bnez  $t1, nofizz
        li    $v0, 4
        la    $a0, fizz
        syscall
        li    $t2, 1
nofizz: li    $t1, 5
        divu  $t0, $t1
        mfhi  $t1
        bnez  $t1, nobuzz
        li    $v0, 4
        la    $a0, buzz
        syscall
        li    $t2, 1
nobuzz: bnez  $t2, fbdone             # Fizz, Buzz or both: no number
        li    $v0, 1
        move  $a0, $t0
        syscall
fbdone: jr    $ra
