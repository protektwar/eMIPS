# MIPS by Example in eMIPS, episode 23, the challenge solved.  first finds
# the first place of a value that repeats: when a[mid] = x, do not stop —
# remember mid and go on to the left.  isqrt finds the largest r with r × r
# <= n by binary search over the answers 0 .. n.
        .data
reps:   .word 1, 3, 3, 3, 5, 8, 8, 9
firstw: .asciiz "first 3 at "
first8: .asciiz ", first 8 at "
sqrtw:  .asciiz "isqrt("
equals: .asciiz ") = "
sqnums: .word 144, 1000, 2
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $a0, reps               # the first 3
        li    $a1, 8
        li    $a2, 3
        jal   first
        move  $s0, $v0
        la    $a0, reps               # the first 8
        li    $a1, 8
        li    $a2, 8
        jal   first
        move  $s1, $v0
        li    $v0, 4
        la    $a0, firstw
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 4
        la    $a0, first8
        syscall
        li    $v0, 1
        move  $a0, $s1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        la    $s0, sqnums             # three square roots
        li    $s1, 3
qnext:  lw    $a0, 0($s0)
        jal   isqrt
        move  $t0, $v0
        li    $v0, 4
        la    $a0, sqrtw
        syscall
        li    $v0, 1
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
        bnez  $s1, qnext
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# first: $a0 = sorted words, $a1 = n, $a2 = x -> $v0 = the first index of
# x, or -1 (a leaf)
first:  li    $t0, 0                  # lo
        addiu $t1, $a1, -1            # hi
        li    $v0, -1                 # not found yet
floop:  bgt   $t0, $t1, fdone
        addu  $t2, $t0, $t1
        srl   $t2, $t2, 1             # mid
        sll   $t3, $t2, 2
        addu  $t3, $a0, $t3
        lw    $t3, 0($t3)
        blt   $t3, $a2, fright
        bne   $t3, $a2, fleft
        move  $v0, $t2                # x at mid: remember it, look further left
fleft:  addiu $t1, $t2, -1
        b     floop
fright: addiu $t0, $t2, 1
        b     floop
fdone:  jr    $ra

# isqrt: $a0 = n -> $v0 = the largest r with r × r <= n (a leaf)
isqrt:  li    $t0, 0                  # lo: 0 × 0 <= n always
        move  $t1, $a0                # hi
        li    $v0, 0
qloop:  bgt   $t0, $t1, qdone
        addu  $t2, $t0, $t1
        srl   $t2, $t2, 1             # mid: a candidate root
        mul   $t3, $t2, $t2
        bgt   $t3, $a0, qbig
        move  $v0, $t2                # mid × mid <= n: an answer, try larger
        addiu $t0, $t2, 1
        b     qloop
qbig:   addiu $t1, $t2, -1            # too large: try smaller
        b     qloop
qdone:  jr    $ra
