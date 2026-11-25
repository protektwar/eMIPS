# MIPS by Example in eMIPS, episode 2, the challenge solved.  fib(n) =
# fib(n - 1) + fib(n - 2) calls itself twice: between the calls it must keep
# n (for n - 2) and, after the second, fib(n - 1) — two values that must
# survive a call: $s0 and $s1, both saved first (they are the caller's).
# sum6 has six arguments: a-d in $a0-$a3, e and f in the caller's frame at
# 16 and 20.  sum6 is a leaf without a frame, so $sp is still the caller's:
# e is at 16($sp), f at 20($sp).
        .data
fibs:   .asciiz "fib:"
sum6s:  .asciiz "sum6(1, 2, 3, 4, 5, 6) = "
        .text
main:   addiu $sp, $sp, -40           # main's frame: 36 $ra, 32 $fp, 28 $s0,
        sw    $ra, 36($sp)            # 0-23 the arguments of what it calls:
        sw    $fp, 32($sp)            # six words, for sum6
        sw    $s0, 28($sp)
        move  $fp, $sp
        li    $v0, 4                  # "fib:"
        la    $a0, fibs
        syscall
        li    $s0, 0                  # $s0 = i, kept across the calls
loop:   li    $v0, 11                 # ' '
        li    $a0, ' '
        syscall
        move  $a0, $s0                # fib(i)
        jal   fib
        move  $a0, $v0
        li    $v0, 1
        syscall
        addiu $s0, $s0, 1
        ble   $s0, 10, loop
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $t0, 5                  # sum6(1, 2, 3, 4, 5, 6): e and f
        sw    $t0, 16($sp)            # at their places, 16 and 20
        li    $t0, 6
        sw    $t0, 20($sp)
        li    $a0, 1
        li    $a1, 2
        li    $a2, 3
        li    $a3, 4
        jal   sum6
        move  $s0, $v0
        li    $v0, 4
        la    $a0, sum6s
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        li    $v0, 10                 # exit
        syscall

# fib: $a0 = n -> $v0 = fib(n) (recursive: two calls, a frame)
fib:    addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 16 $s1,
        sw    $ra, 28($sp)            # 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        move  $v0, $a0                # fib(0) = 0, fib(1) = 1
        ble   $a0, 1, fibdone
        move  $s0, $a0                # n, needed for the second call
        addiu $a0, $a0, -1
        jal   fib                     # fib(n - 1)
        move  $s1, $v0                # kept across the second call
        addiu $a0, $s0, -2
        jal   fib                     # fib(n - 2)
        addu  $v0, $s1, $v0
fibdone:
        move  $sp, $fp
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# sum6: a-d in $a0-$a3, e and f at 16 and 20 in the caller's frame -> $v0
# (a leaf: no frame, so $sp is still the caller's)
sum6:   addu  $v0, $a0, $a1
        addu  $v0, $v0, $a2
        addu  $v0, $v0, $a3
        lw    $t0, 16($sp)            # e
        addu  $v0, $v0, $t0
        lw    $t0, 20($sp)            # f: the sixth argument
        addu  $v0, $v0, $t0
        jr    $ra
