# MIPS by Example in eMIPS, episode 27, the challenge solved.  reduce
# combines an array into one value with any two-argument function: the
# function's address is an argument, called with jalr.  The same loop gives
# the sum, the product and the largest, depending on what it is handed.
        .data
nums:   .word 3, 1, 4, 1, 5
jobs:   .word sumw, add2, 0           # a name, a function, a start value
        .word prodw, mul2, 1
        .word maxw, max2, -2147483648
sumw:   .asciiz "sum "
prodw:  .asciiz "product "
maxw:   .asciiz "largest "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $s0, jobs
        li    $s1, 3
jnext:  la    $a0, nums               # reduce(nums, 5, fn, start)
        li    $a1, 5
        lw    $a2, 4($s0)             # the function: just an address
        lw    $a3, 8($s0)
        jal   reduce
        move  $t0, $v0
        li    $v0, 4
        lw    $a0, 0($s0)
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        addiu $s0, $s0, 12
        addiu $s1, $s1, -1
        bnez  $s1, jnext
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# reduce: $a0 = words, $a1 = n, $a2 = a function (a, b) -> a value, $a3 =
# the start -> $v0 = fn(... fn(fn(start, x0), x1) ..., xn-1); it calls fn:
# a frame, the pointer, the count, the function and the value in $s0-$s3
reduce: addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28-16 $s0-$s3
        sw    $ra, 36($sp)
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        sw    $s3, 16($sp)
        move  $fp, $sp
        move  $s0, $a0
        move  $s1, $a1
        move  $s2, $a2
        move  $s3, $a3
rnext:  move  $a0, $s3                # fn(the value so far, the next word)
        lw    $a1, 0($s0)
        jalr  $s2                     # call whatever function we were given
        move  $s3, $v0
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, rnext
        move  $v0, $s3
        move  $sp, $fp
        lw    $s3, 16($sp)
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra

# add2, mul2, max2: $a0, $a1 -> $v0 (leaves)
add2:   addu  $v0, $a0, $a1
        jr    $ra
mul2:   mul   $v0, $a0, $a1
        jr    $ra
max2:   move  $v0, $a0
        bge   $a0, $a1, mdone
        move  $v0, $a1
mdone:  jr    $ra
