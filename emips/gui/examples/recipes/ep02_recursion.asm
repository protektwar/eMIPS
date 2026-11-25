# MIPS by Example in eMIPS, episode 2 — Arguments and recursion.  o32: the
# first four arguments in $a0-$a3, the caller reserves 16 bytes for them
# anyway (the home area, 0-15 of its frame) and puts a fifth at 16, a sixth
# at 20 ...; the result in $v0.  sum_of_squares takes five; fact calls
# itself: every call gets its own frame, so each keeps its own n.
        .data
squares:.asciiz "sum of squares = "
fact5:  .asciiz "fact(5) = "
fact10: .asciiz "fact(10) = "
        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 16 the fifth argument,
        sw    $ra, 28($sp)            # 0-15 the home area
        sw    $fp, 24($sp)
        move  $fp, $sp
        li    $a0, 1                  # sum_of_squares(1, 2, 3, 4, 5)
        li    $a1, 2
        li    $a2, 3
        li    $a3, 4
        li    $t0, 5
        sw    $t0, 16($sp)            # the fifth argument: in our frame, at 16
        jal   sum_of_squares
        la    $a0, squares
        move  $a1, $v0
        jal   print_result
        li    $a0, 5                  # fact(5)
        jal   fact
        la    $a0, fact5
        move  $a1, $v0
        jal   print_result
        li    $a0, 10                 # fact(10)
        jal   fact
        la    $a0, fact10
        move  $a1, $v0
        jal   print_result
        move  $sp, $fp
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# sum_of_squares: a, b, c, d in $a0-$a3, e at 16 in the caller's frame
sum_of_squares:
        addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 0-15 home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        sw    $a1, 36($fp)            # b, c, d into the caller's home area:
        sw    $a2, 40($fp)            # 32 + 4, 8, 12 (calls may change $a1-$a3)
        sw    $a3, 44($fp)
        jal   square                  # a * a ($a0 is still a)
        move  $s0, $v0                # the sum so far: safe in $s0
        lw    $a0, 36($fp)
        jal   square                  # b * b
        addu  $s0, $s0, $v0
        lw    $a0, 40($fp)
        jal   square                  # c * c
        addu  $s0, $s0, $v0
        lw    $a0, 44($fp)
        jal   square                  # d * d
        addu  $s0, $s0, $v0
        lw    $a0, 48($fp)            # e: 16 in the caller's frame = 32 + 16
        jal   square
        addu  $v0, $s0, $v0           # the result
        move  $sp, $fp
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# square: $a0 -> $v0 = $a0 * $a0 (a leaf: no frame)
square: mul   $v0, $a0, $a0
        jr    $ra

# fact: $a0 = n -> $v0 = n! (recursive: n * fact(n - 1), fact(1) = 1)
fact:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 0-15 home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        li    $v0, 1                  # fact(n <= 1) = 1
        ble   $a0, 1, factdone
        move  $s0, $a0                # this call's n, kept across the call
        addiu $a0, $a0, -1
        jal   fact                    # fact(n - 1): a new frame below ours
        mul   $v0, $s0, $v0           # n * fact(n - 1)
factdone:
        move  $sp, $fp
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# print_result: $a0 = a text, $a1 = a number -> prints both and a newline
print_result:
        li    $v0, 4                  # (a leaf: only system calls)
        syscall
        li    $v0, 1
        move  $a0, $a1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra
