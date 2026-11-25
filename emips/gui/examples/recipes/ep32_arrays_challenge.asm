# MIPS by Example in eMIPS, episode 32's challenge, solved: arr_find in a
# second library file, ep32_arrays_challenge_lib.asm, included next to the
# first; the program asks where 7 and 8 are in primes, and where 4 is in
# temps once reversed.
        .data
primes: .word 2, 3, 5, 7, 11, 13
temps:  .word -3, 4, -1, 7, 2
sumis:  .asciiz "  sum "
maxis:  .asciiz "  max "
revis:  .asciiz "reversed "
findis: .asciiz " at "
        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 0-15 the home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        move  $fp, $sp
        la    $a0, primes             # show(primes, 6)
        li    $a1, 6
        jal   show
        la    $a0, temps              # show(temps, 5)
        li    $a1, 5
        jal   show
        la    $a0, temps              # reverse temps, print it
        li    $a1, 5
        jal   arr_reverse
        li    $v0, 4
        la    $a0, revis
        syscall
        la    $a0, temps
        li    $a1, 5
        jal   arr_print
        li    $v0, 11
        li    $a0, '\n'
        syscall
        la    $a0, primes             # where are 7 and 8 in primes?
        li    $a1, 6
        li    $a2, 7
        jal   find
        la    $a0, primes
        li    $a1, 6
        li    $a2, 8
        jal   find
        la    $a0, temps              # and 4 in temps, now reversed?
        li    $a1, 5
        li    $a2, 4
        jal   find
        move  $sp, $fp                # main's epilogue
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# show(a, n): the array, its sum and its largest, on one line
show:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 16 $s1
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        move  $s0, $a0                # a and n: needed after each call
        move  $s1, $a1
        jal   arr_print               # ($a0, $a1 already set)
        li    $v0, 4
        la    $a0, sumis
        syscall
        move  $a0, $s0
        move  $a1, $s1
        jal   arr_sum
        move  $a0, $v0
        li    $v0, 1
        syscall
        li    $v0, 4
        la    $a0, maxis
        syscall
        move  $a0, $s0
        move  $a1, $s1
        jal   arr_max
        move  $a0, $v0
        li    $v0, 1
        syscall
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

# find(a, n, x): "x at i" (it calls arr_find: a frame)
find:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        move  $s0, $a2                # x: needed after the call
        jal   arr_find
        move  $t0, $v0
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 4
        la    $a0, findis
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

        .include "ep32_arrays_lib.asm"
        .include "ep32_arrays_challenge_lib.asm"
