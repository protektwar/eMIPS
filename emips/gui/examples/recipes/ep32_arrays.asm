# MIPS by Example in eMIPS, episode 32 — A library of your own.  The array
# functions live in their own file, ep32_arrays_lib.asm; .include puts them
# here, as if typed at that line.  The program only knows their contract:
# their names, arguments, results, and the registers they keep.
        .data
primes: .word 2, 3, 5, 7, 11, 13
temps:  .word -3, 4, -1, 7, 2
sumis:  .asciiz "  sum "
maxis:  .asciiz "  max "
revis:  .asciiz "reversed "
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

        .include "ep32_arrays_lib.asm"
