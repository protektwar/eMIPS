# Lab 3 — make this loop faster without changing what it computes.
# Run:  python3 -m emips pipe docs/guide/labs/lab3_sum.asm
        .data
nums:   .word 1, 2, 3, 4, 5, 6, 7, 8
        .text
main:   la    $t0, nums
        li    $t1, 8
        move  $t2, $zero
loop:   lw    $t3, 0($t0)
        addu  $t2, $t2, $t3
        addiu $t0, $t0, 4
        addiu $t1, $t1, -1
        bnez  $t1, loop
        move  $a0, $t2
        li    $v0, 1            # prints 36
        syscall
        li    $v0, 10
        syscall
