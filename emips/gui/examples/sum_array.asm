# Sum an array, print the result (MARS / SPIM syscalls)
        .data
nums:   .word 3, 4, 5, 6
msg:    .asciiz "sum = "
        .text
main:   la    $t0, nums
        li    $t1, 4
        move  $t2, $zero
loop:   lw    $t3, 0($t0)
        addu  $t2, $t2, $t3
        addiu $t0, $t0, 4
        addiu $t1, $t1, -1
        bnez  $t1, loop
        li    $v0, 4
        la    $a0, msg
        syscall
        li    $v0, 1
        move  $a0, $t2
        syscall
        li    $v0, 10
        syscall
