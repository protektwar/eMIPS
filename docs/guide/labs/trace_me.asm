# Lab 29: what a trace costs.  The loop calls one of two functions
# through a table (jalr: a target only the trace can tell) and sums
# what they return.
        .text
main:   la    $s0, table
        la    $s1, data
        li    $s2, 8                # 8 elements
        li    $s3, 0                # the sum
loop:   lw    $t0, 0($s1)
        andi  $t1, $t0, 1           # odd or even: which function
        sll   $t1, $t1, 2
        addu  $t1, $s0, $t1
        lw    $t9, 0($t1)
        jalr  $t9
        addu  $s3, $s3, $v0
        addiu $s1, $s1, 4
        addiu $s2, $s2, -1
        bnez  $s2, loop
        move  $a0, $s3
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall
even:   sll   $v0, $t0, 1           # 2x
        jr    $ra
odd:    move  $v0, $t0              # x
        jr    $ra

        .data
table:  .word even, odd
data:   .word 3, 8, 5, 2, 7, 4, 1, 6
