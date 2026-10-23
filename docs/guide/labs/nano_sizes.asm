# Lab 63: how big is an instruction?  Sum ten numbers in nanoMIPS.
        .data
nums:   .word   3, 1, 4, 1, 5, 9, 2, 6, 5, 3
        .text
main:   la      $t1, nums           # the array
        li      $t2, 10             # how many
        li      $t3, 0              # the sum
loop:   lw      $t0, 0($t1)
        addu    $t3, $t3, $t0
        addiu   $t1, $t1, 4
        addiu   $t2, $t2, -1
        bnezc   $t2, loop
        move    $a0, $t3
        li      $t4, 1              # print_int
        syscall
        li      $t4, 10             # exit
        syscall
# eMIPS: --arch nanomips
