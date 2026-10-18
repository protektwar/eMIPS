# Lab 3 — one solution: no stalls, and the delay slot does useful work.
        .data
nums:   .word 1, 2, 3, 4, 5, 6, 7, 8
        .text
main:   la    $t0, nums
        li    $t1, 8
        move  $t2, $zero
        .set  noreorder         # we fill the delay slot ourselves
loop:   lw    $t3, 0($t0)
        addiu $t1, $t1, -1      # independent: hides the load delay
        addiu $t0, $t0, 4
        bnez  $t1, loop         # $t1 is ready (EX/MEM -> ID forwarding)
        addu  $t2, $t2, $t3     # the delay slot: runs on every round
        .set  reorder
        move  $a0, $t2
        li    $v0, 1            # prints 36
        syscall
        li    $v0, 10
        syscall
