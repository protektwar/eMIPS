# ALU results are forwarded: no stalls.
        .text
main:   addiu $t0, $zero, 5
        addu  $t1, $t0, $t0     # $t0 from EX/MEM (the previous instruction)
        subu  $t2, $t1, $t0     # $t1 from EX/MEM, $t0 from MEM/WB
        li    $v0, 10
        syscall
