# A branch is decided in ID: it needs its operand one stage earlier.
        .text
main:   addiu $t0, $zero, 1
        bne   $t0, $zero, done  # $t0 is still in EX: 1 stall
done:   li    $v0, 10
        syscall
