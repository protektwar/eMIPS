# A branch right after a load of its operand waits for MEM: 2 stalls.
        .text
main:   lw    $t0, 0($sp)
        beq   $t0, $zero, done
done:   li    $v0, 10
        syscall
