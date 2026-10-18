# Who wrote "count"?  Watch register pair 2 watches the doubleword of
# count for stores (WatchLo2.W); the Watch exception (ExcCode 23) says
# where the store is.
        .text
main:   la    $t0, count
        ori   $t0, $t0, 1          # WatchLo2: count's doubleword, W
        mtc0  $t0, $18, 2
        li    $t1, 0x40000000      # WatchHi2.G: any address space
        mtc0  $t1, $19, 2
        ehb
        li    $s0, 3
loop:   addiu $s0, $s0, -1         # ... a lot of other work ...
        bnez  $s0, loop
        la    $t2, count
        sw    $zero, 0($t2)        # the culprit
        li    $v0, 10
        syscall

        .data
count:  .word 5
msg:    .asciiz "count was written at "

        .ktext 0x80000180          # the general exception vector
        mfc0  $k0, $13
        andi  $k0, $k0, 0x7c
        li    $k1, 23 << 2         # ExcCode 23: Watch
        bne   $k0, $k1, other
        la    $a0, msg
        li    $v0, 4
        syscall
        mfc0  $a0, $14             # EPC: the store
        li    $v0, 34
        syscall
        mtc0  $zero, $18, 2        # the watch off: the store may run now
        li    $k0, 7
        mtc0  $k0, $19, 2          # clear WatchHi2's I / R / W (write 1)
        ehb
        eret
other:  li    $v0, 10
        syscall
