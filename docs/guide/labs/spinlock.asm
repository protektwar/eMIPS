# Lab: a spinlock (run with --cores 2). race.asm with the read-modify-write
# in a critical section: ll / sc take the lock atomically, sync orders
# the section's accesses after the acquire and before the release (MIPS
# memory is weakly ordered; LL / SC are no barriers).
        .eqv  N, 50
        .data
lock:   .word 0
count:  .word 0
done:   .word 0
        .text
main:   mfc0  $s0, $15, 1
        andi  $s0, $s0, 0x3ff        # CPUNum
        la    $s1, count
        la    $s2, lock
        li    $t0, N
loop:
acquire:
        ll    $t3, 0($s2)
        bnez  $t3, acquire           # taken: spin
        li    $t3, 1
        sc    $t3, 0($s2)
        beqz  $t3, acquire           # someone was faster: again
        sync                         # acquire: nothing below moves up
        lw    $t1, 0($s1)            # the critical section
        addiu $t1, $t1, 1
        sw    $t1, 0($s1)
        sync                         # release: nothing above moves down
        sw    $zero, 0($s2)          # unlock
        addiu $t0, $t0, -1
        bnez  $t0, loop
        bnez  $s0, core1
w:      lw    $t2, done
        beqz  $t2, w
        lw    $a0, 0($s1)
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall
core1:  sync
        li    $t2, 1
        sw    $t2, done
park:   wait
        b     park
# eMIPS: --cores 2    (the settings this program runs with)
