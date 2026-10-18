# Lab: a race (run with --cores 2). Both cores add 1 to one counter N
# times with lw / addiu / sw — no lock. 2 x N = 100 expected... but the
# read-modify-write of one core can interleave with the other's.
        .eqv  N, 50
        .data
count:  .word 0
done:   .word 0
        .text
main:   mfc0  $s0, $15, 1
        andi  $s0, $s0, 0x3ff        # CPUNum
        la    $s1, count
        li    $t0, N
loop:   lw    $t1, 0($s1)            # read
        addiu $t1, $t1, 1            # modify
        sw    $t1, 0($s1)            # write
        addiu $t0, $t0, -1
        bnez  $t0, loop
        bnez  $s0, core1
w:      lw    $t2, done              # core 0: wait for core 1
        beqz  $t2, w
        lw    $a0, 0($s1)
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall
core1:  sync                         # its last store before the flag
        li    $t2, 1
        sw    $t2, done
park:   wait
        b     park
# eMIPS: --cores 2    (the settings this program runs with)
