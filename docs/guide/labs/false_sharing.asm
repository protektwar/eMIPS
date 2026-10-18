# Lab: false sharing (run with --cores 2 --caches --pipeline --model 24k).
# Each core adds 1 to its own counter 50 times — they share nothing. Core 0
# prints how many Count ticks (2 cycles each) its loop took. With STRIDE 4
# the two counters sit in one 32-byte line; change it to 32 and compare.
        .eqv  STRIDE, 4
        .eqv  N, 50
        .data
        .align 5
cnt:    .space 64
done:   .word 0
        .text
main:   mfc0  $s0, $15, 1
        andi  $s0, $s0, 0x3ff        # CPUNum
        li    $t0, STRIDE
        mul   $t0, $t0, $s0
        la    $s1, cnt
        addu  $s1, $s1, $t0          # this core's counter
        li    $t1, N
        mfc0  $s2, $9                # Count at the start
loop:   lw    $t2, 0($s1)
        addiu $t2, $t2, 1
        sw    $t2, 0($s1)
        addiu $t1, $t1, -1
        bnez  $t1, loop
        mfc0  $s3, $9
        bnez  $s0, core1
        la    $t3, done              # core 0: wait for core 1
w:      lw    $t4, 0($t3)
        beqz  $t4, w
        subu  $a0, $s3, $s2
        li    $v0, 1
        syscall                      # Count ticks of core 0's loop
        li    $v0, 10
        syscall
core1:  li    $t4, 1
        sw    $t4, done
park:   wait
        b     park
# eMIPS: --cores 2 --caches --pipeline --model 24k    (the settings this program runs with)
