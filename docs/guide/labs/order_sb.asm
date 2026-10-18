# Lab: store buffering (run with --cores 2 --caches --pipeline --model 24k).
# Core 0 stores x and loads y; core 1 stores y and loads x. Can both loads
# see 0? Under "sequential consistency" never: one of the stores is first.
# Like a real litmus test, it runs ROUNDS times (fresh x and y lines each
# round) and counts the rounds in which both loads saw 0. In a round both
# cores first read x and y (Shared in both caches), then wait for the same
# Count value (one timebase: the cores started together) and go.
        .eqv  ROUNDS, 16
        .eqv  GO, 300                # Count value of the first round
        .data
        .align 5
x:      .space 32 * ROUNDS           # x[k], y[k]: a 32-byte line each
y:      .space 32 * ROUNDS
res:    .space 8 * ROUNDS            # res[k]: core 0's load, core 1's load
done:   .word 0
        .text
main:   mfc0  $s0, $15, 1
        andi  $s0, $s0, 0x3ff        # CPUNum
        li    $s3, 0                 # k
round:  sll   $t0, $s3, 5
        la    $s1, x                 # core 0: mine = x[k], other = y[k]
        addu  $s1, $s1, $t0
        la    $s2, y
        addu  $s2, $s2, $t0
        move  $t0, $s1
        movn  $s1, $s2, $s0          # core 1: swapped
        movn  $s2, $t0, $s0
        lw    $t1, 0($s1)            # both lines Shared in both cores
        lw    $t1, 0($s2)
        sll   $t2, $s3, 7
        addiu $t2, $t2, GO           # this round starts at GO + 128 k
wait:   mfc0  $t3, $9                # Count
        sltu  $t3, $t3, $t2
        bnez  $t3, wait
        li    $t4, 1
        sw    $t4, 0($s1)            # mine = 1
#       sync                         # <- try it
        lw    $t5, 0($s2)            # the other
        sll   $t6, $s3, 3
        sll   $t7, $s0, 2
        addu  $t6, $t6, $t7
        sw    $t5, res($t6)          # res[k][cpu]
        addiu $s3, $s3, 1
        li    $t0, ROUNDS
        bne   $s3, $t0, round
        bnez  $s0, core1
        la    $t0, done              # core 0: wait for core 1, count
w1:     lw    $t1, 0($t0)
        beqz  $t1, w1
        li    $a0, 0                 # rounds where both saw 0
        li    $t2, 0
count:  lw    $t3, res($t2)
        lw    $t4, res + 4($t2)
        or    $t3, $t3, $t4
        sltiu $t3, $t3, 1
        addu  $a0, $a0, $t3
        addiu $t2, $t2, 8
        li    $t5, 8 * ROUNDS
        bne   $t2, $t5, count
        li    $v0, 1
        syscall
        la    $a0, msg
        li    $v0, 4
        syscall
        li    $v0, 10
        syscall
core1:  sync                         # its results before the flag
        li    $t1, 1
        sw    $t1, done
park:   wait
        b     park
        .data
msg:    .asciiz " of 16 rounds: both loads saw 0\n"
