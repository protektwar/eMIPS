# Lab: how long does a divide take?  Counter 0 counts cycles; the program
# reads it before and after each divu + mflo and prints the difference.
# Run it with --model 24k (the 24K divides 8, 16, 24 or 32 bits at a time:
# "early-in"), and on the 5-stage model (one fixed latency).
        .data
nums:   .word 100, 0x1234, 0x123456, 0x12345678, 0
        .text
main:   li    $t0, 0x2               # PerfCtl0: event 0 (cycles), kernel mode
        mtc0  $t0, $25, 0
        ehb
        la    $s0, nums
loop:   lw    $s1, 0($s0)
        beq   $s1, $zero, done
        li    $t1, 7
        mfc0  $t2, $25, 1            # cycles before
        divu  $s1, $t1
        mflo  $t3                    # waits for the quotient
        mfc0  $t4, $25, 1            # cycles after
        subu  $a0, $t4, $t2
        li    $v0, 1
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        addiu $s0, $s0, 4
        b     loop
done:   li    $v0, 10
        syscall
# eMIPS: --model 24k --pipeline    (the settings this program runs with)
