# Lab 15 — Find the hazard.  The handler skips the instruction that
# caused the exception (here a trap).  On the 5-stage model the program
# prints "survived" and ends; on the 24K (these settings) it never ends.
# Why, and what is the smallest fix?
        .data
msg:    .asciiz "survived"
        .text
main:   li    $t0, 1
        teq   $t0, $t0          # a trap (Tr): the handler skips it
        la    $a0, msg
        li    $v0, 4
        syscall
        li    $v0, 10
        syscall

        .ktext 0x80000180
        mfc0  $k0, $14          # EPC: the teq
        addiu $k0, $k0, 4
        mtc0  $k0, $14
        eret
# eMIPS: --pipeline --model 24k    (the settings this program runs with)
