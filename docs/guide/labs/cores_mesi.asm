# Lab: watch MESI (run with --cores 2 --caches; step with "cm x").
# Both cores run in lockstep, core 0 first; the steps that matter:
#  step 7:  core 0 loads x       no other copy: core 0 Exclusive
#  step 8:  core 1 stores x      core 0 E -> Invalid, core 1 Modified
#  step 10: core 0 loads x       core 1 M -> written back, both Shared
#  step 11: core 0 stores x      upgrade: core 1 S -> Invalid, core 0 M
        .data
x:      .word 0
        .text
main:   mfc0  $s0, $15, 1
        andi  $s0, $s0, 0x3ff        # CPUNum
        la    $t1, x
        bnez  $s0, core1
        lw    $t2, 0($t1)            # core 0, step 7
        nop
        nop
        lw    $t3, 0($t1)            # step 10: 7, from core 1's cache
        sw    $s0, 0($t1)            # step 11
        move  $a0, $t3
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall
core1:  li    $t0, 7
        sw    $t0, 0($t1)            # core 1, step 8
park:   b     park
# eMIPS: --cores 2 --caches    (the settings this program runs with)
