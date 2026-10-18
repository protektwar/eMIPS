# Lab: a CP0 hazard on the MIPS 24K (run with --model 24k).
# MTC0 writes a CP0 register late in the pipeline; an MFC0 of it within
# two instructions still reads the old value (24K manual, Table 2.7).
# EHB ("execution hazard barrier") waits until the write is done.
        .set  noreorder
        .text
main:   li    $t0, 0x12345000
        mtc0  $t0, $4                # Context = 0x12345000
        mfc0  $a0, $4                # too close: still the old Context
        li    $v0, 34                # print_hex
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        li    $t0, 0x6789a000
        mtc0  $t0, $4
        ehb                          # clear the hazard
        mfc0  $a0, $4                # the new Context
        li    $v0, 34
        syscall
        li    $v0, 10
        syscall
# eMIPS: --model 24k --pipeline    (the settings this program runs with)
