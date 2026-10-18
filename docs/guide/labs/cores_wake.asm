# Lab: wake core 1 yourself (run with --cores 2 --boot reset).
# In the reset state only core 0 runs; core 1 is powered down. Core 0
# enables the Cluster Power Controller, sets core 1's reset base and
# sends it PwrUp; core 1 starts there in its own reset state, joins the
# coherence domain and stores 42 for core 0 to print. Core 0 also prints
# core 1's power state (CPC_CO_STAT_CONF.SEQSTATE) before and after.
        .eqv  GCR, 0xbfbf8000        # the CM's registers (kseg1)
        .eqv  CPC, 0xbbde0000        # the CPC's, once enabled (kseg1)
        .data
flag:   .word 0
        .text
main:   li    $s0, GCR
        li    $t0, 0x1bde0001        # GCR_CPC_BASE: the CPC at 0x1BDE0000, on
        sw    $t0, 0x0088($s0)
        li    $t0, 3
        sw    $t0, 0x0020($s0)       # GCR_ACCESS: cores 0 and 1 may write
        li    $t0, 1 << 16
        sw    $t0, 0x2018($s0)       # GCR_CL_OTHER: core 1
        la    $t0, secondary
        li    $t1, 0xa0000000
        or    $t0, $t0, $t1          # its kseg1 address (4 KB aligned)
        sw    $t0, 0x4020($s0)       # GCR_CO_RESET_BASE
        li    $s1, CPC
        li    $t0, 1 << 16
        sw    $t0, 0x2010($s1)       # CPC_CL_OTHER: core 1
        jal   state                  # 0 = D0, powered down
        li    $t0, 3
        sw    $t0, 0x4000($s1)       # CPC_CO_CMD = PwrUp
        jal   state                  # 6 = U5, non-coherent execution
        la    $s2, flag
wait:   lw    $t0, 0($s2)
        beqz  $t0, wait
        jal   state                  # 7 = U6, coherent execution
        lw    $a0, 0($s2)
        li    $v0, 1                 # print what core 1 stored
        syscall
        li    $v0, 10
        syscall

state:  lw    $a0, 0x4008($s1)       # CPC_CO_STAT_CONF
        srl   $a0, $a0, 19
        andi  $a0, $a0, 0xf          # SEQSTATE
        li    $v0, 1
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        jr    $ra

        .align 12                    # a reset base is 4 KB aligned
secondary:                           # core 1, from its reset
        li    $s0, GCR
        li    $t0, 0xff
        sw    $t0, 0x2008($s0)       # GCR_CL_COHERENCE: join the domain
        mfc0  $t0, $15, 1
        andi  $t0, $t0, 0x3ff        # CPUNum = 1
        addiu $t0, $t0, 41
        la    $t1, flag
        sw    $t0, 0($t1)
park:   b     park
