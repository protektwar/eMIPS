# Lazy DSP context, as an operating system does it: a process starts with
# Status.MX = 0, so its first DSP instruction raises DSP State Disabled
# (ExcCode 26).  The handler then sets MX (an OS would also restore that
# process's DSPControl and accumulators) and returns to the same
# instruction, which now runs.  A process that never uses the DSP Module
# costs nothing at a context switch.
        .data
traps:  .word   0                   # DSP State Disabled exceptions taken
        .text
main:   mfc0    $t0, $12            # a new "process": the DSP Module off
        li      $t1, ~0x01000000
        and     $t0, $t0, $t1
        mtc0    $t0, $12
        li      $s0, 0x01020304
        li      $s1, 0x10101010
        li      $s2, 3
loop:   addu.qb $s0, $s0, $s1       # the first one traps, then MX = 1
        addiu   $s2, $s2, -1
        bnez    $s2, loop
        move    $a0, $s0
        li      $v0, 34
        syscall                     # 0x31323334
        li      $a0, ' '
        li      $v0, 11
        syscall
        lw      $a0, traps
        li      $v0, 1
        syscall                     # 1
        li      $v0, 10
        syscall

        .ktext  0x80000180          # the general exception vector
        mfc0    $k0, $13            # Cause
        srl     $k0, $k0, 2
        andi    $k0, $k0, 0x1f      # ExcCode
        li      $k1, 26             # DSPDis?
        bne     $k0, $k1, other
        mfc0    $k0, $12
        lui     $k1, 0x0100         # Status.MX
        or      $k0, $k0, $k1
        mtc0    $k0, $12
        lw      $k0, traps
        addiu   $k0, $k0, 1
        sw      $k0, traps
        eret                        # back to EPC: the same addu.qb (ERET is a
                                    # hazard barrier: it sees MX = 1)
other:  li      $v0, 10             # anything else: give up
        syscall
