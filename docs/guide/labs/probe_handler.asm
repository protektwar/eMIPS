# Lab 28: a debug handler for the probe's EJTAG memory (dmseg).  It counts
# its calls in the probe's memory (0xff200100) and returns past the sdbbp.
        .ktext 0xff200200            # ECR.ProbTrap = 1: the debug vector
        mtc0  $k0, $31               # DESAVE
        lui   $k0, 0xff20
        sw    $k1, 0x104($k0)        # save $k1 in the probe's memory
        lw    $k1, 0x100($k0)        # the count, in the probe's memory
        addiu $k1, $k1, 1
        sw    $k1, 0x100($k0)
        lw    $k1, 0x104($k0)
        mfc0  $k0, $24
        addiu $k0, $k0, 4            # past the sdbbp
        mtc0  $k0, $24
        mfc0  $k0, $31
        ehb                          # MTC0 DEPC -> DERET (EJTAG Table 2.6)
        deret
