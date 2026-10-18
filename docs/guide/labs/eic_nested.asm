# Lab 21's solution (run with --cores 2): labs/eic.asm with a vector-2
# handler that lets more important interrupts in.  It clears its source,
# saves EPC / Status / SRSCtl, raises Status.IPL to its own RIPL (2),
# clears EXL and sets IE: the timer (vector 5 > 2) now interrupts it.  It
# prints the vectors, the ticks, and the ticks that came while the
# vector-2 handler ran: 0x000000a2 3 3.
        .eqv  GIC, 0xbbdc0000        # the GIC, kseg1
        .eqv  VL, 0xbbdc8000         # this core's local section
        .eqv  MAP, 0x80000000        # MAP_TO_PIN (in EIC mode: a vector)
        .data
order:  .word 0                      # the vectors, as taken: 4 bits each
ticks:  .word 0
during: .word 0
        .text
main:   mfc0  $t0, $15, 1
        andi  $t0, $t0, 0x3ff        # CPUNum
        bnez  $t0, idle
        li    $t1, VL
        li    $t0, 1
        sw    $t0, 0($t1)            # GIC_VL_CTL.EIC_MODE = 1
        li    $t0, MAP | 4           # the CP0 timer -> vector 5
        sw    $t0, 0x48($t1)
        li    $t0, MAP | 1           # software interrupt 0 -> vector 2
        sw    $t0, 0x54($t1)
        li    $t0, 1 << 4            # ... enabled (GIC_VL_SMASK)
        sw    $t0, 0x10($t1)
        li    $t2, GIC
        li    $t0, MAP | 9           # source 24, core 0's IPI -> vector 10
        sw    $t0, 0x500 + 4 * 24($t2)
        li    $t0, 1                 # the shadow set of each vector
        sw    $t0, 0x100 + 4 * 5($t1)
        li    $t0, 2
        sw    $t0, 0x100 + 4 * 10($t1)
        li    $t0, 3
        sw    $t0, 0x100 + 4 * 2($t1)
        li    $t0, 4 << 5            # IntCtl.VS = 4: vectors 128 bytes apart
        mtc0  $t0, $12, 1
        li    $t0, 0x80000000 + 24   # raise the IPI (GIC_SH_WEDGE) ...
        sw    $t0, 0x280($t2)
        li    $t0, 0x00800100        # ... and software interrupt 0; Cause.IV
        mtc0  $t0, $13
        mtc0  $zero, $9              # Count = 0
        li    $t0, 200
        mtc0  $t0, $11               # Compare: the first tick
        li    $t0, 1                 # Status: IPL 0, IE
        mtc0  $t0, $12
        la    $t3, ticks
wait:   lw    $t0, 0($t3)
        li    $t1, 3
        bne   $t0, $t1, wait
        lw    $a0, order
        li    $v0, 34                # the vectors in the order taken
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        lw    $a0, ticks
        li    $v0, 1
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        lw    $a0, during
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall
idle:   b     idle

        .ktext 0x80000300            # vector 2: software interrupt 0, set 3
        li    $t0, 0x00800000        # clear IP0 (keep IV): the source first
        mtc0  $t0, $13
        mfc0  $s0, $14               # save EPC, Status, SRSCtl (set 3's $s0-$s2)
        mfc0  $s1, $12
        mfc0  $s2, $12, 2
        mfc0  $t0, $13
        srl   $t0, $t0, 10
        andi  $t0, $t0, 0x3f         # Cause.RIPL (2)
        move  $s3, $s1
        ins   $s3, $t0, 10, 6        # Status.IPL = RIPL
        ins   $s3, $zero, 1, 4       # EXL = 0, KSU = 0 (ERL stays 0)
        ori   $s3, $s3, 1            # IE
        mtc0  $s3, $12
        ehb
        li    $t2, 2000              # long enough for the timer to come in
spin:   addiu $t2, $t2, -1
        bnez  $t2, spin
        lw    $t3, ticks             # the timer ran inside us
        sw    $t3, during
        di
        mtc0  $s1, $12               # Status (EXL = 1 again)
        mtc0  $s0, $14
        mtc0  $s2, $12, 2
        ehb
        li    $t1, 2
        b     note
        nop

        .ktext 0x80000480            # vector 5: the timer, set 1
        addiu $s0, $s0, 1            # set 1's $s0: the ticks so far
        sw    $s0, ticks
        li    $t1, 3
        beq   $s0, $t1, last
        mfc0  $t0, $11
        addiu $t0, $t0, 200          # the next tick (clears this one)
        mtc0  $t0, $11
        ehb
        eret
last:   li    $t0, VL                # the third: timer off (GIC_VL_RMASK)
        li    $t1, 1 << 2
        sw    $t1, 0x0c($t0)
        mtc0  $zero, $11             # (and its request)
        ehb
        eret

        .ktext 0x80000700            # vector 10: the IPI, set 2
        li    $t0, GIC
        li    $t1, 24
        sw    $t1, 0x280($t0)        # clear source 24 (GIC_SH_WEDGE)
        li    $t1, 10
note:   lw    $t0, order             # order = order << 4 | vector
        sll   $t0, $t0, 4
        or    $t0, $t0, $t1
        sw    $t0, order
        eret
