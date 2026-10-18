# Lab: inter-processor interrupts (run with --cores 4).
# Cores 1-3 enable their interrupt pin 0 (Cause.IP2) and sleep in WAIT.
# Core 0 wakes them one by one through the GIC: a write to GIC_SH_WEDGE
# with bit 31 set raises source 32 + n, which the boot firmware (as
# Boot-MIPS) made a rising-edge source routed to core n, pin 0. The
# handler clears its source (the same write without bit 31), notes that
# it woke up and returns; core 0 prints the order the cores woke in.
        .eqv  GIC, 0xbbdc0000        # the GIC, kseg1
        .eqv  WEDGE, 0x280           # GIC_SH_WEDGE
        .eqv  FIRST_IPI, 32
        .data
order:  .word 0                      # the cores, as they woke: 4 bits each
count:  .word 0
        .text
main:   mfc0  $s0, $15, 1
        andi  $s0, $s0, 0x3ff        # CPUNum
        bnez  $s0, sleeper
        li    $t0, 100               # core 0: give them time to fall asleep
pause:  addiu $t0, $t0, -1
        bnez  $t0, pause
        li    $t1, GIC
        li    $t2, 3                 # wake core 3, then 2, then 1
send:   li    $t3, 0x80000000 + FIRST_IPI
        addu  $t3, $t3, $t2
        sw    $t3, WEDGE($t1)        # IPI to core $t2
        la    $t4, count
w:      lw    $t5, 0($t4)            # wait until it has woken up
        li    $t6, 4
        subu  $t6, $t6, $t2
        bne   $t5, $t6, w
        addiu $t2, $t2, -1
        bnez  $t2, send
        lw    $a0, order
        li    $v0, 34                # print hex: the order
        syscall
        li    $v0, 10
        syscall

sleeper:
        mfc0  $t0, $12               # Status
        ori   $t0, $t0, 0x0401       # IM2 (pin 0) and IE
        mtc0  $t0, $12
sleep:  wait
        b     sleep

        .ktext 0x80000180            # the interrupt handler (every core)
        li    $k0, GIC
        mfc0  $k1, $15, 1
        andi  $k1, $k1, 0x3ff
        addiu $k1, $k1, FIRST_IPI
        sw    $k1, WEDGE($k0)        # clear this core's IPI source
        la    $k0, order             # order = order << 4 | cpu
        lw    $k1, 0($k0)
        sll   $k1, $k1, 4
        sw    $k1, 0($k0)
        mfc0  $k1, $15, 1
        andi  $k1, $k1, 0x3ff
        lw    $k0, order
        or    $k0, $k0, $k1
        la    $k1, order
        sw    $k0, 0($k1)
        la    $k0, count             # count = count + 1
        lw    $k1, 0($k0)
        addiu $k1, $k1, 1
        sw    $k1, 0($k0)
        sync                         # visible before the eret
        eret
# eMIPS: --cores 4    (the settings this program runs with)
