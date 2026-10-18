# Lab 31: interrupts on the Malta (run with --board malta).  The RTC's
# periodic interrupt (IRQ 8) goes through the two 8259 interrupt
# controllers of the PIIX4 to the CPU's INT0 (Cause.IP2).  The handler
# asks the GT-64120 for the vector (a PCI interrupt acknowledge cycle),
# clears the RTC's flag and sends both controllers an EOI.  main counts
# the ticks the handler saw until 10 have come.
        .text
main:   li    $s0, 0xb8000000       # ISA port P is at 0xb8000000 + P
        li    $t0, 0x11             # ICW1: edge, cascade, ICW4 follows
        sb    $t0, 0x20($s0)        # master (ports 0x20 / 0x21)
        sb    $t0, 0xa0($s0)        # slave  (ports 0xa0 / 0xa1)
        li    $t0, 0x20             # ICW2: master vectors 0x20-0x27
        sb    $t0, 0x21($s0)
        li    $t0, 0x28             # ICW2: slave vectors 0x28-0x2f
        sb    $t0, 0xa1($s0)
        li    $t0, 0x04             # ICW3: the slave on the master's IRQ 2
        sb    $t0, 0x21($s0)
        li    $t0, 0x02             # ICW3: the slave's identity, 2
        sb    $t0, 0xa1($s0)
        li    $t0, 0x01             # ICW4: 8086 mode, normal EOI
        sb    $t0, 0x21($s0)
        sb    $t0, 0xa1($s0)
        li    $t0, 0xfb             # OCW1: master: only IRQ 2 (the slave)
        sb    $t0, 0x21($s0)
        li    $t0, 0xfe             # OCW1: slave: only IRQ 8 (the RTC)
        sb    $t0, 0xa1($s0)

        li    $t0, 0x0a             # RTC register A: 32.768 kHz, rate 3 (8192 Hz)
        sb    $t0, 0x70($s0)
        li    $t0, 0x23
        sb    $t0, 0x71($s0)
        li    $t0, 0x0b             # RTC register B: + PIE (periodic interrupt)
        sb    $t0, 0x70($s0)
        li    $t0, 0x42
        sb    $t0, 0x71($s0)

        mfc0  $t0, $12              # Status: IM2 (INT0) and IE
        ori   $t0, $t0, 0x0401
        mtc0  $t0, $12
        li    $s1, 10
wait:   lw    $t0, ticks
        bne   $t0, $s1, wait
        mtc0  $zero, $12            # interrupts off
        lw    $a0, ticks
        li    $v0, 1
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        lw    $a0, vector
        li    $v0, 34
        syscall
        li    $v0, 10
        syscall

        .kdata
ticks:  .word 0
vector: .word 0

        .ktext 0x80000180
        lui   $k0, 0xbbe0           # the GT-64120's registers
        lw    $k1, 0xc34($k0)       # PCI interrupt acknowledge: the vector
        srl   $k1, $k1, 24          # (little-endian register: byte-swapped)
        sw    $k1, vector
        lui   $k0, 0xb800
        li    $k1, 0x0c             # RTC register C: reading it clears the flag
        sb    $k1, 0x70($k0)
        lbu   $k1, 0x71($k0)
        li    $k1, 0x20             # OCW2: non-specific EOI, slave then master
        sb    $k1, 0xa0($k0)
        sb    $k1, 0x20($k0)
        lw    $k1, ticks
        addiu $k1, $k1, 1
        sw    $k1, ticks
        eret
# eMIPS: --board malta    (the settings this program runs with)
