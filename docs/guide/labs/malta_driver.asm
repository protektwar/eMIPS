# Lab 36: an interrupt-driven disk driver.  The bus master reads sector 0
# into a buffer by DMA; the disk's interrupt (IRQ 14 through the 8259s)
# says it is done: the ISR, registered with the boot monitor, stops the
# bus master, acknowledges the disk and sets a flag.  Before the transfer
# the driver writes back and invalidates the data cache (the monitor's
# flush_cache): after it the CPU reads what the DMA wrote.
        .eqv  YAMON, 0x9fc00500
        .eqv  FLUSH_CACHE, 0x2c     # flush_cache(1): the D-cache
        .eqv  PRINT, 0x34
        .eqv  REGISTER_IC_ISR, 0x40

        .text
main:   addiu $sp, $sp, -24
        sw    $ra, 20($sp)
        li    $s7, YAMON
        li    $s0, 0xb8000000       # ISA / PCI I/O ports
        li    $s6, 0x1fffffff       # kseg0 -> physical

        li    $a0, 14               # IRQ 14: the primary IDE channel
        la    $a1, disk_isr
        li    $a2, 0
        li    $a3, 0
        lw    $t9, REGISTER_IC_ISR($s7)
        jalr  $t9

        la    $t0, buf              # the PRD: one region, the buffer, 512
        and   $t0, $t0, $s6         # bytes, EOT (little-endian words)
        wsbh  $t0, $t0
        rotr  $t0, $t0, 16
        la    $t2, prd
        lui   $t3, 0xa000
        or    $t2, $t2, $t3         # through kseg1
        sw    $t0, 0($t2)
        li    $t0, 0x00020080
        sw    $t0, 4($t2)
        la    $t0, prd
        and   $t0, $t0, $s6
        wsbh  $t0, $t0
        rotr  $t0, $t0, 16
        sw    $t0, 0x1004($s0)      # PRD table pointer
        li    $t0, 0x06
        sb    $t0, 0x1002($s0)      # clear Error, Interrupt
        li    $t0, 0x08
        sb    $t0, 0x1000($s0)      # write to memory

        li    $a0, 1                # the buffer's lines out of the cache
        lw    $t9, FLUSH_CACHE($s7)
        jalr  $t9

        mfc0  $t0, $12              # interrupts on
        ori   $t0, $t0, 1
        mtc0  $t0, $12

busy:   lbu   $t0, 0x1f7($s0)       # READ DMA, LBA 0, 1 sector
        andi  $t0, $t0, 0x80
        bnez  $t0, busy
        li    $t0, 0xe0
        sb    $t0, 0x1f6($s0)
        li    $t0, 1
        sb    $t0, 0x1f2($s0)
        sb    $zero, 0x1f3($s0)
        sb    $zero, 0x1f4($s0)
        sb    $zero, 0x1f5($s0)
        li    $t0, 0xc8
        sb    $t0, 0x1f7($s0)
        li    $t0, 0x09
        sb    $t0, 0x1000($s0)      # start the bus master

wait:   lw    $t0, done             # the CPU is free: here it just waits
        beqz  $t0, wait
        mtc0  $zero, $12

        li    $a0, 0                # the sector, read through the cache
        la    $a1, buf
        lw    $t9, PRINT($s7)
        jalr  $t9
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 0
        jr    $ra

disk_isr:                           # IRQ 14: the transfer is over
        li    $t9, 0xb8000000
        lbu   $t0, 0x1002($t9)      # the bus master's status
        sb    $t0, 0x1002($t9)      # write its Interrupt bit back: cleared
        sb    $zero, 0x1000($t9)    # stop the bus master
        lbu   $t1, 0x1f7($t9)       # the disk's Status: acknowledges INTRQ
        sw    $t1, status
        li    $t0, 1
        sw    $t0, done
        jr    $ra

        .data
done:   .word 0
status: .word 0
        .align 5
buf:    .space 513
        .align 3
prd:    .space 8
# eMIPS: --board malta --caches --disk malta_driver.txt    (the settings this program runs with)
