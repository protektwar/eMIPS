# Lab 35: an interrupt-driven serial driver.  tty0's receiver interrupts
# (IRQ 4 through the 8259s) when a key arrives; the ISR, registered with
# the boot monitor, moves the characters into a ring buffer; main takes
# them out, sends them back in upper case (through the monitor's
# print_count) and stops at a '.'.
        .eqv  YAMON, 0x9fc00500
        .eqv  PRINT_COUNT, 0x04     # print_count(port, s, count)
        .eqv  REGISTER_IC_ISR, 0x40

        .text
main:   addiu $sp, $sp, -24
        sw    $ra, 20($sp)
        li    $s7, YAMON
        li    $s0, 0xb8000000       # ISA ports: tty0 at 0x3f8

        li    $a0, 4                # IRQ 4: tty0
        la    $a1, rx_isr
        li    $a2, 0
        li    $a3, 0
        lw    $t9, REGISTER_IC_ISR($s7)
        jalr  $t9

        li    $t0, 0x01             # IER: interrupt on received data
        sb    $t0, 0x3f9($s0)
        li    $t0, 0x0b             # MCR: DTR, RTS, OUT2 (the PC's IRQ gate)
        sb    $t0, 0x3fc($s0)
        mfc0  $t0, $12              # interrupts on
        ori   $t0, $t0, 1
        mtc0  $t0, $12

next:   lw    $t0, head             # wait for a character in the buffer
        lw    $t1, tail
        beq   $t0, $t1, next
        la    $t2, ring             # take it (tail moves: only main writes it)
        andi  $t3, $t1, 15
        addu  $t2, $t2, $t3
        lbu   $s1, 0($t2)
        addiu $t1, $t1, 1
        sw    $t1, tail
        li    $t0, 'a'              # lower case -> upper case
        sltu  $t0, $s1, $t0
        bnez  $t0, send
        li    $t0, 'z' + 1
        sltu  $t0, $s1, $t0
        beqz  $t0, send
        addiu $s1, $s1, 'A' - 'a'
send:   sb    $s1, out
        li    $a0, 0
        la    $a1, out
        li    $a2, 1
        lw    $t9, PRINT_COUNT($s7)
        jalr  $t9
        li    $t0, '.'
        bne   $s1, $t0, next

        mtc0  $zero, $12            # done: interrupts off, the receiver quiet
        sb    $zero, 0x3f9($s0)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 0
        jr    $ra

rx_isr: li    $t9, 0xb8000000       # every character the receiver has (LSR.DR):
more:   lbu   $t0, 0x3fd($t9)       # reading RBR takes it and clears the
        andi  $t0, $t0, 0x01        # interrupt
        beqz  $t0, back
        lbu   $t1, 0x3f8($t9)
        lw    $t2, head             # into the ring (head moves: only the ISR
        la    $t3, ring             # writes it)
        andi  $t4, $t2, 15
        addu  $t3, $t3, $t4
        sb    $t1, 0($t3)
        addiu $t2, $t2, 1
        sw    $t2, head
        b     more
back:   jr    $ra

        .data
head:   .word 0
tail:   .word 0
ring:   .space 16
out:    .byte 0
# eMIPS: --board malta    (the settings this program runs with)
