# Lab 34: a YAMON application.  The Malta's boot monitor (YAMON on the
# real board, eMIPS's monitor here) has a table of functions at
# 0x1fc00500 for the program it starts: print, getchar, exit, and the
# registration of interrupt and exception handlers.  This program lets
# the monitor dispatch: an ISR for IRQ 8 (the RTC's periodic interrupt,
# through the 8259s: the monitor acknowledges and EOIs) and an ESR for
# the Breakpoint exception.  It uses no system call — only the monitor.
        .eqv  YAMON, 0x9fc00500     # the function table (kseg0)
        .eqv  PRINT, 0x34           # print(port, string)
        .eqv  REGISTER_IC_ISR, 0x40 # register_ic_isr(line, isr, data, &ref)
        .eqv  REGISTER_ESR, 0x48    # register_esr(exception, esr, &ref, &retfunc)

        .text
main:   addiu $sp, $sp, -24         # a C-like function: the monitor returns here
        sw    $ra, 20($sp)
        li    $s7, YAMON

        li    $a0, 8                # IRQ 8: the RTC
        la    $a1, tick
        li    $a2, 0
        la    $a3, tick_ref
        lw    $t9, REGISTER_IC_ISR($s7)
        jalr  $t9

        li    $a0, 9                # ExcCode 9: Breakpoint
        la    $a1, on_break
        la    $a2, break_ref
        la    $a3, ret_func
        lw    $t9, REGISTER_ESR($s7)
        jalr  $t9

        li    $s0, 0xb8000000       # the RTC: 32.768 kHz, rate 3 (8192 Hz),
        li    $t0, 0x0a             # periodic interrupt on
        sb    $t0, 0x70($s0)
        li    $t0, 0x23
        sb    $t0, 0x71($s0)
        li    $t0, 0x0b
        sb    $t0, 0x70($s0)
        li    $t0, 0x42
        sb    $t0, 0x71($s0)

        mfc0  $t0, $12              # interrupts on (the monitor set IM2)
        ori   $t0, $t0, 1
        mtc0  $t0, $12
        li    $s1, 5
wait:   lw    $t0, ticks
        bne   $t0, $s1, wait
        mtc0  $zero, $12            # off again
        break                       # the ESR counts it and steps over it

        lw    $t0, ticks            # "ticks N, breaks M\n" (one digit each)
        addiu $t0, $t0, '0'
        sb    $t0, digits
        lw    $t0, breaks
        addiu $t0, $t0, '0'
        sb    $t0, digits + 10
        li    $a0, 0
        la    $a1, message
        lw    $t9, PRINT($s7)
        jalr  $t9

        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 0                # return to the monitor
        jr    $ra

tick:   li    $t0, 0xb8000000       # the ISR (called by the monitor's handler)
        li    $t1, 0x0c             # RTC register C: clears the flag
        sb    $t1, 0x70($t0)
        lbu   $t1, 0x71($t0)
        lw    $t1, ticks
        addiu $t1, $t1, 1
        sw    $t1, ticks
        jr    $ra                   # the monitor sends the EOIs

on_break:                           # the ESR: count it, continue after it
        lw    $t1, breaks
        addiu $t1, $t1, 1
        sw    $t1, breaks
        mfc0  $t0, $14              # EPC: the break
        addiu $t0, $t0, 4
        mtc0  $t0, $14
        jr    $ra

        .data
message: .ascii "ticks "
digits:  .ascii "0, breaks 0\n"
         .byte 0
        .align 2
ticks:     .word 0
breaks:    .word 0
tick_ref:  .word 0
break_ref: .word 0
ret_func:  .word 0
# eMIPS: --board malta    (the settings this program runs with)
