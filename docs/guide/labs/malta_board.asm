# Lab 30: the MIPS Malta board (run with --board malta).  Count down on
# the LEDs, greet on the ASCII display, then read the word just past the
# end of the SDRAM (128 MB): what happens, and where?
        .text
main:   li    $s0, 0xbf000000       # the board registers (kseg1: uncached)
        li    $t0, 4
loop:   sw    $t0, 0x408($s0)       # LEDBAR: one bit per LED
        addiu $t0, $t0, -1
        bnez  $t0, loop
        la    $t1, msg
        li    $t2, 0x418            # ASCIIPOS0: the left-most character
show:   lbu   $t3, 0($t1)
        beqz  $t3, done
        addu  $t4, $s0, $t2
        sw    $t3, 0($t4)
        addiu $t1, $t1, 1
        addiu $t2, $t2, 8           # the registers sit 8 bytes apart
        b     show
done:   lui   $t5, 0xa800           # kseg1 0xa8000000 = physical 128 MB
        lw    $t6, 0($t5)           # nothing there
        addiu $s1, $zero, 1
        li    $v0, 10
        syscall

        .data
msg:    .asciiz "eMIPS"
taken:  .asciiz "bus error: ExcCode "
at:     .asciiz ", EPC "

        .ktext 0x80000180           # the general exception vector
        la    $a0, taken
        li    $v0, 4
        syscall
        mfc0  $a0, $13              # Cause.ExcCode
        srl   $a0, $a0, 2
        andi  $a0, $a0, 0x1f
        li    $v0, 1
        syscall
        la    $a0, at
        li    $v0, 4
        syscall
        mfc0  $a0, $14              # EPC
        li    $v0, 34
        syscall
        li    $a0, 10
        li    $v0, 11
        syscall
        eret                        # back to EPC
# eMIPS: --board malta    (the settings this program runs with)
