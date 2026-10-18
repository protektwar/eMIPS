# Lab 37: a user program and its kernel.  The user program runs in user
# mode in kuseg (the TLB maps it into the Malta's SDRAM); it cannot touch
# the devices or CP0, so it asks the kernel with SYSCALL: service 100
# prints a number, 101 ends the program.  The kernel is the exception
# handler at 0x80000180: it serves the system calls (with the boot
# monitor's print), saving and restoring the user's registers it uses,
# and kills the program on any other exception.
        .eqv  YAMON, 0x9fc00500
        .eqv  PRINT, 0x34
        .eqv  EXIT, 0x20

        .text                       # the user program (0x00400000)
main:   li    $s0, 1                # 1 + 2 + ... + 10
        li    $s1, 0
loop:   addu  $s1, $s1, $s0
        addiu $s0, $s0, 1
        slti  $t0, $s0, 11
        bnez  $t0, loop
        move  $a0, $s1
        li    $v0, 100              # the kernel: print a number
        syscall
        mfc0  $t0, $12              # not allowed in user mode
        li    $v0, 101              # (never reached)
        syscall

        .ktext 0x80000180           # the kernel
        .set  noat
        lui   $k1, %hi(saved)       # the user's registers it will use
        addiu $k1, $k1, %lo(saved)
        sw    $at, 0($k1)
        sw    $a0, 4($k1)
        sw    $a1, 8($k1)
        sw    $t1, 12($k1)
        sw    $t2, 16($k1)
        sw    $t3, 20($k1)
        sw    $t9, 24($k1)
        sw    $ra, 28($k1)
        sw    $sp, 32($k1)
        mfhi  $t1
        sw    $t1, 36($k1)
        mflo  $t1
        sw    $t1, 40($k1)
        .set  at
        la    $sp, kstack_top       # its own stack (in kseg0)
        mfc0  $k0, $13              # Cause.ExcCode
        srl   $k0, $k0, 2
        andi  $k0, $k0, 0x1f
        li    $k1, 8                # Sys: a system call
        bne   $k0, $k1, kill
        li    $k1, 100
        beq   $v0, $k1, k_print
        li    $k1, 101
        beq   $v0, $k1, k_exit
kill:   la    $a1, killed           # anything else ends the program
        jal   k_puts
        move  $a0, $k0
        jal   k_putnum
        li    $a0, 1
k_exit: li    $t9, YAMON
        lw    $t9, EXIT($t9)
        jr    $t9                   # exit(a0): never comes back

k_print:                            # print a0 in decimal, then return
        jal   k_putnum
        la    $a1, newline
        jal   k_puts
        mfc0  $k0, $14              # after the syscall
        addiu $k0, $k0, 4
        mtc0  $k0, $14
        .set  noat
        lui   $k1, %hi(saved)       # the user's registers back
        addiu $k1, $k1, %lo(saved)
        lw    $t1, 36($k1)
        mthi  $t1
        lw    $t1, 40($k1)
        mtlo  $t1
        lw    $at, 0($k1)
        lw    $a0, 4($k1)
        lw    $a1, 8($k1)
        lw    $t1, 12($k1)
        lw    $t2, 16($k1)
        lw    $t3, 20($k1)
        lw    $t9, 24($k1)
        lw    $ra, 28($k1)
        lw    $sp, 32($k1)
        .set  at
        sll   $zero, $zero, 3       # ehb
        eret

k_putnum:                           # a0: an unsigned number, in decimal
        la    $t1, digits_end
        sb    $zero, 0($t1)
1:      addiu $t1, $t1, -1
        li    $t2, 10
        divu  $a0, $t2
        mfhi  $t3
        mflo  $a0
        addiu $t3, $t3, '0'
        sb    $t3, 0($t1)
        bnez  $a0, 1b
        move  $a1, $t1
k_puts:                             # a1: a string, through the monitor
        addiu $sp, $sp, -8
        sw    $ra, 4($sp)
        li    $a0, 0
        li    $t9, YAMON
        lw    $t9, PRINT($t9)
        jalr  $t9
        lw    $ra, 4($sp)
        addiu $sp, $sp, 8
        jr    $ra

        .kdata
saved:      .space 44               # at a0 a1 t1 t2 t3 t9 ra sp hi lo
killed:     .asciiz "killed by the kernel: ExcCode "
newline:    .asciiz "\n"
digits:     .space 11
digits_end: .byte 0
        .align 2
kstack:     .space 256
kstack_top:
# eMIPS: --board malta --boot user --mmu tlb    (the settings this program runs with)
