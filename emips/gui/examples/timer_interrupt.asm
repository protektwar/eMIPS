# Interrupts: the CP0 timer.  Count ($9) counts up every 2 cycles; when it
# equals Compare ($11) the timer interrupt (Cause.IP7) is raised.  With
# Status ($12) IE = 1 and IM7 = 1 the CPU jumps to the exception handler
# at 0x80000180, which counts the tick, sets the next Compare and returns
# with eret (EPC = the interrupted instruction).
        .data
ticks:  .word 0
msg:    .asciiz "timer interrupts: "
        .text
main:   mfc0  $t0, $9           # Count
        addiu $t0, $t0, 40
        mtc0  $t0, $11          # Compare = Count + 40
        mfc0  $t0, $12          # Status
        ori   $t0, $t0, 0x8001  # IM7 (timer) | IE
        mtc0  $t0, $12
        la    $s0, ticks
wait:   lw    $t1, 0($s0)       # the handler increments ticks
        slti  $t2, $t1, 3
        bnez  $t2, wait
        li    $v0, 4
        la    $a0, msg
        syscall
        li    $v0, 1
        move  $a0, $t1
        syscall                 # prints 3
        li    $v0, 10
        syscall

        .ktext 0x80000180       # general exception / interrupt vector
handler: la   $k0, ticks        # $k0 / $k1: reserved for the kernel
        lw    $k1, 0($k0)
        addiu $k1, $k1, 1
        sw    $k1, 0($k0)
        mfc0  $k0, $9           # next tick in 40 Count steps;
        addiu $k0, $k0, 40      # writing Compare clears the interrupt
        mtc0  $k0, $11
        eret
