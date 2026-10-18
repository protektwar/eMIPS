# EJTAG Debug Mode: a tiny debug monitor at the debug vector.
# "sdbbp" enters it; its code field says what to do: 0 = single step
# the program (Debug.SSt = 1), 1 = stop stepping.  Each Debug Single
# Step exception counts one execution step in $k1.
        .text
main:   sdbbp                        # monitor: single step from here
        li    $t0, 4
        li    $t1, 0
loop:   addu  $t1, $t1, $t0          # 4 + 3 + 2 + 1
        addiu $t0, $t0, -1
        bnez  $t0, loop              # a branch and its delay slot: one step
        sdbbp 1                      # monitor: stop stepping
        move  $a0, $t1
        li    $v0, 1
        syscall
        la    $a0, in
        li    $v0, 4
        syscall
        move  $a0, $k1
        li    $v0, 1
        syscall
        la    $a0, steps
        li    $v0, 4
        syscall
        li    $v0, 10
        syscall

        .data
in:     .asciiz " in "
steps:  .asciiz " steps"

        .ktext 0xbfc00480            # the debug vector (Debug Mode)
monitor:
        mtc0  $k0, $31               # DESAVE: one free register
        mfc0  $k0, $23               # Debug: why are we here?
        andi  $k0, $k0, 1            # DSS: a single step was taken
        beqz  $k0, brk
        addiu $k1, $k1, 1            # count it
        b     back
brk:    mfc0  $k0, $24               # DBp: DEPC = the sdbbp
        lw    $k0, 0($k0)
        ext   $k0, $k0, 6, 20        # its code field (bits 25..6)
        bnez  $k0, off
        li    $k0, 0x02000100        # code 0: CountDM, SSt = 1
        b     set
off:    li    $k0, 0x02000000        # code 1: CountDM, SSt = 0
set:    mtc0  $k0, $23
        mfc0  $k0, $24
        addiu $k0, $k0, 4            # return past the sdbbp
        mtc0  $k0, $24
back:   mfc0  $k0, $31               # restore $k0
        ehb
        deret                        # back to Non-Debug Mode, at DEPC
