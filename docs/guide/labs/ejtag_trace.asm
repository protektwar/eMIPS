# Lab 26: a single-step tracer.  The debug monitor (Debug Mode, at the
# debug vector) single-steps the code between "sdbbp" and "sdbbp 1" and
# writes the DEPC of every Debug Single Step exception into "trace";
# then main prints them.
        .text
main:   sdbbp                        # monitor: single step from here
        li    $a0, 5
        jal   double
        addiu $a0, $v0, 1
        sdbbp 1                      # monitor: stop stepping
        la    $s0, trace
show:   beq   $s0, $k1, done         # $k1: the end of the trace
        lw    $a0, 0($s0)
        li    $v0, 34                # print_hex
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        addiu $s0, $s0, 4
        b     show
done:   li    $v0, 10
        syscall

double: addu  $v0, $a0, $a0
        jr    $ra

        .data
trace:  .space 64

        .ktext 0xbfc00480            # the debug vector
monitor:
        mtc0  $k0, $31               # DESAVE: one free register
        bnez  $k1, logged            # $k1 = 0: the first entry
        la    $k1, trace
logged: mfc0  $k0, $23               # Debug
        andi  $k0, $k0, 1            # DSS?
        beqz  $k0, brk
        mfc0  $k0, $24               # DSS: log DEPC, the next instruction
        sw    $k0, 0($k1)
        addiu $k1, $k1, 4
        b     back
brk:    mfc0  $k0, $24               # DBp: DEPC = the sdbbp
        lw    $k0, 0($k0)
        ext   $k0, $k0, 6, 20        # its code: 0 step, 1 stop
        bnez  $k0, off
        li    $k0, 0x02000100        # CountDM, SSt = 1
        b     set
off:    li    $k0, 0x02000000        # CountDM, SSt = 0
set:    mtc0  $k0, $23
        mfc0  $k0, $24
        addiu $k0, $k0, 4            # past the sdbbp
        mtc0  $k0, $24
back:   mfc0  $k0, $31
        ehb
        deret
