# MIPS by Example in eMIPS, episode 40 — the challenge, solved: a countdown.
# 00:05 down to 00:00, then "time's up!"; the handler turns its own interrupt
# off at zero (Status.IM7 cleared).  (The episode: a timer interrupt, a
# clock.)  CP0's
# Count register counts up by itself (one every second cycle); when it
# reaches Compare the timer raises hardware line 5 (Cause.IP7, and Cause.TI).
# With Status.IM7 and IE set the handler runs: it counts a second and moves
# Compare one period on (writing Compare also clears the interrupt).  main
# sleeps with WAIT and prints the time at every new second.  This clock runs
# 100 times fast — a "second" is 10000 cycles — so it does not keep us
# waiting.
        .eqv  PERIOD, 5000            # Count steps a "second" (10000 cycles)
        .eqv  SECONDS, 5              # how long the clock runs
        .data
start:  .asciiz "the countdown starts\n"
stop:   .asciiz "time's up!\n"

        .kdata
seconds: .word SECONDS                # the handler counts down, main reads

        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 0-15 the home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        li    $v0, 4
        la    $a0, start
        syscall
        mfc0  $t0, $9                 # Count
        addiu $t0, $t0, PERIOD        # the first interrupt: one period from now
        mtc0  $t0, $11                # Compare
        mfc0  $t0, $12                # Status: IM7 (bit 15) and IE (bit 0)
        ori   $t0, $t0, 0x8001
        mtc0  $t0, $12
        li    $a0, SECONDS            # the start: 00:05
        jal   print_time
        li    $s0, SECONDS            # the second main printed last
tick:   lw    $t0, seconds
        bne   $t0, $s0, show          # a new second?
        wait                          # no: sleep until an interrupt
        b     tick
show:   move  $s0, $t0
        move  $a0, $t0
        jal   print_time              # "mm:ss"
        bnez  $s0, tick               # (at zero the handler stopped the timer)
        li    $v0, 4
        la    $a0, stop
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# print_time(seconds): "mm:ss" and a new line (a non-leaf: two_digits)
print_time:
        addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        move  $s0, $a0
        li    $t0, 60
        divu  $zero, $s0, $t0         # minutes = s / 60, seconds = s % 60
        mflo  $a0
        jal   two_digits
        li    $v0, 11
        li    $a0, ':'
        syscall
        li    $t0, 60
        divu  $zero, $s0, $t0
        mfhi  $a0
        jal   two_digits
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# two_digits(n): n as two digits, 07 (a leaf)
two_digits:
        li    $t0, 10
        divu  $zero, $a0, $t0
        mflo  $t1                     # the tens
        mfhi  $t2                     # the units
        li    $v0, 11
        addiu $a0, $t1, '0'
        syscall
        addiu $a0, $t2, '0'
        syscall
        jr    $ra

# the interrupt handler: Cause.IP7 = the timer.  A second less; at zero the
# timer's interrupt off (Status.IM7), else Compare one period on.
        .ktext 0x80000180
        .set  noat
        move  $k1, $at
        .set  at
        mfc0  $k0, $13                # Cause
        andi  $k0, $k0, 0x8000        # IP7: the timer?
        beqz  $k0, tdone
        lw    $k0, seconds            # one second less
        addiu $k0, $k0, -1
        sw    $k0, seconds
        bnez  $k0, tnext
        mfc0  $k0, $12                # zero: the timer's interrupt off —
        ori   $k0, $k0, 0x8000        # IM7 set, then flipped: cleared, and
        xori  $k0, $k0, 0x8000        # every other bit of Status kept
        mtc0  $k0, $12
        mtc0  $zero, $11              # (and its pending interrupt cleared)
        b     tdone
tnext:  mfc0  $k0, $11                # Compare + PERIOD: the next second
        addiu $k0, $k0, PERIOD
        mtc0  $k0, $11
tdone:  .set  noat
        move  $at, $k1
        .set  at
        eret
