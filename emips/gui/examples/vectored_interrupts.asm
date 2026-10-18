# Vectored interrupts and GPR shadow sets (Release 2 and later).
#
# IntCtl.VS = 4: the interrupt vectors are 128 bytes apart, from
# EBase + 0x200: vector n (= the IP bit: 7 the timer, 1 software
# interrupt 1) is at EBase + 0x200 + n * 0x80.  SRSMap gives each vector
# its own register set: the handlers need not save a register, and the
# timer handler keeps its count in "its" $s0 from one interrupt to the
# next.  Step it and watch Registers > Shadow sets.
        .text
main:   li    $t0, 4 << 5            # IntCtl.VS = 4: 128-byte spacing
        mtc0  $t0, $12, 1
        li    $t0, 0x20000010        # SRSMap: vector 7 -> set 2, vector 1 -> set 1
        mtc0  $t0, $12, 3
        li    $t0, 0x00800000        # Cause.IV: vectored interrupts
        mtc0  $t0, $13
        mtc0  $zero, $9              # Count = 0
        li    $t0, 100
        mtc0  $t0, $11               # Compare: the first timer interrupt
        li    $t0, 0x8201            # IM7 (timer), IM1 (software 1), IE
        mtc0  $t0, $12
wait:   beq   $s1, $zero, wait       # set 1's handler sets our $s1
        nop
        li    $t0, 2 << 6            # SRSCtl.PSS = 2: the timer's set ...
        mtc0  $t0, $12, 2
        la    $t1, read
        jr.hb $t1                    # (clear the hazard before rdpgpr)
read:   rdpgpr $a1, $s0              # ... its $s0
        la    $a0, ticks
        li    $v0, 4
        syscall
        move  $a0, $a1
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall

        .data
ticks:  .asciiz "timer ticks: "

        .ktext 0x80000280            # vector 1: software interrupt 1, set 1
        mfc0  $t0, $13
        andi  $t0, $t0, 0xfdff       # clear IP1 (keep IV)
        lui   $t1, 0x0080
        or    $t0, $t0, $t1
        mtc0  $t0, $13
        mfc0  $t0, $12               # the timer off: clear IM7
        andi  $t0, $t0, 0x7fff
        mtc0  $t0, $12
        li    $t2, 1
        wrpgpr $s1, $t2              # main's $s1 (the previous set) = 1
        ehb
        eret

        .ktext 0x80000580            # vector 7: the timer, set 2
        addiu $s0, $s0, 1            # survives: nobody else uses set 2
        mfc0  $t0, $11
        addiu $t0, $t0, 100          # the next tick (clears the interrupt)
        mtc0  $t0, $11
        li    $t1, 3
        bne   $s0, $t1, out
        mfc0  $t0, $13               # third tick: request software interrupt 1
        ori   $t0, $t0, 0x0200
        mtc0  $t0, $13
out:    ehb
        eret
