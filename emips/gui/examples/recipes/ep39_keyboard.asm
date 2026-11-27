# MIPS by Example in eMIPS, episode 39 — An interrupt-driven keyboard.
# Polling asks the keyboard again and again "a key yet?"; an interrupt lets
# the keyboard say it.  The MMIO console raises hardware line 1 (Cause.IP3)
# when a key is ready and its receiver's interrupt is on (RX control bit 1);
# with Status.IM3 and Status.IE set the processor stops whatever it does and
# runs the handler at 0x80000180, which puts the key in a ring buffer and
# returns to the very instruction it interrupted (an interrupt is not a
# failure: EPC is not moved).  main sleeps with WAIT until there is a key,
# and echoes them in capitals until Enter.  The keys come 2000 cycles apart
# (the header below: --key-delay): a real keyboard is slower still; at the
# default 6 cycles the next key would be there before the handler is done
# and main would never run — an interrupt storm.
        .eqv  RX_CONTROL, 0xffff0000  # bit 0: a key is ready, bit 1: interrupt on
        .eqv  RX_DATA, 0xffff0004     # the key (reading it clears bit 0)
        .data
prompt: .asciiz "type, Enter ends: "
keys:   .asciiz "keys: "

        .kdata                        # the handler's data, shared with main
buf:    .space 16                     # the ring buffer: keys in, keys out
head:   .word 0                       # the handler writes at head
tail:   .word 0                       # main reads at tail (tail = head: empty)

        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 0-15 the home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        li    $v0, 4
        la    $a0, prompt
        syscall
        li    $t0, 2                  # the keyboard: interrupt when a key is ready
        sw    $t0, RX_CONTROL
        mfc0  $t0, $12                # Status: IM3 (bit 11) and IE (bit 0)
        ori   $t0, $t0, 0x0801
        mtc0  $t0, $12
        li    $s0, 0                  # keys seen
next:   lw    $t0, head               # a key in the buffer?
        lw    $t1, tail
        bne   $t0, $t1, take
        wait                          # no: sleep until an interrupt
        b     next
take:   lbu   $a0, buf($t1)           # the key at tail ...
        addiu $t1, $t1, 1
        andi  $t1, $t1, 15            # ... tail + 1, around the ring
        sw    $t1, tail
        li    $t0, '\n'
        beq   $a0, $t0, done          # Enter: the end
        addiu $s0, $s0, 1
        jal   upper                   # echo it in capitals
        move  $a0, $v0
        li    $v0, 11
        syscall
        b     next
done:   mfc0  $t0, $12                # interrupts off again: IM3 and IE
        li    $t1, 0xfffff7fe         # cleared, every other bit of Status kept
        and   $t0, $t0, $t1
        mtc0  $t0, $12
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $v0, 4                  # "keys: n"
        la    $a0, keys
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# upper(c): c in capitals when it is a lower-case letter (a leaf)
upper:  move  $v0, $a0
        blt   $a0, 'a', uend
        bgt   $a0, 'z', uend
        addiu $v0, $a0, -32
uend:   jr    $ra

# the interrupt handler: Cause.ExcCode = 0 (an interrupt), Cause.IP3 = the
# keyboard.  It takes the key and goes back to EPC unchanged.  Only $k0 /
# $k1 are its own; $at is saved in $k1 (lw / sw of a label use it).
        .ktext 0x80000180
        .set  noat
        move  $k1, $at
        .set  at
        mfc0  $k0, $13                # Cause
        andi  $k0, $k0, 0x0800        # IP3: the keyboard?
        beqz  $k0, kdone
        sw    $t0, tsave              # one more register: $t0, saved
        lw    $k0, RX_DATA            # the key (the device's Ready bit clears)
        lw    $t0, head
        sb    $k0, buf($t0)           # buf[head] = key
        addiu $t0, $t0, 1
        andi  $t0, $t0, 15            # head + 1, around the ring
        sw    $t0, head
        lw    $t0, tsave
kdone:  .set  noat
        move  $at, $k1
        .set  at
        eret                          # back to the interrupted instruction
        .kdata
tsave:  .word 0
# eMIPS: --key-delay 2000
