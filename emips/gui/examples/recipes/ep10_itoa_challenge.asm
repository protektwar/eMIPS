# MIPS by Example in eMIPS, episode 10, the challenge solved.  itoa writes a
# signed number in decimal with its minus sign into the buffer.  Episode 10's
# main used abs, which is sub: -(-2147483648) does not fit in 32 bits, so
# sub raises an Arithmetic Overflow exception.  subu wraps instead: 0 minus
# 0x80000000 is 0x80000000, which read unsigned is exactly 2147483648 — and
# utoa divides unsigned.  hex8 writes exactly 8 hex digits with shifts and
# masks: no divu.
        .data
numbers:.word 2026, -42, -2147483648, 0
digits: .ascii "0123456789abcdef"
buf:    .space 12                     # a sign, 10 digits, the 0
hexbuf: .space 9                      # 8 digits, the 0
hex:    .asciiz " = 0x"
        .text
main:   addiu $sp, $sp, -40           # main's frame: 36 $ra, 32 $fp,
        sw    $ra, 36($sp)            # 28-20 $s0-$s2, 0-15 the home area
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        move  $fp, $sp
        la    $s0, numbers
        li    $s1, 4                  # four numbers
next:   lw    $s2, 0($s0)
        move  $a0, $s2                # itoa(n, the buffer's end)
        la    $a1, buf + 11
        jal   itoa
        move  $a0, $v0                # print from its first character
        li    $v0, 4
        syscall
        li    $v0, 4
        la    $a0, hex
        syscall
        move  $a0, $s2                # hex8(n, hexbuf)
        la    $a1, hexbuf
        jal   hex8
        li    $v0, 4
        la    $a0, hexbuf
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, next
        move  $sp, $fp                # main's epilogue
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        li    $v0, 10
        syscall

# itoa: $a0 = a signed number, $a1 = the end of a buffer -> its decimal
# digits (with - if negative) ending with a 0 at $a1; $v0 = the first
# character (it calls utoa: a frame; the sign kept in the home area)
itoa:   addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        sw    $a0, 24($fp)            # n, in the caller's home area
        bgez  $a0, unsigned
        subu  $a0, $zero, $a0         # |n|, wrapping: no overflow
unsigned:
        move  $a2, $a1                # utoa(|n|, 10, end)
        li    $a1, 10
        jal   utoa
        lw    $t0, 24($fp)            # was it negative?
        bgez  $t0, itdone
        addiu $v0, $v0, -1            # the sign before the first digit
        li    $t0, '-'
        sb    $t0, 0($v0)
itdone: move  $sp, $fp
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        jr    $ra

# utoa: episode 10's (a leaf)
utoa:   sb    $zero, 0($a2)
        la    $t1, digits
digit:  divu  $a0, $a1
        mfhi  $t0
        addu  $t0, $t1, $t0
        lbu   $t0, 0($t0)
        addiu $a2, $a2, -1
        sb    $t0, 0($a2)
        mflo  $a0
        bnez  $a0, digit
        move  $v0, $a2
        jr    $ra

# hex8: $a0 = a number, $a1 = a 9-byte buffer -> 8 hex digits and a 0
# (a leaf: the top nibble first, by shifting right 28, 24, ... 0)
hex8:   la    $t1, digits
        li    $t2, 28                 # the shift of the next nibble
h8next: srlv  $t0, $a0, $t2
        andi  $t0, $t0, 0xf           # one hex digit's value
        addu  $t0, $t1, $t0
        lbu   $t0, 0($t0)
        sb    $t0, 0($a1)
        addiu $a1, $a1, 1
        addiu $t2, $t2, -4
        bgez  $t2, h8next
        sb    $zero, 0($a1)
        jr    $ra
