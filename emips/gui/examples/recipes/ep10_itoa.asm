# MIPS by Example in eMIPS, episode 10 — Numbers to text: decimal, binary,
# hex.  utoa divides by the base again and again: each remainder is a digit,
# the last one first — so the digits go into the buffer from its end.
        .data
numbers:.word 2026, -42, 0
digits: .ascii "0123456789abcdef"
buf:    .space 34                     # 32 binary digits, a sign, the 0
equals: .asciiz " = 0b"
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
        li    $s1, 2                  # two numbers
next:   lw    $s2, 0($s0)
        bgez  $s2, positive           # decimal: the sign, then |n|
        li    $v0, 11
        li    $a0, '-'
        syscall
positive:
        abs   $a0, $s2
        li    $a1, 10
        jal   print_base
        li    $v0, 4
        la    $a0, equals
        syscall
        move  $a0, $s2                # binary: the bits as they are
        li    $a1, 2
        jal   print_base
        li    $v0, 4
        la    $a0, hex
        syscall
        move  $a0, $s2                # hex
        li    $a1, 16
        jal   print_base
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

# print_base: print $a0 (unsigned) in base $a1 (it calls utoa: a frame)
print_base:
        addiu $sp, $sp, -32           # the frame: 28 $ra, 24 $fp, 0-15 home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        move  $fp, $sp
        la    $a2, buf + 33           # the end of the buffer
        jal   utoa
        move  $a0, $v0                # the first digit
        li    $v0, 4
        syscall
        move  $sp, $fp                # restore, in reverse
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# utoa: $a0 = a number (unsigned), $a1 = the base, $a2 = the end of a
# buffer -> the digits, ending with a 0 at $a2; $v0 = the first digit
# (a leaf: no frame)
utoa:   sb    $zero, 0($a2)           # the 0 at the end
        la    $t1, digits
digit:  divu  $a0, $a1                # the quotient in lo, the remainder in hi
        mfhi  $t0                     # the remainder: the next digit, right to left
        addu  $t0, $t1, $t0
        lbu   $t0, 0($t0)             # its character
        addiu $a2, $a2, -1
        sb    $t0, 0($a2)
        mflo  $a0                     # go on with the quotient
        bnez  $a0, digit              # until it is 0
        move  $v0, $a2
        jr    $ra
