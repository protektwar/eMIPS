# MIPS by Example in eMIPS, episode 11 — Text to numbers: parse a line, min
# and max.  parse_int reads one number from a string: spaces skipped, an
# optional minus sign, then digits, n = n * 10 + digit.  It returns two
# results: the number in $v0 and where it stopped in $v1 (0: no number).
        .data
buf:    .space 64
ask:    .asciiz "Numbers? "
cnt:    .asciiz "count "
mins:   .asciiz ", min "
maxs:   .asciiz ", max "
sums:   .asciiz ", sum "
        .text
main:   addiu $sp, $sp, -48           # main's frame: 44 $ra, 40 $fp,
        sw    $ra, 44($sp)            # 36-20 $s0-$s4, 0-15 the home area
        sw    $fp, 40($sp)
        sw    $s0, 36($sp)
        sw    $s1, 32($sp)
        sw    $s2, 28($sp)
        sw    $s3, 24($sp)
        sw    $s4, 20($sp)
        move  $fp, $sp
        li    $v0, 4                  # ask, read a line
        la    $a0, ask
        syscall
        li    $v0, 8
        la    $a0, buf
        li    $a1, 64
        syscall
        la    $s0, buf                # where the next number starts
        li    $s1, 0                  # the count
        li    $s2, 0x7fffffff         # the minimum so far: the largest int
        li    $s3, 0x80000000         # the maximum so far: the smallest int
        li    $s4, 0                  # the sum
next:   move  $a0, $s0                # parse_int(p): $v0 = n, $v1 = after it
        jal   parse_int
        beqz  $v1, done               # no more numbers
        move  $s0, $v1
        addiu $s1, $s1, 1
        addu  $s4, $s4, $v0
        bge   $v0, $s2, notmin
        move  $s2, $v0
notmin: ble   $v0, $s3, next
        move  $s3, $v0
        b     next
done:   li    $v0, 4                  # count n, min a, max b, sum s
        la    $a0, cnt
        syscall
        li    $v0, 1
        move  $a0, $s1
        syscall
        li    $v0, 4
        la    $a0, mins
        syscall
        li    $v0, 1
        move  $a0, $s2
        syscall
        li    $v0, 4
        la    $a0, maxs
        syscall
        li    $v0, 1
        move  $a0, $s3
        syscall
        li    $v0, 4
        la    $a0, sums
        syscall
        li    $v0, 1
        move  $a0, $s4
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s4, 20($sp)
        lw    $s3, 24($sp)
        lw    $s2, 28($sp)
        lw    $s1, 32($sp)
        lw    $s0, 36($sp)
        lw    $fp, 40($sp)
        lw    $ra, 44($sp)
        addiu $sp, $sp, 48
        li    $v0, 10
        syscall

# parse_int: $a0 = a string -> $v0 = the number at its start (spaces
# skipped), $v1 = the address after it, or 0 if there is no number (a leaf)
parse_int:
        li    $t1, ' '
skip:   lbu   $t0, 0($a0)             # skip the spaces
        bne   $t0, $t1, sign
        addiu $a0, $a0, 1
        b     skip
sign:   li    $t2, 0                  # $t2 = 1: negative
        li    $t1, '-'
        bne   $t0, $t1, first
        li    $t2, 1
        addiu $a0, $a0, 1
        lbu   $t0, 0($a0)
first:  addiu $t3, $t0, -48           # a digit? c - '0' < 10 (unsigned)
        sltiu $t4, $t3, 10
        beqz  $t4, none               # no digit: no number here
        li    $v0, 0
digit:  lbu   $t0, 0($a0)
        addiu $t3, $t0, -48           # the digit's value
        sltiu $t4, $t3, 10
        beqz  $t4, pdone
        sll   $t5, $v0, 3             # n * 10 = n * 8 + n * 2
        sll   $t6, $v0, 1
        addu  $v0, $t5, $t6
        addu  $v0, $v0, $t3           # + the digit
        addiu $a0, $a0, 1
        b     digit
pdone:  beqz  $t2, positive
        subu  $v0, $zero, $v0         # the minus sign
positive:
        move  $v1, $a0
        jr    $ra
none:   move  $v1, $zero
        jr    $ra
