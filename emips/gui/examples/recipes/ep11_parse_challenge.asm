# MIPS by Example in eMIPS, episode 11, the challenge solved.  parse_num
# reads decimal or hex: "0x" after the sign sets the base to 16, and a hex
# digit's value is 10 to 15 for a-f (or A-F).  main prints the count, the
# sum and the average: div puts the quotient in LO and the remainder in HI.
        .data
buf:    .space 64
ask:    .asciiz "Numbers? "
cnt:    .asciiz "count "
sums:   .asciiz ", sum "
avg:    .asciiz ", average "
rem:    .asciiz " remainder "
        .text
main:   addiu $sp, $sp, -40           # main's frame: 36 $ra, 32 $fp,
        sw    $ra, 36($sp)            # 28-20 $s0-$s2, 0-15 the home area
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        move  $fp, $sp
        li    $v0, 4                  # ask, read a line
        la    $a0, ask
        syscall
        li    $v0, 8
        la    $a0, buf
        li    $a1, 64
        syscall
        la    $s0, buf
        li    $s1, 0                  # the count
        li    $s2, 0                  # the sum
next:   move  $a0, $s0                # parse_num(p): $v0 = n, $v1 = after it
        jal   parse_num
        beqz  $v1, done
        move  $s0, $v1
        addiu $s1, $s1, 1
        addu  $s2, $s2, $v0
        b     next
done:   li    $v0, 4                  # count, sum
        la    $a0, cnt
        syscall
        li    $v0, 1
        move  $a0, $s1
        syscall
        li    $v0, 4
        la    $a0, sums
        syscall
        li    $v0, 1
        move  $a0, $s2
        syscall
        div   $s2, $s1                # LO = sum / count, HI = the remainder
        li    $v0, 4
        la    $a0, avg
        syscall
        li    $v0, 1
        mflo  $a0
        syscall
        li    $v0, 4
        la    $a0, rem
        syscall
        li    $v0, 1
        mfhi  $a0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        li    $v0, 10
        syscall

# digit: $a0 = a character, $a1 = the base -> $v0 = its value, or -1 if it
# is not a digit of that base (a leaf)
digit:  addiu $v0, $a0, -48           # 0-9?
        sltiu $t0, $v0, 10
        bnez  $t0, dcheck
        ori   $v0, $a0, 0x20          # a-f (A-F folded)?
        addiu $v0, $v0, -87           # 'a' - 10
        sltiu $t0, $v0, 10            # below 10: not a letter digit
        bnez  $t0, dnone
        sltiu $t0, $v0, 16
        beqz  $t0, dnone
dcheck: slt   $t0, $v0, $a1           # a digit of this base?
        bnez  $t0, dok
dnone:  li    $v0, -1
dok:    jr    $ra

# parse_num: $a0 = a string -> $v0 = the number at its start, decimal or
# 0x hex (spaces skipped), $v1 = the address after it, or 0 (it calls digit:
# a frame; $s0 = where it is, $s1 = the base, $s2 = the value, $s3 = 1 if
# negative)
parse_num:
        addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28-16 $s0-$s3,
        sw    $ra, 36($sp)            # 0-15 the home area
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        sw    $s3, 16($sp)
        move  $fp, $sp
        move  $s0, $a0
        li    $t1, ' '
pskip:  lbu   $t0, 0($s0)
        bne   $t0, $t1, psign
        addiu $s0, $s0, 1
        b     pskip
psign:  li    $s3, 0
        li    $t1, '-'
        bne   $t0, $t1, pbase
        li    $s3, 1
        addiu $s0, $s0, 1
pbase:  li    $s1, 10                 # decimal, unless "0x"
        lbu   $t0, 0($s0)
        li    $t1, '0'
        bne   $t0, $t1, pfirst
        lbu   $t0, 1($s0)
        li    $t1, 'x'
        bne   $t0, $t1, pfirst
        li    $s1, 16
        addiu $s0, $s0, 2
pfirst: lbu   $a0, 0($s0)             # at least one digit
        move  $a1, $s1
        jal   digit
        bltz  $v0, pnone
        li    $s2, 0
pdigit: lbu   $a0, 0($s0)
        move  $a1, $s1
        jal   digit
        bltz  $v0, pend
        mul   $s2, $s2, $s1           # n * base + the digit
        addu  $s2, $s2, $v0
        addiu $s0, $s0, 1
        b     pdigit
pend:   move  $v0, $s2
        beqz  $s3, ppos
        subu  $v0, $zero, $v0
ppos:   move  $v1, $s0
        b     pret
pnone:  move  $v1, $zero
pret:   move  $sp, $fp
        lw    $s3, 16($sp)
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra
