# MIPS by Example in eMIPS, episode 12, the challenge solved: %x (hex,
# without leading zeros) and a width, one digit after %: the item is written
# as before, then moved right and padded with spaces if it is shorter than
# the width.  "[%5d] [%x] [%3s]" with 42, 255, "ab" -> "[   42] [ff] [ ab]".
        .data
buf:    .space 64
fmt:    .asciiz "[%5d] [%x] [%3s] [%c]"
ab:     .asciiz "ab"
hexd:   .ascii "0123456789abcdef"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp, 0-15 the
        sw    $ra, 28($sp)            # home area, 16 the fifth argument
        sw    $fp, 24($sp)
        move  $fp, $sp
        la    $a0, buf                # tiny_sprintf(buf, fmt, 42, 255, "ab", '!')
        la    $a1, fmt
        li    $a2, 42
        li    $a3, 255
        la    $t0, ab
        sw    $t0, 16($sp)
        li    $t0, '!'
        sw    $t0, 20($sp)            # the sixth argument
        jal   tiny_sprintf
        li    $v0, 4
        la    $a0, buf
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# tiny_sprintf: episode 12's, with %x and a width (it calls put_dec,
# put_hex and pad: a frame; $s0 = where to write, $s1 = in the format, $s2 =
# the next argument, $s3 = the width, $s4 = where the item started)
tiny_sprintf:
        addiu $sp, $sp, -48           # 44 $ra, 40 $fp, 36-20 $s0-$s4,
        sw    $ra, 44($sp)            # 0-15 the home area
        sw    $fp, 40($sp)
        sw    $s0, 36($sp)
        sw    $s1, 32($sp)
        sw    $s2, 28($sp)
        sw    $s3, 24($sp)
        sw    $s4, 20($sp)
        move  $fp, $sp
        sw    $a0, 48($fp)            # the caller's home area
        sw    $a2, 56($fp)
        sw    $a3, 60($fp)
        move  $s0, $a0
        move  $s1, $a1
        addiu $s2, $fp, 56
fnext:  lbu   $t0, 0($s1)
        beqz  $t0, fend
        addiu $s1, $s1, 1
        li    $t1, '%'
        beq   $t0, $t1, percent
        sb    $t0, 0($s0)
        addiu $s0, $s0, 1
        b     fnext
percent:
        lbu   $t0, 0($s1)             # a width digit, 1-9?
        addiu $s3, $t0, -48
        sltiu $t1, $s3, 10
        bnez  $t1, width
        li    $s3, 0                  # no width
        b     letter
width:  addiu $s1, $s1, 1
letter: lbu   $t0, 0($s1)
        addiu $s1, $s1, 1
        move  $s4, $s0                # the item starts here
        li    $t1, 'd'
        beq   $t0, $t1, pd
        li    $t1, 'x'
        beq   $t0, $t1, px
        li    $t1, 's'
        beq   $t0, $t1, ps
        li    $t1, 'c'
        beq   $t0, $t1, pc
        sb    $t0, 0($s0)
        addiu $s0, $s0, 1
        b     fnext
pd:     lw    $a1, 0($s2)
        addiu $s2, $s2, 4
        move  $a0, $s0
        jal   put_dec
        move  $s0, $v0
        b     padit
px:     lw    $a1, 0($s2)
        addiu $s2, $s2, 4
        move  $a0, $s0
        jal   put_hex
        move  $s0, $v0
        b     padit
ps:     lw    $t2, 0($s2)
        addiu $s2, $s2, 4
pscopy: lbu   $t3, 0($t2)
        beqz  $t3, padit
        sb    $t3, 0($s0)
        addiu $t2, $t2, 1
        addiu $s0, $s0, 1
        b     pscopy
pc:     lw    $t2, 0($s2)
        addiu $s2, $s2, 4
        sb    $t2, 0($s0)
        addiu $s0, $s0, 1
padit:  move  $a0, $s4                # pad(start, end, width)
        move  $a1, $s0
        move  $a2, $s3
        jal   pad
        move  $s0, $v0
        b     fnext
fend:   sb    $zero, 0($s0)
        lw    $t0, 48($fp)
        subu  $v0, $s0, $t0
        move  $sp, $fp
        lw    $s4, 20($sp)
        lw    $s3, 24($sp)
        lw    $s2, 28($sp)
        lw    $s1, 32($sp)
        lw    $s0, 36($sp)
        lw    $fp, 40($sp)
        lw    $ra, 44($sp)
        addiu $sp, $sp, 48
        jr    $ra

# pad: the item from $a0 to $a1 is moved right and spaces put before it if
# it is shorter than $a2 -> $v0 = its new end (a leaf)
pad:    subu  $t0, $a1, $a0           # its length
        subu  $t1, $a2, $t0           # the spaces needed
        move  $v0, $a1
        blez  $t1, paddone
        addu  $v0, $a1, $t1           # the new end
        move  $t2, $a1                # move it right, the last byte first
pmove:  beq   $t2, $a0, pfill
        addiu $t2, $t2, -1
        lbu   $t3, 0($t2)
        addu  $t4, $t2, $t1
        sb    $t3, 0($t4)
        b     pmove
pfill:  li    $t3, ' '                # then the spaces in front
pspace: sb    $t3, 0($a0)
        addiu $a0, $a0, 1
        addiu $t1, $t1, -1
        bnez  $t1, pspace
paddone:
        jr    $ra

# put_dec: episode 12's (a leaf)
put_dec:
        bgez  $a1, pdpos
        li    $t0, '-'
        sb    $t0, 0($a0)
        addiu $a0, $a0, 1
        subu  $a1, $zero, $a1
pdpos:  li    $t2, 10
        move  $t1, $a1
        move  $t3, $a0
pdlen:  divu  $t1, $t2
        mflo  $t1
        addiu $t3, $t3, 1
        bnez  $t1, pdlen
        move  $v0, $t3
pdput:  divu  $a1, $t2
        mfhi  $t4
        addiu $t4, $t4, '0'
        addiu $t3, $t3, -1
        sb    $t4, 0($t3)
        mflo  $a1
        bnez  $a1, pdput
        jr    $ra

# put_hex: write $a1 in hex, no leading zeros, at $a0 -> $v0 = the address
# after it (a leaf: the nibbles from the top, skipping the leading 0s)
put_hex:
        la    $t1, hexd
        li    $t2, 28                 # the shift of the next nibble
        li    $t5, 0                  # 1 once a digit has been written
phnext: srlv  $t0, $a1, $t2
        andi  $t0, $t0, 0xf
        bnez  $t0, phput
        bnez  $t5, phput
        beqz  $t2, phput              # the last nibble: always written
        b     phstep
phput:  addu  $t0, $t1, $t0
        lbu   $t0, 0($t0)
        sb    $t0, 0($a0)
        addiu $a0, $a0, 1
        li    $t5, 1
phstep: addiu $t2, $t2, -4
        bgez  $t2, phnext
        move  $v0, $a0
        jr    $ra
