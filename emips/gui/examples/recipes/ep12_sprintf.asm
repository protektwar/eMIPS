# MIPS by Example in eMIPS, episode 12 — A tiny sprintf: tiny_sprintf(buf,
# fmt, ...) understands %d, %s, %c and %%.  Its arguments after the format
# arrive in $a2, $a3 and the caller's frame; it stores $a2 and $a3 into the
# caller's home area, so every argument is one row of words in memory, and
# $s2 walks them: that is what the home area is for.
        .data
buf:    .space 64
fmt1:   .asciiz "%s is %d years old, grade %c%%"
fmt2:   .asciiz "%d + %d = %d"
ada:    .asciiz "Ada"
open:   .asciiz " ("
close:  .asciiz " characters)\n"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp, 0-15 the
        sw    $ra, 28($sp)            # home area, 16 the fifth argument
        sw    $fp, 24($sp)
        move  $fp, $sp
        la    $a0, buf                # tiny_sprintf(buf, fmt1, "Ada", 36, 'A')
        la    $a1, fmt1
        la    $a2, ada
        li    $a3, 36
        li    $t0, 'A'
        sw    $t0, 16($sp)            # the fifth argument: at its place
        jal   tiny_sprintf
        move  $a0, $v0
        jal   show
        la    $a0, buf                # tiny_sprintf(buf, fmt2, 2, 40, 42)
        la    $a1, fmt2
        li    $a2, 2
        li    $a3, 40
        li    $t0, 42
        sw    $t0, 16($sp)
        jal   tiny_sprintf
        move  $a0, $v0
        jal   show
        move  $sp, $fp                # main's epilogue
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# show: print buf and " (n characters)" for n = $a0 (a leaf: system calls)
show:   move  $t0, $a0
        li    $v0, 4
        la    $a0, buf
        syscall
        la    $a0, open
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 4
        la    $a0, close
        syscall
        jr    $ra

# tiny_sprintf: $a0 = buf, $a1 = the format, then its arguments -> buf
# written, $v0 = its length (it calls put_dec: a frame; $s0 = where to
# write, $s1 = in the format, $s2 = the next argument)
tiny_sprintf:
        addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28-20 $s0-$s2,
        sw    $ra, 36($sp)            # 0-15 the home area
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        move  $fp, $sp
        sw    $a0, 40($fp)            # the caller's home area: buf ...
        sw    $a2, 48($fp)            # ... and the first two arguments,
        sw    $a3, 52($fp)            # right below the fifth at 56
        move  $s0, $a0
        move  $s1, $a1
        addiu $s2, $fp, 48            # the arguments, one word each
fnext:  lbu   $t0, 0($s1)
        beqz  $t0, fend
        addiu $s1, $s1, 1
        li    $t1, '%'
        beq   $t0, $t1, percent
        sb    $t0, 0($s0)             # an ordinary character: copied
        addiu $s0, $s0, 1
        b     fnext
percent:
        lbu   $t0, 0($s1)             # the letter after %
        addiu $s1, $s1, 1
        li    $t1, 'd'
        beq   $t0, $t1, pd
        li    $t1, 's'
        beq   $t0, $t1, ps
        li    $t1, 'c'
        beq   $t0, $t1, pc
        sb    $t0, 0($s0)             # %% (or anything else): itself
        addiu $s0, $s0, 1
        b     fnext
pd:     lw    $a1, 0($s2)             # %d: the next argument, in decimal
        addiu $s2, $s2, 4
        move  $a0, $s0
        jal   put_dec
        move  $s0, $v0
        b     fnext
ps:     lw    $t2, 0($s2)             # %s: the next argument is a string
        addiu $s2, $s2, 4
pscopy: lbu   $t3, 0($t2)
        beqz  $t3, fnext
        sb    $t3, 0($s0)
        addiu $t2, $t2, 1
        addiu $s0, $s0, 1
        b     pscopy
pc:     lw    $t2, 0($s2)             # %c: the next argument is a character
        addiu $s2, $s2, 4
        sb    $t2, 0($s0)
        addiu $s0, $s0, 1
        b     fnext
fend:   sb    $zero, 0($s0)           # the 0, and the length
        lw    $t0, 40($fp)
        subu  $v0, $s0, $t0
        move  $sp, $fp
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra

# put_dec: write $a1 in decimal at $a0 -> $v0 = the address after it (a
# leaf: it counts the digits first, then writes them from the right)
put_dec:
        bgez  $a1, pdpos
        li    $t0, '-'
        sb    $t0, 0($a0)
        addiu $a0, $a0, 1
        subu  $a1, $zero, $a1
pdpos:  li    $t2, 10
        move  $t1, $a1
        move  $t3, $a0
pdlen:  divu  $t1, $t2                # one more digit while the quotient is
        mflo  $t1                     # not 0
        addiu $t3, $t3, 1
        bnez  $t1, pdlen
        move  $v0, $t3                # the end
pdput:  divu  $a1, $t2
        mfhi  $t4                     # the last digit ...
        addiu $t4, $t4, '0'
        addiu $t3, $t3, -1            # ... written from the right
        sb    $t4, 0($t3)
        mflo  $a1
        bnez  $a1, pdput
        jr    $ra
