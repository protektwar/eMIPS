# MIPS by Example in eMIPS, episode 44 — From text to a float.  "3.14159"
# is seven characters; the FPU wants a number.  The digits make an integer,
# 314159, counting the digits after the point: 5.  The integer becomes a
# float (mtc1, then cvt.s.w: the bits of 314159 are not the float 314159),
# and one division by 10^5 puts the point back.  10^5 is built by
# multiplying 10.0 — exact up to 10^10 — so the division is the only
# rounding: the nearest float to the text.
        .data
inputs: .word in1, in2, in3, in4, in5, 0   # the texts, 0 at the end
in1:    .asciiz "3.14159"
in2:    .asciiz "-0.5"
in3:    .asciiz "100"
in4:    .asciiz "2.75"
in5:    .asciiz "0.1"
arrow:  .asciiz " -> "
equals: .asciiz " = "
        .align 2
ten:    .float 10.0
one:    .float 1.0

        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s1, 16 $s0, 0-15 the home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s1, 20($sp)
        sw    $s0, 16($sp)
        move  $fp, $sp
        la    $s0, inputs             # $s0: the next text's address in the list
next:   lw    $s1, 0($s0)             # $s1: the text
        beqz  $s1, done
        li    $v0, 4                  # "3.14159 -> "
        move  $a0, $s1
        syscall
        li    $v0, 4
        la    $a0, arrow
        syscall
        move  $a0, $s1                # $f0 = parse_float(text)
        jal   parse_float
        li    $v0, 2                  # the float, as a float
        mov.s $f12, $f0
        syscall
        li    $v0, 4                  # " = " and its bits
        la    $a0, equals
        syscall
        li    $v0, 34
        mfc1  $a0, $f0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        addiu $s0, $s0, 4
        b     next
done:   move  $sp, $fp                # main's epilogue
        lw    $s0, 16($sp)
        lw    $s1, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# parse_float(text) -> $f0: [-]digits[.digits] (a leaf)
#   $t0 the digits as an integer, $t1 the digits after the point,
#   $t2 1 after the point, $t3 1 for a minus sign
parse_float:
        li    $t0, 0
        li    $t1, 0
        li    $t2, 0
        li    $t3, 0
        lbu   $t4, 0($a0)
        bne   $t4, '-', digits        # a sign?
        li    $t3, 1
        addiu $a0, $a0, 1
digits: lbu   $t4, 0($a0)
        bne   $t4, '.', digit
        li    $t2, 1                  # the point: count the digits after it
        addiu $a0, $a0, 1
        b     digits
digit:  blt   $t4, '0', end           # not a digit: the end
        bgt   $t4, '9', end
        mul   $t0, $t0, 10            # the integer: × 10 + the digit
        addiu $t4, $t4, -48
        addu  $t0, $t0, $t4
        addu  $t1, $t1, $t2           # one more after the point (if past it)
        addiu $a0, $a0, 1
        b     digits
end:    mtc1  $t0, $f0                # the integer's bits into the FPU ...
        cvt.s.w $f0, $f0              # ... made a float: 314159.0
        lwc1  $f2, one                # 10 ^ (digits after the point)
        lwc1  $f4, ten
power:  beqz  $t1, scale
        mul.s $f2, $f2, $f4           # exact: 10.0 × 10.0 ... up to 10^10
        addiu $t1, $t1, -1
        b     power
scale:  div.s $f0, $f0, $f2           # one rounding: the nearest float
        beqz  $t3, pdone
        neg.s $f0, $f0                # the sign
pdone:  jr    $ra
