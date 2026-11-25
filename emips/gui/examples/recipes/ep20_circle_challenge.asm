# MIPS by Example in eMIPS, episode 20, the challenge solved.  fahrenheit
# with doubles: 37 °C is 98.6 °F, the fraction kept this time.  roots
# solves a x² + b x + c = 0: the integers become doubles (mtc1, cvt.d.w),
# the discriminant b² − 4ac decides (c.lt.d, bc1t), sqrt.d does the rest.
        .data
temps:  .double 37.0, -40.0, 100.0
nine:   .double 9.0
five:   .double 5.0
f32:    .double 32.0
quads:  .word 1, -3, 2, 1, 2, 5
cw:     .asciiz " C = "
fw:     .asciiz " F\n"
rootsw: .asciiz "roots of "
colon:  .asciiz ":"
nonew:  .asciiz " none\n"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $s0, temps              # each temperature in Fahrenheit
        li    $s1, 3
tnext:  ldc1  $f12, 0($s0)            # (.double: 8-aligned in .data)
        li    $v0, 3
        syscall
        li    $v0, 4
        la    $a0, cw
        syscall
        ldc1  $f12, 0($s0)
        jal   fahrenheit
        mov.d $f12, $f0
        li    $v0, 3
        syscall
        li    $v0, 4
        la    $a0, fw
        syscall
        addiu $s0, $s0, 8
        addiu $s1, $s1, -1
        bnez  $s1, tnext
        la    $s0, quads              # each equation's roots
        li    $s1, 2
qnext:  lw    $a0, 0($s0)
        lw    $a1, 4($s0)
        lw    $a2, 8($s0)
        jal   roots
        addiu $s0, $s0, 12
        addiu $s1, $s1, -1
        bnez  $s1, qnext
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# fahrenheit: $f12 = degrees C -> $f0 = C × 9 / 5 + 32 (a leaf)
fahrenheit:
        ldc1  $f2, nine
        mul.d $f0, $f12, $f2
        ldc1  $f2, five
        div.d $f0, $f0, $f2           # the fraction stays: 66.6
        ldc1  $f2, f32
        add.d $f0, $f0, $f2
        jr    $ra

# roots: print "roots of a b c:" and the real roots of a x² + b x + c, or
# none ($a0-$a2 = a, b, c, integers; a leaf: system calls only)
roots:  move  $t0, $a0
        li    $v0, 4
        la    $a0, rootsw
        syscall
        li    $v0, 1                  # a b c
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        move  $a0, $a1
        syscall
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        move  $a0, $a2
        syscall
        li    $v0, 4
        la    $a0, colon
        syscall
        mtc1  $t0, $f4                # the integers into the FPU ...
        cvt.d.w $f4, $f4              # ... and converted: a
        mtc1  $a1, $f6
        cvt.d.w $f6, $f6              # b
        mtc1  $a2, $f8
        cvt.d.w $f8, $f8              # c
        mul.d $f10, $f6, $f6          # b²
        mul.d $f16, $f4, $f8          # ac
        add.d $f16, $f16, $f16        # 2ac
        add.d $f16, $f16, $f16        # 4ac
        sub.d $f10, $f10, $f16        # the discriminant
        mtc1  $zero, $f16             # 0.0 (all zero bits)
        mtc1  $zero, $f17
        c.lt.d $f10, $f16             # below 0: no real root
        bc1t  rnone
        sqrt.d $f10, $f10
        add.d $f4, $f4, $f4           # 2a
        neg.d $f6, $f6                # −b
        add.d $f12, $f6, $f10         # (−b + √) / 2a
        div.d $f12, $f12, $f4
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 3
        syscall
        sub.d $f12, $f6, $f10         # (−b − √) / 2a
        div.d $f12, $f12, $f4
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 3
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra
rnone:  li    $v0, 4
        la    $a0, nonew
        syscall
        jr    $ra
