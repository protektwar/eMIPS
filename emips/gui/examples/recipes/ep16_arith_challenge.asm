# MIPS by Example in eMIPS, episode 16, the challenge solved.  hms prints a
# number of seconds as h:mm:ss — two divisions, each giving a quotient (LO)
# and a remainder (HI).  fahrenheit is F = C * 9 / 5 + 32: multiply first,
# then divide, or the fraction is lost before it is multiplied.
        .data
seconds:.word 3725, 86399
temps:  .word 100, -40, 37
secw:   .asciiz " s = "
cw:     .asciiz " C = "
fw:     .asciiz " F\n"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $s0, seconds            # each number of seconds as h:mm:ss
        li    $s1, 2
snext:  li    $v0, 1
        lw    $a0, 0($s0)
        syscall
        li    $v0, 4
        la    $a0, secw
        syscall
        lw    $a0, 0($s0)
        jal   hms
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, snext
        la    $s0, temps              # each temperature in Fahrenheit
        li    $s1, 3
tnext:  li    $v0, 1
        lw    $a0, 0($s0)
        syscall
        li    $v0, 4
        la    $a0, cw
        syscall
        lw    $a0, 0($s0)
        jal   fahrenheit
        move  $a0, $v0
        li    $v0, 1
        syscall
        li    $v0, 4
        la    $a0, fw
        syscall
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, tnext
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# hms: print $a0 seconds as h:mm:ss and a newline (it calls two2: a frame)
hms:    addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 0-15 home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        li    $t0, 3600
        divu  $a0, $t0                # hours in LO, the seconds left in HI
        mfhi  $s0
        mflo  $a0
        li    $v0, 1                  # the hours
        syscall
        li    $t0, 60
        divu  $s0, $t0                # minutes in LO, seconds in HI
        mfhi  $s0
        mflo  $a0
        jal   two2                    # :mm
        move  $a0, $s0
        jal   two2                    # :ss
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# two2: print ':' and $a0 with two digits, 05 for 5 (a leaf)
two2:   move  $t0, $a0
        li    $v0, 11
        li    $a0, ':'
        syscall
        li    $t1, 10
        divu  $t0, $t1                # tens in LO, units in HI
        mflo  $a0
        addiu $a0, $a0, '0'
        syscall
        mfhi  $a0
        addiu $a0, $a0, '0'
        syscall
        jr    $ra

# fahrenheit: $a0 = degrees C -> $v0 = degrees F, rounded toward 0 (a leaf)
fahrenheit:
        li    $t0, 9
        mul   $v0, $a0, $t0           # C * 9 first ...
        li    $t0, 5
        div   $v0, $t0                # ... then / 5 (signed: -40 too)
        mflo  $v0
        addiu $v0, $v0, 32
        jr    $ra
