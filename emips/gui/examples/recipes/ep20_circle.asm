# MIPS by Example in eMIPS, episode 20 — Floating point: a circle's area
# with doubles.  The FPU has its own 32 registers; a double takes an even /
# odd pair ($f12 = $f12 and $f13).  o32: a double argument in $f12, the
# result in $f0; $f20-$f30 are saved like $s registers.  read_double (7)
# leaves the number in $f0, print_double (3) prints $f12.
        .data
pi:     .double 3.141592653589793
ask:    .asciiz "Radius:\n"
areaw:  .asciiz "area "
roundw: .asciiz ", rounded "
circw:  .asciiz "\ncircumference "
        .text
main:   addiu $sp, $sp, -40           # main's frame: 36 $ra, 32 $fp, 24-31 $f20,
        sw    $ra, 36($sp)            # 0-15 the home area
        sw    $fp, 32($sp)
        swc1  $f20, 24($sp)           # $f20 is saved, like an $s register: its two
        swc1  $f21, 28($sp)           # halves (sdc1 needs 8-alignment; $sp is 4-)
        move  $fp, $sp
        li    $v0, 4
        la    $a0, ask
        syscall
        li    $v0, 7                  # read_double -> $f0
        syscall
        mov.d $f20, $f0               # r, kept across the calls
        mov.d $f12, $f20              # area(r)
        jal   area
        li    $v0, 4
        la    $a0, areaw
        syscall
        mov.d $f12, $f0               # print_double prints $f12
        li    $v0, 3
        syscall
        li    $v0, 4
        la    $a0, roundw
        syscall
        round.w.d $f4, $f12           # to the nearest whole number ...
        mfc1  $a0, $f4                # ... into an integer register
        li    $v0, 1
        syscall
        mov.d $f12, $f20              # circumference(r)
        jal   circumference
        li    $v0, 4
        la    $a0, circw
        syscall
        mov.d $f12, $f0
        li    $v0, 3
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lwc1  $f21, 28($sp)
        lwc1  $f20, 24($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        li    $v0, 10
        syscall

# area: $f12 = r -> $f0 = r × r × pi (a leaf)
area:   mul.d $f0, $f12, $f12         # r × r
        ldc1  $f2, pi                 # (lui $at + ldc1: a label load)
        mul.d $f0, $f0, $f2           # × pi
        jr    $ra

# circumference: $f12 = r -> $f0 = 2r × pi (a leaf)
circumference:
        add.d $f0, $f12, $f12         # 2r: r + r, exact
        ldc1  $f2, pi                 # (lui $at + ldc1: a label load)
        mul.d $f0, $f0, $f2
        jr    $ra
