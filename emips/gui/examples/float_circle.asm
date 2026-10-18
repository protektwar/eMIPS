# Floating point (coprocessor 1): area of a circle, pi * r * r.
# Single-precision registers $f0..$f31; print_float prints $f12.
        .data
pi:     .float 3.14159265
r:      .float 2.5
msg:    .asciiz "area = "
        .text
main:   la    $t0, pi
        lwc1  $f0, 0($t0)       # pi
        lwc1  $f1, 4($t0)       # r
        mul.s $f2, $f1, $f1     # r * r
        mul.s $f12, $f2, $f0    # * pi
        li    $v0, 4
        la    $a0, msg
        syscall
        li    $v0, 2            # print_float($f12)
        syscall                 # prints 19.634954
        li    $v0, 10
        syscall
