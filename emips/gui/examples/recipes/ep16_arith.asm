# MIPS by Example in eMIPS, episode 16 — Arithmetic: the four operations,
# shifts, area and perimeter.  addu, subu, mul; div leaves the quotient in
# LO and the remainder in HI; sll multiplies by a power of 2, and sra divides
# a signed number by one (rounding down, where div rounds toward 0).
        .data
ask:    .asciiz "Two numbers:\n"
plus:   .asciiz " + "
minus:  .asciiz " - "
times:  .asciiz " * "
over:   .asciiz " / "
mod:    .asciiz " % "
shl:    .asciiz " << "
sar:    .asciiz " >> "
equals: .asciiz " = "
area:   .asciiz "area "
perim:  .asciiz ", perimeter "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        li    $v0, 4
        la    $a0, ask
        syscall
        li    $v0, 5                  # a
        syscall
        move  $s0, $v0
        li    $v0, 5                  # b
        syscall
        move  $s1, $v0
        move  $a0, $s0                # a + b
        la    $a1, plus
        move  $a2, $s1
        addu  $a3, $s0, $s1
        jal   print_eq
        move  $a0, $s0                # a - b
        la    $a1, minus
        move  $a2, $s1
        subu  $a3, $s0, $s1
        jal   print_eq
        move  $a0, $s0                # a * b
        la    $a1, times
        move  $a2, $s1
        mul   $a3, $s0, $s1
        jal   print_eq
        div   $s0, $s1                # a / b and a % b: one div
        move  $a0, $s0
        la    $a1, over
        move  $a2, $s1
        mflo  $a3                     # the quotient
        jal   print_eq
        div   $s0, $s1                # (print_eq may change HI: again)
        move  $a0, $s0
        la    $a1, mod
        move  $a2, $s1
        mfhi  $a3                     # the remainder
        jal   print_eq
        move  $a0, $s0                # a << 2: times 4
        la    $a1, shl
        li    $a2, 2
        sll   $a3, $s0, 2
        jal   print_eq
        subu  $a0, $zero, $s0         # -a >> 1: half, rounded down
        la    $a1, sar
        li    $a2, 1
        sra   $a3, $a0, 1
        jal   print_eq
        li    $v0, 4                  # a rectangle a by b
        la    $a0, area
        syscall
        li    $v0, 1
        mul   $a0, $s0, $s1           # area: a * b
        syscall
        li    $v0, 4
        la    $a0, perim
        syscall
        li    $v0, 1
        addu  $a0, $s0, $s1           # perimeter: 2 * (a + b)
        sll   $a0, $a0, 1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# print_eq: print "$a0 op $a2 = $a3", op the text at $a1 (a leaf: four
# arguments, system calls only)
print_eq:
        move  $t0, $a0
        li    $v0, 1                  # a
        syscall
        li    $v0, 4                  # op
        move  $a0, $a1
        syscall
        li    $v0, 1                  # b
        move  $a0, $a2
        syscall
        li    $v0, 4                  # =
        la    $a0, equals
        syscall
        li    $v0, 1                  # the result
        move  $a0, $a3
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra
