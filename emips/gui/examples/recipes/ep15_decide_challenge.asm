# MIPS by Example in eMIPS, episode 15, the challenge solved.  sort3 puts
# three numbers in order with three compare-and-swap steps (a, b; b, c; a,
# b again).  leap decides a leap year: divisible by 4 and not by 100, or
# divisible by 400 — each "divisible" a div and its remainder in HI.
        .data
nums:   .word 42, -5, 17
years:  .word 1900, 2000, 2024, 2026
sorted: .asciiz "sorted:"
leapw:  .asciiz " is a leap year\n"
notw:   .asciiz " is not a leap year\n"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $a0, nums               # sort3(nums), printed
        jal   sort3
        li    $v0, 4
        la    $a0, sorted
        syscall
        la    $s0, nums
        li    $s1, 3
pnext:  li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        lw    $a0, 0($s0)
        syscall
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, pnext
        li    $v0, 11
        li    $a0, '\n'
        syscall
        la    $s0, years              # each year: leap or not
        li    $s1, 4
ynext:  lw    $a0, 0($s0)
        jal   leap
        move  $t0, $v0
        li    $v0, 1
        lw    $a0, 0($s0)
        syscall
        li    $v0, 4
        la    $a0, leapw
        bnez  $t0, ysay
        la    $a0, notw
ysay:   syscall
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, ynext
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# sort3: the three words at $a0 in ascending order (a leaf)
sort3:  lw    $t0, 0($a0)
        lw    $t1, 4($a0)
        lw    $t2, 8($a0)
        slt   $t3, $t1, $t0           # a > b? swap them
        beqz  $t3, s1
        move  $t3, $t0
        move  $t0, $t1
        move  $t1, $t3
s1:     slt   $t3, $t2, $t1           # b > c? swap them: c is the largest now
        beqz  $t3, s2
        move  $t3, $t1
        move  $t1, $t2
        move  $t2, $t3
s2:     slt   $t3, $t1, $t0           # a > b again? the smaller two
        beqz  $t3, s3
        move  $t3, $t0
        move  $t0, $t1
        move  $t1, $t3
s3:     sw    $t0, 0($a0)
        sw    $t1, 4($a0)
        sw    $t2, 8($a0)
        jr    $ra

# leap: $a0 = a year -> $v0 = 1 if it is a leap year (a leaf)
leap:   li    $t0, 400
        div   $a0, $t0
        mfhi  $t1
        beqz  $t1, yes                # divisible by 400: leap
        li    $t0, 100
        div   $a0, $t0
        mfhi  $t1
        beqz  $t1, no                 # by 100 (not 400): not leap
        andi  $t1, $a0, 3             # by 4: the low two bits are 0
        beqz  $t1, yes
no:     li    $v0, 0
        jr    $ra
yes:    li    $v0, 1
        jr    $ra
