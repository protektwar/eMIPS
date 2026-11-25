# MIPS by Example in eMIPS, episode 29, the challenge solved.  count runs
# Hanoi's recursion without the pegs, only counting how often each disk
# moves: disk 1 every other move, the largest once.  total is the number of
# moves itself, T(n) = 2 T(n - 1) + 1 — which is 2^n - 1, a shift.
        .data
counts: .word 0, 0, 0, 0              # moves of disks 1-4
perw:   .asciiz "4 disks, moves per disk:"
tw:     .asciiz "T(n):"
sw:     .asciiz "2^n - 1:"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20 $s0, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        li    $a0, 4                  # count(4)
        jal   count
        li    $v0, 4
        la    $a0, perw
        syscall
        li    $s0, 0                  # disks 1 to 4
pnext:  li    $v0, 11
        li    $a0, ' '
        syscall
        sll   $t0, $s0, 2
        li    $v0, 1
        lw    $a0, counts($t0)
        syscall
        addiu $s0, $s0, 1
        blt   $s0, 4, pnext
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $v0, 4                  # T(n) for n = 1 to 8
        la    $a0, tw
        syscall
        li    $s0, 1
tnext:  move  $a0, $s0
        jal   total
        move  $t0, $v0
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        addiu $s0, $s0, 1
        ble   $s0, 8, tnext
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $v0, 4                  # and 2^n - 1
        la    $a0, sw
        syscall
        li    $s0, 1
snext:  li    $v0, 11
        li    $a0, ' '
        syscall
        li    $t0, 1
        sllv  $t0, $t0, $s0           # 2^n
        addiu $a0, $t0, -1
        li    $v0, 1
        syscall
        addiu $s0, $s0, 1
        ble   $s0, 8, snext
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# count: $a0 = n: the moves of n disks, counted per disk in counts (it
# calls itself: a frame, n in $s0)
count:  addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        beqz  $a0, cdone
        move  $s0, $a0
        addiu $a0, $s0, -1            # n - 1 disks out of the way
        jal   count
        sll   $t0, $s0, 2             # disk n moves: counts[n - 1]++
        lw    $t1, counts - 4($t0)
        addiu $t1, $t1, 1
        sw    $t1, counts - 4($t0)
        addiu $a0, $s0, -1            # n - 1 disks back on top
        jal   count
cdone:  move  $sp, $fp
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# total: $a0 = n -> $v0 = T(n) = 2 T(n - 1) + 1, T(0) = 0 (it calls itself:
# a frame; nothing needed after the call but the result)
total:  addiu $sp, $sp, -24           # 20 $ra, 16 $fp
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        li    $v0, 0
        beqz  $a0, tdone
        addiu $a0, $a0, -1
        jal   total
        sll   $v0, $v0, 1             # 2 T(n - 1) + 1
        addiu $v0, $v0, 1
tdone:  move  $sp, $fp
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        jr    $ra
