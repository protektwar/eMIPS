# MIPS by Example in eMIPS, episode 29 — Towers of Hanoi.  To move n disks
# from one peg to another: move n - 1 out of the way onto the third peg,
# move disk n, move the n - 1 back on top of it.  The pegs live in memory —
# 4 bytes each, a disk number a byte, bottom first — so every move can be
# watched, and checked: never a larger disk on a smaller one.
        .data
pegs:   .byte 3, 2, 1, 0              # peg A: disks 3, 2, 1
        .byte 0, 0, 0, 0              # peg B
        .byte 0, 0, 0, 0              # peg C
height: .word 3, 0, 0                 # how many disks on each peg
moves:  .word 0
diskw:  .asciiz "disk "
colon:  .asciiz ": "
arrow:  .asciiz " -> "
movesw: .asciiz " moves\n"
badw:   .asciiz "illegal: a smaller disk under disk "
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        li    $a0, 3                  # hanoi(3, A, C, B)
        li    $a1, 0
        li    $a2, 2
        li    $a3, 1
        jal   hanoi
        li    $v0, 1
        lw    $a0, moves
        syscall
        li    $v0, 4
        la    $a0, movesw
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# hanoi: move $a0 disks from peg $a1 to peg $a2, using peg $a3 (pegs 0-2;
# it calls itself: a frame, the four arguments in $s0-$s3)
hanoi:  addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28-16 $s0-$s3
        sw    $ra, 36($sp)
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        sw    $s3, 16($sp)
        move  $fp, $sp
        beqz  $a0, hdone              # no disks: nothing to do
        move  $s0, $a0                # n
        move  $s1, $a1                # from
        move  $s2, $a2                # to
        move  $s3, $a3                # via
        addiu $a0, $s0, -1            # n - 1 disks: from -> via, out of the way
        move  $a1, $s1
        move  $a2, $s3
        move  $a3, $s2
        jal   hanoi
        move  $a0, $s0                # disk n: from -> to
        move  $a1, $s1
        move  $a2, $s2
        jal   move_disk
        addiu $a0, $s0, -1            # n - 1 disks: via -> to, on top of it
        move  $a1, $s3
        move  $a2, $s2
        move  $a3, $s1
        jal   hanoi
hdone:  move  $sp, $fp
        lw    $s3, 16($sp)
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra

# move_disk: take the top disk of peg $a1, put it on peg $a2 (it should be
# disk $a0) and print "disk n: X -> Y" (a leaf)
move_disk:
        sll   $t0, $a1, 2             # height[from] - 1: the top disk's place
        lw    $t1, height($t0)
        addiu $t1, $t1, -1
        sw    $t1, height($t0)
        addu  $t2, $t0, $t1
        lbu   $t3, pegs($t2)          # the disk
        sb    $zero, pegs($t2)        # gone from there
        sll   $t4, $a2, 2             # on top of peg to
        lw    $t5, height($t4)
        beqz  $t5, mfree              # an empty peg takes any disk
        addu  $t6, $t4, $t5
        lbu   $t6, pegs - 1($t6)      # the disk it would sit on
        bltu  $t6, $t3, mbad          # smaller: illegal
mfree:  addu  $t6, $t4, $t5
        sb    $t3, pegs($t6)
        addiu $t5, $t5, 1
        sw    $t5, height($t4)
        lw    $t6, moves              # one more move
        addiu $t6, $t6, 1
        sw    $t6, moves
        li    $v0, 4                  # "disk n: X -> Y"
        la    $a0, diskw
        syscall
        li    $v0, 1
        move  $a0, $t3
        syscall
        li    $v0, 4
        la    $a0, colon
        syscall
        li    $v0, 11
        addiu $a0, $a1, 'A'
        syscall
        li    $v0, 4
        la    $a0, arrow
        syscall
        li    $v0, 11
        addiu $a0, $a2, 'A'
        syscall
        li    $a0, '\n'
        syscall
        jr    $ra
mbad:   sb    $t3, pegs($t2)          # put it back where it was
        addiu $t1, $t1, 1
        sw    $t1, height($t0)
        li    $v0, 4                  # "illegal: disk n"
        la    $a0, badw
        syscall
        li    $v0, 1
        move  $a0, $t3
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra
