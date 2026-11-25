# MIPS by Example in eMIPS, episode 22, the challenge solved.  selection
# sort finds the smallest of the rest and swaps it into place: at most one
# swap a position, 5 here against bubble's 11.  merge joins two sorted
# arrays into a third: always take the smaller front element.
        .data
nums:   .word 64, 25, 12, 22, 11, 90, 35
left:   .word 3, 8, 15, 21
right:  .word 1, 9, 10, 30, 42
merged: .space 36                     # 4 + 5 words
selw:   .asciiz "selection:"
swapw:  .asciiz " swaps\n"
mergew: .asciiz "merged:"
comma:  .asciiz ", "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20 $s0, 16 a fifth argument, 0-15 home
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        la    $a0, nums               # selection sort, in place
        li    $a1, 7
        jal   selection
        move  $s0, $v0
        la    $a0, selw
        la    $a1, nums
        li    $a2, 7
        jal   print_array
        li    $v0, 4
        la    $a0, comma
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 4
        la    $a0, swapw
        syscall
        la    $a0, left               # merge(left, 4, right, 5) into merged
        li    $a1, 4
        la    $a2, right
        li    $a3, 5
        la    $t0, merged             # the fifth argument: at 16 in our frame
        sw    $t0, 16($sp)
        jal   merge
        la    $a0, mergew
        la    $a1, merged
        li    $a2, 9
        jal   print_array
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

# selection: sort the $a1 words at $a0 -> $v0 = swaps: for each place, the
# smallest of the rest swapped into it (a leaf)
selection:
        li    $v0, 0
snext:  addiu $a1, $a1, -1            # the rest after this place
        beqz  $a1, sdone
        move  $t0, $a0                # the smallest so far: a[i]
        addiu $t1, $a0, 4             # look from a[i + 1]
        move  $t2, $a1
smin:   lw    $t3, 0($t1)
        lw    $t4, 0($t0)
        bge   $t3, $t4, snot
        move  $t0, $t1                # a new smallest
snot:   addiu $t1, $t1, 4
        addiu $t2, $t2, -1
        bnez  $t2, smin
        beq   $t0, $a0, sstay         # already in place: no swap
        lw    $t3, 0($a0)
        lw    $t4, 0($t0)
        sw    $t4, 0($a0)
        sw    $t3, 0($t0)
        addiu $v0, $v0, 1
sstay:  addiu $a0, $a0, 4
        b     snext
sdone:  jr    $ra

# merge: $a0 / $a1 = the left words and their count, $a2 / $a3 = the
# right ones, the fifth argument (16 in the caller's frame) = where to put
# them all, in order (a leaf: it reads 16($sp) as it is)
merge:  lw    $t9, 16($sp)            # the output
mnext:  beqz  $a1, mright             # left empty: the rest of right
        beqz  $a3, mleft              # right empty: the rest of left
        lw    $t0, 0($a0)
        lw    $t1, 0($a2)
        bgt   $t0, $t1, mtake_r
mtake_l:sw    $t0, 0($t9)             # the left one is smaller (or equal)
        addiu $a0, $a0, 4
        addiu $a1, $a1, -1
        addiu $t9, $t9, 4
        b     mnext
mtake_r:sw    $t1, 0($t9)
        addiu $a2, $a2, 4
        addiu $a3, $a3, -1
        addiu $t9, $t9, 4
        b     mnext
mleft:  lw    $t0, 0($a0)             # what is left of one side
        b     mtake_l
mright: beqz  $a3, mdone
        lw    $t1, 0($a2)
        b     mtake_r
mdone:  jr    $ra

# print_array: the text at $a0, then the $a2 words at $a1 (a leaf)
print_array:
        li    $v0, 4
        syscall
panext: li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        lw    $a0, 0($a1)
        syscall
        addiu $a1, $a1, 4
        addiu $a2, $a2, -1
        bnez  $a2, panext
        jr    $ra
