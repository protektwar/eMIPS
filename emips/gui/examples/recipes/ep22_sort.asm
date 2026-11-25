# MIPS by Example in eMIPS, episode 22 — Sorting numbers: bubble,
# insertion, into a second array.  The numbers stay as they are: each sort
# works on a copy.  bubble swaps neighbours in the wrong order, pass after
# pass, until a pass swaps nothing; insertion moves each number down into
# place.  Both do one step per pair in the wrong order — 11 here.
        .data
nums:   .word 64, 25, 12, 22, 11, 90, 35
sorted1:.space 28                     # 7 words: bubble's copy
sorted2:.space 28                     # insertion's copy
numw:   .asciiz "numbers:"
bubw:   .asciiz "bubble:"
insw:   .asciiz "insertion:"
swapw:  .asciiz " swaps in "
passw:  .asciiz " passes\n"
movew:  .asciiz " moves\n"
comma:  .asciiz ", "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $a0, numw               # the numbers
        la    $a1, nums
        jal   print_array
        li    $v0, 11
        li    $a0, '\n'
        syscall
        la    $a0, nums               # bubble sort a copy
        la    $a1, sorted1
        li    $a2, 7
        jal   copy
        la    $a0, sorted1
        li    $a1, 7
        jal   bubble
        move  $s0, $v0                # swaps
        move  $s1, $v1                # passes
        la    $a0, bubw
        la    $a1, sorted1
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
        li    $v0, 1
        move  $a0, $s1
        syscall
        li    $v0, 4
        la    $a0, passw
        syscall
        la    $a0, nums               # insertion sort another copy
        la    $a1, sorted2
        li    $a2, 7
        jal   copy
        la    $a0, sorted2
        li    $a1, 7
        jal   insertion
        move  $s0, $v0                # moves
        la    $a0, insw
        la    $a1, sorted2
        jal   print_array
        li    $v0, 4
        la    $a0, comma
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 4
        la    $a0, movew
        syscall
        la    $a0, numw               # and the numbers: unchanged
        la    $a1, nums
        jal   print_array
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

# copy: the $a2 words at $a0 to $a1 (a leaf)
copy:   lw    $t0, 0($a0)
        sw    $t0, 0($a1)
        addiu $a0, $a0, 4
        addiu $a1, $a1, 4
        addiu $a2, $a2, -1
        bnez  $a2, copy
        jr    $ra

# bubble: sort the $a1 words at $a0 -> $v0 = swaps, $v1 = passes; each
# pass compares a[i] with a[i + 1] up to the last unsorted pair; a pass
# without a swap ends it (a leaf)
bubble: li    $v0, 0
        li    $v1, 0
        addiu $t5, $a1, -1            # pairs in the first pass: n - 1
bpass:  addiu $v1, $v1, 1
        li    $t4, 0                  # swapped in this pass?
        move  $t0, $a0                # &a[i]
        move  $t1, $t5
bpair:  lw    $t2, 0($t0)             # a[i] ...
        lw    $t3, 4($t0)             # ... and a[i + 1]
        ble   $t2, $t3, bnext         # in order
        sw    $t3, 0($t0)             # swap them
        sw    $t2, 4($t0)
        addiu $v0, $v0, 1
        li    $t4, 1
bnext:  addiu $t0, $t0, 4
        addiu $t1, $t1, -1
        bnez  $t1, bpair
        beqz  $t4, bdone              # no swap: sorted
        addiu $t5, $t5, -1            # the largest is in place now
        bnez  $t5, bpass
bdone:  jr    $ra

# insertion: sort the $a1 words at $a0 -> $v0 = moves: each a[i] goes
# down past the larger ones, which move up one place (a leaf)
insertion:
        li    $v0, 0
        li    $t0, 1                  # i
inext:  beq   $t0, $a1, idone
        sll   $t1, $t0, 2
        addu  $t1, $a0, $t1           # &a[i]
        lw    $t2, 0($t1)             # the key
imove:  beq   $t1, $a0, iplace        # at a[0]: nothing below
        lw    $t3, -4($t1)            # the one below
        ble   $t3, $t2, iplace        # not larger: the key goes here
        sw    $t3, 0($t1)             # it moves up one place
        addiu $v0, $v0, 1
        addiu $t1, $t1, -4
        b     imove
iplace: sw    $t2, 0($t1)
        addiu $t0, $t0, 1
        b     inext
idone:  jr    $ra

# print_array: the text at $a0, then the 7 words at $a1 (a leaf)
print_array:
        li    $v0, 4
        syscall
        li    $t0, 7
panext: li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        lw    $a0, 0($a1)
        syscall
        addiu $a1, $a1, 4
        addiu $t0, $t0, -1
        bnez  $t0, panext
        jr    $ra
