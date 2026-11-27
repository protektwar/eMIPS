# MIPS by Example in eMIPS, episode 41 — Quicksort.  Pick a pivot (here: the
# last element), move everything smaller to its left (partition), and sort
# the two sides the same way — recursion.  Each call keeps the array, its
# range and the pivot's place in $s registers, saved in its frame: the two
# recursive calls would overwrite anything else.
        .data
array:  .word 34, 7, 23, 32, 5, 62, 32, 7, -1, 18
        .eqv  N, 10
before: .asciiz "before: "
after:  .asciiz "after:  "

        .text
main:   addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        li    $v0, 4
        la    $a0, before
        syscall
        la    $a0, array
        li    $a1, N
        jal   print_array
        la    $a0, array              # quicksort(array, 0, N - 1)
        li    $a1, 0
        li    $a2, N - 1
        jal   quicksort
        li    $v0, 4
        la    $a0, after
        syscall
        la    $a0, array
        li    $a1, N
        jal   print_array
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# quicksort(a, lo, hi): sorts a[lo..hi]
quicksort:
        addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28 $s3, 24 $s2, 20 $s1, 16 $s0
        sw    $ra, 36($sp)
        sw    $fp, 32($sp)
        sw    $s3, 28($sp)
        sw    $s2, 24($sp)
        sw    $s1, 20($sp)
        sw    $s0, 16($sp)
        move  $fp, $sp
        move  $s0, $a0                # a
        move  $s1, $a1                # lo
        move  $s2, $a2                # hi
        bge   $s1, $s2, qdone         # 0 or 1 element: sorted already
        jal   partition               # p = partition(a, lo, hi)
        move  $s3, $v0
        move  $a0, $s0                # quicksort(a, lo, p - 1)
        move  $a1, $s1
        addiu $a2, $s3, -1
        jal   quicksort
        move  $a0, $s0                # quicksort(a, p + 1, hi)
        addiu $a1, $s3, 1
        move  $a2, $s2
        jal   quicksort
qdone:  move  $sp, $fp
        lw    $s0, 16($sp)
        lw    $s1, 20($sp)
        lw    $s2, 24($sp)
        lw    $s3, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra

# partition(a, lo, hi): the pivot a[hi] to its sorted place p; everything
# smaller before it, the rest after; returns p (a leaf: Lomuto's scheme)
partition:
        sll   $t0, $a2, 2
        addu  $t0, $a0, $t0           # &a[hi]
        lw    $t1, 0($t0)             # the pivot
        sll   $t2, $a1, 2
        addu  $t2, $a0, $t2           # &a[i], i = lo: where the next small one goes
        move  $t3, $t2                # &a[j], j = lo
ploop:  beq   $t3, $t0, pend          # j < hi
        lw    $t4, 0($t3)             # a[j]
        bge   $t4, $t1, pnext         # a[j] < pivot:
        lw    $t5, 0($t2)             #   swap a[i], a[j]
        sw    $t4, 0($t2)
        sw    $t5, 0($t3)
        addiu $t2, $t2, 4             #   i++
pnext:  addiu $t3, $t3, 4             # j++
        b     ploop
pend:   lw    $t5, 0($t2)             # swap a[i], a[hi]: the pivot in place
        sw    $t1, 0($t2)
        sw    $t5, 0($t0)
        subu  $v0, $t2, $a0           # p = i
        sra   $v0, $v0, 2
        jr    $ra

# print_array(a, n): "a0 a1 ..." and a new line (a leaf)
print_array:
        move  $t0, $a0
        sll   $t1, $a1, 2
        addu  $t1, $t0, $t1           # the end
pa:     lw    $a0, 0($t0)
        li    $v0, 1
        syscall
        addiu $t0, $t0, 4
        li    $v0, 11
        li    $a0, ' '                # a space between two, a new line after the last
        bne   $t0, $t1, pspace
        li    $a0, '\n'
pspace: syscall
        bne   $t0, $t1, pa
        jr    $ra
