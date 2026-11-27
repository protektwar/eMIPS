# MIPS by Example in eMIPS, episode 42 — the challenge, solved: counting
# inversions.  An inversion is a pair in the wrong order (a bigger number
# before a smaller one).  Merge sort finds them all on the way: when merge
# takes the right half's front, it jumps over every element still waiting in
# the left half — each of them bigger, each an inversion.  n log n steps
# instead of checking all n * n / 2 pairs.  (The episode: merge sort.)
        .data
array:  .word 38, 27, 43, 3, 9, 82, 10, 3
reversed: .word 8, 7, 6, 5, 4, 3, 2, 1
        .eqv  N, 8
tmp:    .space 4 * N                  # merge's room
inversions: .word 0                   # merge counts them
before: .asciiz "before: "
after:  .asciiz "after:  "
linv:   .asciiz "inversions: "

        .text
main:   addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        la    $a0, array
        jal   sort_and_count
        la    $a0, reversed           # the most there can be: 8 * 7 / 2
        jal   sort_and_count
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# sort_and_count(a): prints a, sorts it, prints it and its inversions
sort_and_count:
        addiu $sp, $sp, -24           # 20 $ra, 16 $fp
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        sw    $a0, 24($fp)            # a: in its home slot (the caller's frame, 0)
        li    $v0, 4
        la    $a0, before
        syscall
        lw    $a0, 24($fp)
        li    $a1, N
        jal   print_array
        sw    $zero, inversions
        lw    $a0, 24($fp)            # mergesort(a, 0, N)
        li    $a1, 0
        li    $a2, N
        jal   mergesort
        li    $v0, 4
        la    $a0, after
        syscall
        lw    $a0, 24($fp)
        li    $a1, N
        jal   print_array
        li    $v0, 4
        la    $a0, linv
        syscall
        li    $v0, 1
        lw    $a0, inversions
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        jr    $ra

# mergesort(a, lo, hi): sorts a[lo..hi-1] (hi: one past the end)
mergesort:
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
        subu  $t0, $s2, $s1
        blt   $t0, 2, mdone           # 0 or 1 element: sorted already
        addu  $s3, $s1, $s2
        srl   $s3, $s3, 1             # mid = (lo + hi) / 2
        move  $a2, $s3                # mergesort(a, lo, mid)
        jal   mergesort
        move  $a0, $s0                # mergesort(a, mid, hi)
        move  $a1, $s3
        move  $a2, $s2
        jal   mergesort
        move  $a0, $s0                # merge(a, lo, mid, hi)
        move  $a1, $s1
        move  $a2, $s3
        move  $a3, $s2
        jal   merge
mdone:  move  $sp, $fp
        lw    $s0, 16($sp)
        lw    $s1, 20($sp)
        lw    $s2, 24($sp)
        lw    $s3, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra

# merge(a, lo, mid, hi): a[lo..mid-1] and a[mid..hi-1] are sorted; merges
# them into tmp, then copies tmp back (a leaf)
merge:
        sll   $t0, $a1, 2
        addu  $t0, $a0, $t0           # i: &a[lo], the left half's front
        sll   $t1, $a2, 2
        addu  $t1, $a0, $t1           # j: &a[mid], the right half's front
        move  $t7, $t1                # the left half's end
        sll   $t2, $a3, 2
        addu  $t2, $a0, $t2           # the right half's end
        la    $t3, tmp                # k: where the next one goes in tmp
both:   beq   $t0, $t7, rest          # one half empty: copy the other
        beq   $t1, $t2, rest
        lw    $t4, 0($t0)
        lw    $t5, 0($t1)
        bgt   $t4, $t5, right         # a[i] <= a[j]: the left one (equal: left
        sw    $t4, 0($t3)             # first — merge sort is stable)
        addiu $t0, $t0, 4
        b     moved
right:  sw    $t5, 0($t3)             # else the right one: it jumps over
        addiu $t1, $t1, 4             # all that waits in the left half
        subu  $t6, $t7, $t0
        srl   $t6, $t6, 2             # (left end - i) / 4 of them
        lw    $t4, inversions
        addu  $t4, $t4, $t6           # each one an inversion
        sw    $t4, inversions
moved:  addiu $t3, $t3, 4
        b     both
rest:   beq   $t0, $t7, rright        # what is left of the left half
        lw    $t4, 0($t0)
        sw    $t4, 0($t3)
        addiu $t0, $t0, 4
        addiu $t3, $t3, 4
        b     rest
rright: beq   $t1, $t2, back          # what is left of the right half
        lw    $t4, 0($t1)
        sw    $t4, 0($t3)
        addiu $t1, $t1, 4
        addiu $t3, $t3, 4
        b     rright
back:   sll   $t0, $a1, 2             # tmp back into a[lo..hi-1]
        addu  $t0, $a0, $t0
        la    $t1, tmp
copy:   beq   $t0, $t2, mend
        lw    $t4, 0($t1)
        sw    $t4, 0($t0)
        addiu $t0, $t0, 4
        addiu $t1, $t1, 4
        b     copy
mend:   jr    $ra

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
