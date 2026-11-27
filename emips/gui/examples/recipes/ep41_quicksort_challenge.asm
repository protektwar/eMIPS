# MIPS by Example in eMIPS, episode 41 — the challenge, solved: how many
# comparisons?  partition counts every "a[j] < pivot".  The last element as
# the pivot is fine on mixed-up numbers, but on an array that is already
# sorted every pivot is the largest: one side is empty, the other n - 1
# long, and the comparisons grow as n * n / 2.  The median of three (the
# middle one of a[lo], a[mid], a[hi], put at hi) splits a sorted array in
# halves.  (The episode: quicksort.)
        .data
mixed:  .word 34, 7, 23, 32, 5, 62, 32, 7, -1, 18
sorted1: .word 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
sorted2: .word 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        .eqv  N, 10
median: .word 0                       # 1: the median of three as the pivot
compares: .word 0                     # comparisons with the pivot
lmixed: .asciiz "mixed,  last element:    "
llast:  .asciiz "sorted, last element:    "
lmed:   .asciiz "sorted, median of three: "
ltail:  .asciiz " comparisons\n"

        .text
main:   addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        la    $a0, mixed              # measure(array, median?, label)
        li    $a1, 0
        la    $a2, lmixed
        jal   measure
        la    $a0, sorted1
        li    $a1, 0
        la    $a2, llast
        jal   measure
        la    $a0, sorted2
        li    $a1, 1
        la    $a2, lmed
        jal   measure
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# measure(a, median, label): sorts a with that pivot rule and prints
# "label n comparisons"
measure:
        addiu $sp, $sp, -24           # 20 $ra, 16 $fp
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        sw    $a2, 32($fp)            # label: in its home slot (main's frame, 8)
        sw    $a1, median
        sw    $zero, compares
        li    $a1, 0                  # quicksort(a, 0, N - 1)
        li    $a2, N - 1
        jal   quicksort
        li    $v0, 4
        lw    $a0, 32($fp)            # the label
        syscall
        li    $v0, 1
        lw    $a0, compares
        syscall
        li    $v0, 4
        la    $a0, ltail
        syscall
        move  $sp, $fp
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        jr    $ra

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
# smaller before it, the rest after; returns p (a leaf: Lomuto's scheme).
# With median = 1 the median of a[lo], a[mid], a[hi] goes to a[hi] first.
partition:
        sll   $t0, $a2, 2
        addu  $t0, $a0, $t0           # &a[hi]
        lw    $t6, median
        beqz  $t6, pstart
        sll   $t2, $a1, 2
        addu  $t2, $a0, $t2           # &a[lo]
        addu  $t3, $a1, $a2
        sra   $t3, $t3, 1
        sll   $t3, $t3, 2
        addu  $t3, $a0, $t3           # &a[mid], mid = (lo + hi) / 2
        lw    $t4, 0($t2)
        lw    $t5, 0($t3)
        bge   $t5, $t4, m1            # a[mid] < a[lo]: swap them
        sw    $t5, 0($t2)
        sw    $t4, 0($t3)
m1:     lw    $t4, 0($t2)
        lw    $t5, 0($t0)
        bge   $t5, $t4, m2            # a[hi] < a[lo]: swap them (a[lo]: the smallest)
        sw    $t5, 0($t2)
        sw    $t4, 0($t0)
m2:     lw    $t4, 0($t3)
        lw    $t5, 0($t0)
        bge   $t4, $t5, pstart        # a[mid] < a[hi]: swap them (a[hi]: the median)
        sw    $t5, 0($t3)
        sw    $t4, 0($t0)
pstart:
        lw    $t1, 0($t0)             # the pivot
        sll   $t2, $a1, 2
        addu  $t2, $a0, $t2           # &a[i], i = lo: where the next small one goes
        move  $t3, $t2                # &a[j], j = lo
        lw    $t6, compares
ploop:  beq   $t3, $t0, pend          # j < hi
        lw    $t4, 0($t3)             # a[j]
        addiu $t6, $t6, 1             # one comparison more
        bge   $t4, $t1, pnext         # a[j] < pivot:
        lw    $t5, 0($t2)             #   swap a[i], a[j]
        sw    $t4, 0($t2)
        sw    $t5, 0($t3)
        addiu $t2, $t2, 4             #   i++
pnext:  addiu $t3, $t3, 4             # j++
        b     ploop
pend:   sw    $t6, compares
        lw    $t5, 0($t2)             # swap a[i], a[hi]: the pivot in place
        sw    $t1, 0($t2)
        sw    $t5, 0($t0)
        subu  $v0, $t2, $a0           # p = i
        sra   $v0, $v0, 2
        jr    $ra
