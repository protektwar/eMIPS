# MIPS by Example in eMIPS, episode 21, the challenge solved.  best finds
# the student with the highest average of the two exams: two results, the
# index and the average.  count_grades counts the letters into an array of
# five counters — letter - 'A' is the index, F (5) moved down to 4.
        .data
midterm:.word 78, 92, 65, 88, 71, 95
final:  .word 84, 88, 59, 92, 75, 99
letters:.ascii "BADACA"
counts: .word 0, 0, 0, 0, 0           # A B C D F
names:  .ascii "ABCDF"
bestw:  .asciiz "best: student "
withw:  .asciiz ", average "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $a0, midterm            # the best student
        la    $a1, final
        li    $a2, 6
        jal   best
        move  $s0, $v0
        move  $s1, $v1
        li    $v0, 4
        la    $a0, bestw
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 4
        la    $a0, withw
        syscall
        li    $v0, 1
        move  $a0, $s1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        la    $a0, letters            # how many of each letter
        li    $a1, 6
        la    $a2, counts
        jal   count_grades
        li    $s0, 0                  # print "A 3, B 1, ..."
cnext:  la    $t0, names
        addu  $t0, $t0, $s0
        lbu   $a0, 0($t0)
        li    $v0, 11
        syscall
        li    $a0, ' '
        syscall
        sll   $t0, $s0, 2
        la    $t1, counts
        addu  $t0, $t1, $t0
        lw    $a0, 0($t0)
        li    $v0, 1
        syscall
        addiu $s0, $s0, 1
        li    $v0, 11
        li    $a0, '\n'
        beq   $s0, 5, clast
        li    $a0, ','
        syscall
        li    $a0, ' '
clast:  syscall
        blt   $s0, 5, cnext
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# best: $a0 = midterm, $a1 = final, $a2 = n -> $v0 = the index of the best
# average, $v1 = that average (a leaf)
best:   li    $v0, 0
        li    $v1, -1                 # below any average: the first one wins
        li    $t2, 0                  # i
bnext:  lw    $t0, 0($a0)
        lw    $t1, 0($a1)
        addu  $t0, $t0, $t1
        srl   $t0, $t0, 1             # this student's average
        ble   $t0, $v1, bskip         # not better: a tie keeps the first
        move  $v0, $t2
        move  $v1, $t0
bskip:  addiu $a0, $a0, 4
        addiu $a1, $a1, 4
        addiu $t2, $t2, 1
        bne   $t2, $a2, bnext
        jr    $ra

# count_grades: $a0 = the letters (bytes), $a1 = n, $a2 = five counters
# (words): counters[letter - 'A']++, F at 4 (a leaf)
count_grades:
        lbu   $t0, 0($a0)
        addiu $t0, $t0, -'A'          # A 0, B 1, C 2, D 3, F 5
        blt   $t0, 5, cslot
        li    $t0, 4                  # F: the fifth counter
cslot:  sll   $t0, $t0, 2             # a word each
        addu  $t0, $a2, $t0
        lw    $t1, 0($t0)
        addiu $t1, $t1, 1
        sw    $t1, 0($t0)
        addiu $a0, $a0, 1
        addiu $a1, $a1, -1
        bnez  $a1, count_grades
        jr    $ra
