# MIPS by Example in eMIPS, episode 21 — Arrays: sum, search, reverse, two
# score arrays into grades.  An array of words: element i at base + 4 i.
# sum and find walk it, reverse swaps from both ends, and grades turns two
# arrays — midterm and final — into a third: one letter (a byte) a student.
        .data
midterm:.word 78, 92, 65, 88, 71, 95
final:  .word 84, 88, 59, 92, 75, 99
letters:.space 6                      # one grade letter a student
midw:   .asciiz "midterm:"
sumw:   .asciiz "sum "
avgw:   .asciiz ", average "
atw:    .asciiz " is at index "
gradew: .asciiz "grades:"
revw:   .asciiz "reversed:"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20 $s0, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        la    $a0, midw               # the midterm scores
        la    $a1, midterm
        li    $a2, 6
        jal   print_array
        la    $a0, midterm            # their sum and average
        li    $a1, 6
        jal   sum
        move  $s0, $v0
        li    $v0, 4
        la    $a0, sumw
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 4
        la    $a0, avgw
        syscall
        li    $t0, 6
        divu  $s0, $t0
        mflo  $a0
        li    $v0, 1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $a0, 88                 # where is 88? and 100?
        jal   show_find
        li    $a0, 100
        jal   show_find
        la    $a0, midterm            # the grades: midterm and final together
        la    $a1, final
        la    $a2, letters
        li    $a3, 6
        jal   grades
        li    $v0, 4
        la    $a0, gradew
        syscall
        la    $s0, letters
gnext:  li    $v0, 11
        li    $a0, ' '
        syscall
        lbu   $a0, 0($s0)
        syscall
        addiu $s0, $s0, 1
        la    $t0, letters + 6
        bne   $s0, $t0, gnext
        li    $a0, '\n'
        syscall
        la    $a0, midterm            # reversed in place
        li    $a1, 6
        jal   reverse
        la    $a0, revw
        la    $a1, midterm
        li    $a2, 6
        jal   print_array
        move  $sp, $fp                # main's epilogue
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# sum: $a0 = the words, $a1 = n -> $v0 = their sum (a leaf)
sum:    li    $v0, 0
snext:  lw    $t0, 0($a0)
        addu  $v0, $v0, $t0
        addiu $a0, $a0, 4             # the next element: 4 bytes on
        addiu $a1, $a1, -1
        bnez  $a1, snext
        jr    $ra

# find: $a0 = the words, $a1 = n, $a2 = x -> $v0 = x's index, or -1
# (a leaf)
find:   li    $v0, 0                  # the index
fnext:  beq   $v0, $a1, fnone         # past the end: not there
        sll   $t0, $v0, 2             # the index × 4 ...
        addu  $t0, $a0, $t0           # ... + the base: &a[i]
        lw    $t0, 0($t0)
        beq   $t0, $a2, fdone         # found
        addiu $v0, $v0, 1
        b     fnext
fnone:  li    $v0, -1
fdone:  jr    $ra

# show_find: print "x is at index i" for x = $a0 in the midterm scores
# (it calls find: a frame; x waits in the caller's home area)
show_find:
        addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        sw    $a0, 24($fp)            # x, in the caller's home area
        move  $a2, $a0
        la    $a0, midterm
        li    $a1, 6
        jal   find
        move  $t0, $v0
        li    $v0, 1
        lw    $a0, 24($fp)
        syscall
        li    $v0, 4
        la    $a0, atw
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        jr    $ra

# grades: $a0 = midterm, $a1 = final (words), $a2 = the letters (bytes),
# $a3 = n: each student's average as a letter (it calls letter: a frame,
# the four arguments in $s0-$s3)
grades: addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28-16 $s0-$s3
        sw    $ra, 36($sp)
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        sw    $s3, 16($sp)
        move  $fp, $sp
        move  $s0, $a0
        move  $s1, $a1
        move  $s2, $a2
        move  $s3, $a3
grnext: lw    $t0, 0($s0)             # midterm[i] + final[i] ...
        lw    $t1, 0($s1)
        addu  $a0, $t0, $t1
        srl   $a0, $a0, 1             # ... / 2
        jal   letter
        sb    $v0, 0($s2)             # letters[i]: a byte
        addiu $s0, $s0, 4             # words: 4 bytes on
        addiu $s1, $s1, 4
        addiu $s2, $s2, 1             # bytes: 1 on
        addiu $s3, $s3, -1
        bnez  $s3, grnext
        move  $sp, $fp
        lw    $s3, 16($sp)
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra

# letter: $a0 = an average -> $v0 = 'A' (90+), 'B' (80+), 'C' (70+), 'D'
# (60+) or 'F' (a leaf)
letter: li    $v0, 'A'
        bge   $a0, 90, ldone
        li    $v0, 'B'
        bge   $a0, 80, ldone
        li    $v0, 'C'
        bge   $a0, 70, ldone
        li    $v0, 'D'
        bge   $a0, 60, ldone
        li    $v0, 'F'
ldone:  jr    $ra

# reverse: the $a1 words at $a0, in place: swap the ends, step inwards,
# n / 2 times (a leaf)
reverse:
        sll   $t0, $a1, 2
        addu  $t1, $a0, $t0
        addiu $t1, $t1, -4            # the last element
        srl   $t2, $a1, 1             # n / 2 swaps
rnext:  beqz  $t2, rdone
        lw    $t3, 0($a0)
        lw    $t4, 0($t1)
        sw    $t4, 0($a0)
        sw    $t3, 0($t1)
        addiu $a0, $a0, 4             # step inwards from both ends
        addiu $t1, $t1, -4
        addiu $t2, $t2, -1
        b     rnext
rdone:  jr    $ra

# print_array: the text at $a0, then the $a2 words at $a1, each after a
# space (a leaf: system calls only)
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
        li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra
