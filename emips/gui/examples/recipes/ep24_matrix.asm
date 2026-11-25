# MIPS by Example in eMIPS, episode 24 — Matrices: add and multiply.  A
# matrix is stored row after row: element (i, j) of an n × n matrix of words
# is at base + (i × n + j) × 4.  Adding is one flat loop over n × n words;
# multiplying walks a row of A (4 bytes a step) against a column of B (4 n
# bytes a step).
        .data
ma:     .word 1, 2, 3
        .word 4, 5, 6
        .word 7, 8, 9
mb:     .word 9, 8, 7
        .word 6, 5, 4
        .word 3, 2, 1
mc:     .space 36                     # 3 × 3 words: the result
sumw:   .asciiz "A + B:\n"
prodw:  .asciiz "A x B:\n"
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        la    $a0, ma                 # C = A + B
        la    $a1, mb
        la    $a2, mc
        li    $a3, 3
        jal   madd
        la    $a0, sumw
        la    $a1, mc
        li    $a2, 3
        jal   print_matrix
        la    $a0, ma                 # C = A × B
        la    $a1, mb
        la    $a2, mc
        li    $a3, 3
        jal   mmul
        la    $a0, prodw
        la    $a1, mc
        li    $a2, 3
        jal   print_matrix
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# madd: $a2 = $a0 + $a1, element by element, $a3 × $a3 words: the rows do
# not matter, one flat loop (a leaf)
madd:   mul   $t0, $a3, $a3           # n × n elements
anext:  lw    $t1, 0($a0)
        lw    $t2, 0($a1)
        addu  $t1, $t1, $t2
        sw    $t1, 0($a2)
        addiu $a0, $a0, 4
        addiu $a1, $a1, 4
        addiu $a2, $a2, 4
        addiu $t0, $t0, -1
        bnez  $t0, anext
        jr    $ra

# mmul: $a2 = $a0 × $a1, $a3 × $a3 matrices of words: c[i][j] = the sum of
# a[i][k] × b[k][j] (a leaf)
mmul:   sll   $t9, $a3, 2             # a row: 4 n bytes
        li    $t0, 0                  # i
iloop:  li    $t1, 0                  # j
jloop:  mul   $t3, $t0, $t9
        addu  $t3, $a0, $t3           # &a[i][0]: row i
        sll   $t4, $t1, 2
        addu  $t4, $a1, $t4           # &b[0][j]: column j
        li    $t5, 0                  # the sum
        move  $t2, $a3                # k: n products
kloop:  lw    $t6, 0($t3)
        lw    $t7, 0($t4)
        mul   $t6, $t6, $t7
        addu  $t5, $t5, $t6
        addiu $t3, $t3, 4             # along the row: the next word
        addu  $t4, $t4, $t9           # down the column: the next row
        addiu $t2, $t2, -1
        bnez  $t2, kloop
        sw    $t5, 0($a2)             # c[i][j], the results in order
        addiu $a2, $a2, 4
        addiu $t1, $t1, 1
        bne   $t1, $a3, jloop
        addiu $t0, $t0, 1
        bne   $t0, $a3, iloop
        jr    $ra

# print_matrix: the text at $a0, then the $a2 × $a2 words at $a1, a row a
# line (a leaf: system calls only)
print_matrix:
        li    $v0, 4
        syscall
        move  $t0, $a2                # rows
prow:   move  $t1, $a2                # columns
pcol:   li    $v0, 1
        lw    $a0, 0($a1)
        syscall
        addiu $a1, $a1, 4
        addiu $t1, $t1, -1
        li    $v0, 11
        li    $a0, ' '
        beqz  $t1, peol
        syscall
        b     pcol
peol:   li    $a0, '\n'
        syscall
        addiu $t0, $t0, -1
        bnez  $t0, prow
        jr    $ra
