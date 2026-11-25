# MIPS by Example in eMIPS, episode 24, the challenge solved.  transpose
# swaps a[i][j] with a[j][i] above the diagonal only (below it, each pair
# would be swapped back).  trace adds the diagonal: from one diagonal
# element to the next is one row and one column, (n + 1) × 4 bytes.
        .data
ma:     .word 1, 2, 3
        .word 4, 5, 6
        .word 7, 8, 9
transw: .asciiz "transposed:\n"
tracew: .asciiz "trace "
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        la    $a0, ma                 # transpose in place
        li    $a1, 3
        jal   transpose
        la    $a0, transw
        la    $a1, ma
        li    $a2, 3
        jal   print_matrix
        la    $a0, ma                 # the trace
        li    $a1, 3
        jal   trace
        move  $t0, $v0
        li    $v0, 4
        la    $a0, tracew
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# transpose: the $a1 × $a1 words at $a0, in place: for each i, swap
# a[i][j] and a[j][i] for j > i (a leaf)
transpose:
        sll   $t9, $a1, 2             # a row: 4 n bytes
        li    $t0, 0                  # i
tirow:  addiu $t1, $t0, 1             # j starts right of the diagonal
tjcol:  bge   $t1, $a1, tinext
        mul   $t2, $t0, $t9           # &a[i][j] = base + i × 4n + j × 4
        sll   $t3, $t1, 2
        addu  $t2, $t2, $t3
        addu  $t2, $a0, $t2
        mul   $t3, $t1, $t9           # &a[j][i] = base + j × 4n + i × 4
        sll   $t4, $t0, 2
        addu  $t3, $t3, $t4
        addu  $t3, $a0, $t3
        lw    $t4, 0($t2)
        lw    $t5, 0($t3)
        sw    $t5, 0($t2)
        sw    $t4, 0($t3)
        addiu $t1, $t1, 1
        b     tjcol
tinext: addiu $t0, $t0, 1
        bne   $t0, $a1, tirow
        jr    $ra

# trace: $a0 = an $a1 × $a1 matrix -> $v0 = the sum of its diagonal
# (a leaf)
trace:  li    $v0, 0
        addiu $t0, $a1, 1
        sll   $t0, $t0, 2             # (n + 1) × 4: the next diagonal element
        move  $t1, $a1
tnext:  lw    $t2, 0($a0)
        addu  $v0, $v0, $t2
        addu  $a0, $a0, $t0
        addiu $t1, $t1, -1
        bnez  $t1, tnext
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
