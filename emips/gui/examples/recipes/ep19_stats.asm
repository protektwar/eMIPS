# MIPS by Example in eMIPS, episode 19 — Multiples and statistics.  The
# multiples of 3 or 5 below 1000: those of 3 plus those of 5, minus those of
# 15, counted twice.  Ten scores: the smallest, the largest, the mean rounded
# to the nearest whole number, and how many are above it.
        .data
scores: .word 72, 85, 90, 64, 78, 95, 58, 81, 88, 74
mult:   .asciiz "multiples of 3 or 5 below 1000: "
scorew: .asciiz "scores:"
minw:   .asciiz "min "
maxw:   .asciiz ", max "
meanw:  .asciiz ", mean "
abovew: .asciiz ", above the mean "
        .text
main:   addiu $sp, $sp, -40           # main's frame: 36 $ra, 32 $fp,
        sw    $ra, 36($sp)            # 28-20 $s0-$s2, 0-15 the home area
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        move  $fp, $sp
        li    $a0, 1000               # those of 3 ...
        li    $a1, 3
        jal   sum_mult
        move  $s0, $v0
        li    $a0, 1000               # ... plus those of 5 ...
        li    $a1, 5
        jal   sum_mult
        addu  $s0, $s0, $v0
        li    $a0, 1000               # ... minus those of 15: counted twice
        li    $a1, 15
        jal   sum_mult
        subu  $s0, $s0, $v0
        li    $v0, 4
        la    $a0, mult
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        la    $a0, scores             # the scores, on one line
        li    $a1, 10
        jal   print_array
        la    $a0, scores             # min and max: two results
        li    $a1, 10
        jal   minmax
        move  $s0, $v0
        move  $s1, $v1
        la    $a0, scores             # the mean, rounded
        li    $a1, 10
        jal   mean
        move  $s2, $v0
        la    $a0, scores             # how many above it
        li    $a1, 10
        move  $a2, $s2
        jal   above
        move  $t0, $v0
        li    $v0, 4
        la    $a0, minw
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 4
        la    $a0, maxw
        syscall
        li    $v0, 1
        move  $a0, $s1
        syscall
        li    $v0, 4
        la    $a0, meanw
        syscall
        li    $v0, 1
        move  $a0, $s2
        syscall
        li    $v0, 4
        la    $a0, abovew
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        li    $v0, 10
        syscall

# sum_mult: $a0 = limit, $a1 = k -> $v0 = k + 2k + 3k + ... below that
# limit, with no division: just steps of k (a leaf)
sum_mult:
        li    $v0, 0
        move  $t0, $a1                # the first multiple
smnext: bge   $t0, $a0, smdone
        addu  $v0, $v0, $t0
        addu  $t0, $t0, $a1           # the next one
        b     smnext
smdone: jr    $ra

# minmax: $a0 = the words, $a1 = n -> $v0 = the smallest, $v1 = the
# largest (a leaf)
minmax: lw    $v0, 0($a0)             # both start at the first
        move  $v1, $v0
mmnext: addiu $a1, $a1, -1
        beqz  $a1, mmdone
        addiu $a0, $a0, 4
        lw    $t0, 0($a0)
        bge   $t0, $v0, notmin
        move  $v0, $t0                # a new smallest
notmin: ble   $t0, $v1, mmnext
        move  $v1, $t0                # a new largest
        b     mmnext
mmdone: jr    $ra

# mean: $a0 = the words, $a1 = n -> $v0 = (sum + n / 2) / n, the mean
# rounded to the nearest (a leaf)
mean:   li    $t0, 0                  # the sum
        move  $t1, $a1
mnext:  lw    $t2, 0($a0)
        addu  $t0, $t0, $t2
        addiu $a0, $a0, 4
        addiu $t1, $t1, -1
        bnez  $t1, mnext
        srl   $t2, $a1, 1             # + n / 2: rounds instead of cutting
        addu  $t0, $t0, $t2
        divu  $t0, $a1
        mflo  $v0
        jr    $ra

# above: $a0 = the words, $a1 = n, $a2 = m -> $v0 = how many are > m
# (a leaf)
above:  li    $v0, 0
anext:  lw    $t0, 0($a0)
        ble   $t0, $a2, askip
        addiu $v0, $v0, 1
askip:  addiu $a0, $a0, 4
        addiu $a1, $a1, -1
        bnez  $a1, anext
        jr    $ra

# print_array: "scores:" and the $a1 words at $a0, each after a space
# (a leaf: system calls only)
print_array:
        move  $t0, $a0
        li    $v0, 4
        la    $a0, scorew
        syscall
panext: li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        lw    $a0, 0($t0)
        syscall
        addiu $t0, $t0, 4
        addiu $a1, $a1, -1
        bnez  $a1, panext
        li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra
