# MIPS by Example in eMIPS, episode 8 — Counting letters: the most frequent
# character.  counts is an array of 26 words, one a letter: letter c's count
# is at counts + 4 * (c - 'a').  count fills it; most finds the largest and
# returns two results, the letter in $v0 and its count in $v1.
        .data
text:   .asciiz "The quick brown fox jumps over the lazy dog"
        .align 2                      # words start at a multiple of 4
counts: .space 104                    # 26 words, all 0
most1:  .asciiz "most frequent: "
times:  .asciiz " times\n"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $a0, text               # count(text, counts)
        la    $a1, counts
        jal   count
        li    $s0, 0                  # the letters counted twice or more
        la    $s1, counts
show:   lw    $t0, 0($s1)
        slti  $t1, $t0, 2
        bnez  $t1, skip
        li    $v0, 11                 # the letter ...
        addiu $a0, $s0, 'a'
        syscall
        li    $a0, ' '
        syscall
        li    $v0, 1                  # ... its count
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
skip:   addiu $s0, $s0, 1
        addiu $s1, $s1, 4             # the next word
        slti  $t1, $s0, 26
        bnez  $t1, show
        la    $a0, counts             # most(counts): $v0, $v1
        jal   most
        move  $s0, $v0
        move  $s1, $v1
        li    $v0, 4
        la    $a0, most1
        syscall
        li    $v0, 11
        move  $a0, $s0
        syscall
        li    $a0, ','
        syscall
        li    $a0, ' '
        syscall
        li    $v0, 1
        move  $a0, $s1
        syscall
        li    $v0, 4
        la    $a0, times
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# count: count the letters of the string at $a0 into the 26 words at $a1,
# capitals as small letters (a leaf)
count:  lbu   $t0, 0($a0)
        beqz  $t0, cdone
        ori   $t0, $t0, 0x20          # A-Z -> a-z
        addiu $t0, $t0, -97           # its index, c - 'a'
        sltiu $t1, $t0, 26            # a letter? (unsigned)
        beqz  $t1, cnext
        sll   $t0, $t0, 2             # a word each: index * 4
        addu  $t0, $a1, $t0           # &counts[index]
        lw    $t1, 0($t0)
        addiu $t1, $t1, 1             # one more
        sw    $t1, 0($t0)
cnext:  addiu $a0, $a0, 1
        b     count
cdone:  jr    $ra

# most: the largest of the 26 counts at $a0 -> $v0 = its letter, $v1 = the
# count (the first one, if several are equal; a leaf)
most:   li    $v0, 'a'
        lw    $v1, 0($a0)             # the best so far: a
        li    $t0, 1                  # the next index
mnext:  addiu $a0, $a0, 4
        lw    $t1, 0($a0)
        slt   $t2, $v1, $t1           # larger than the best?
        beqz  $t2, mstep
        addiu $v0, $t0, 'a'           # the new best
        move  $v1, $t1
mstep:  addiu $t0, $t0, 1
        slti  $t2, $t0, 26
        bnez  $t2, mnext
        jr    $ra
