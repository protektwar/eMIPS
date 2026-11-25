# MIPS by Example in eMIPS, episode 8, the challenge solved.  bars draws a
# bar of stars for every letter that appears; ties lists every letter whose
# count equals the largest one (most finds it, as in episode 8).
        .data
text:   .asciiz "MIPS by Example in eMIPS"
        .align 2
counts: .space 104                    # 26 words, all 0
top:    .asciiz "most frequent, "
tail:   .asciiz " times:"
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
        la    $a0, counts             # bars(counts)
        jal   bars
        la    $a0, counts             # most(counts): $v1 = the largest count
        jal   most
        move  $s0, $v1
        li    $v0, 4
        la    $a0, top
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 4
        la    $a0, tail
        syscall
        la    $a0, counts             # ties(counts, the largest)
        move  $a1, $s0
        jal   ties
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

# count: episode 8's (a leaf)
count:  lbu   $t0, 0($a0)
        beqz  $t0, cdone
        ori   $t0, $t0, 0x20
        addiu $t0, $t0, -97
        sltiu $t1, $t0, 26
        beqz  $t1, cnext
        sll   $t0, $t0, 2
        addu  $t0, $a1, $t0
        lw    $t1, 0($t0)
        addiu $t1, $t1, 1
        sw    $t1, 0($t0)
cnext:  addiu $a0, $a0, 1
        b     count
cdone:  jr    $ra

# most: episode 8's (a leaf)
most:   li    $v0, 'a'
        lw    $v1, 0($a0)
        li    $t0, 1
mnext:  addiu $a0, $a0, 4
        lw    $t1, 0($a0)
        slt   $t2, $v1, $t1
        beqz  $t2, mstep
        addiu $v0, $t0, 'a'
        move  $v1, $t1
mstep:  addiu $t0, $t0, 1
        slti  $t2, $t0, 26
        bnez  $t2, mnext
        jr    $ra

# bars: for each letter counted, "x ***" with one star a time (a leaf:
# only system calls; $t2 = the letter's index, $t3 = its count)
bars:   move  $t0, $a0                # the next count
        li    $t2, 0
bnext:  lw    $t3, 0($t0)
        beqz  $t3, bskip
        li    $v0, 11
        addiu $a0, $t2, 'a'           # the letter, a space
        syscall
        li    $a0, ' '
        syscall
        li    $a0, '*'
star:   syscall                       # $t3 stars
        addiu $t3, $t3, -1
        bnez  $t3, star
        li    $a0, '\n'
        syscall
bskip:  addiu $t0, $t0, 4
        addiu $t2, $t2, 1
        slti  $t1, $t2, 26
        bnez  $t1, bnext
        jr    $ra

# ties: print " x" for every letter whose count is $a1 (a leaf)
ties:   move  $t0, $a0
        li    $t2, 0
tnext:  lw    $t3, 0($t0)
        bne   $t3, $a1, tskip
        li    $v0, 11
        li    $a0, ' '
        syscall
        addiu $a0, $t2, 'a'
        syscall
tskip:  addiu $t0, $t0, 4
        addiu $t2, $t2, 1
        slti  $t1, $t2, 26
        bnez  $t1, tnext
        jr    $ra
