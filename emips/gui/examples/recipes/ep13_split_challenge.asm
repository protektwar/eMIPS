# MIPS by Example in eMIPS, episode 13, the challenge solved.  After split
# (episode 13's), join writes the words back into a new line with exactly
# one space between them, and longest finds the longest word: both walk the
# array of addresses split made.
        .data
buf:    .space 64
line:   .space 64
        .align 2
words:  .space 64
ask:    .asciiz "A line? "
joined: .asciiz "joined: \""
quote:  .asciiz "\"\nlongest: "
open:   .asciiz " ("
close:  .asciiz ")\n"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        li    $v0, 4
        la    $a0, ask
        syscall
        li    $v0, 8
        la    $a0, buf
        li    $a1, 64
        syscall
        la    $a0, buf                # split(buf, words, 16)
        la    $a1, words
        li    $a2, 16
        jal   split
        move  $s0, $v0
        la    $a0, line               # join(line, words, n)
        la    $a1, words
        move  $a2, $s0
        jal   join
        li    $v0, 4
        la    $a0, joined
        syscall
        la    $a0, line
        syscall
        la    $a0, quote
        syscall
        la    $a0, words              # longest(words, n): $v0 = the word,
        move  $a1, $s0                # $v1 = its length
        jal   longest
        move  $s1, $v1
        move  $a0, $v0
        li    $v0, 4
        syscall
        la    $a0, open
        syscall
        li    $v0, 1
        move  $a0, $s1
        syscall
        li    $v0, 4
        la    $a0, close
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# split: episode 13's (a leaf)
split:  li    $v0, 0
        li    $t1, ' '
        li    $t2, '\n'
skip:   lbu   $t0, 0($a0)
        bne   $t0, $t1, wstart
        addiu $a0, $a0, 1
        b     skip
wstart: beqz  $t0, sdone
        beq   $t0, $t2, sdone
        beq   $v0, $a2, sdone
        sw    $a0, 0($a1)
        addiu $a1, $a1, 4
        addiu $v0, $v0, 1
inword: addiu $a0, $a0, 1
        lbu   $t0, 0($a0)
        beqz  $t0, sdone
        beq   $t0, $t2, cutnl
        bne   $t0, $t1, inword
        sb    $zero, 0($a0)
        addiu $a0, $a0, 1
        b     skip
cutnl:  sb    $zero, 0($a0)
sdone:  jr    $ra

# join: write the $a2 words whose addresses are at $a1 into $a0, one space
# between them (a leaf)
join:   beqz  $a2, jdone
jword:  lw    $t0, 0($a1)             # copy one word
jcopy:  lbu   $t1, 0($t0)
        beqz  $t1, jnext
        sb    $t1, 0($a0)
        addiu $t0, $t0, 1
        addiu $a0, $a0, 1
        b     jcopy
jnext:  addiu $a1, $a1, 4
        addiu $a2, $a2, -1
        beqz  $a2, jdone
        li    $t1, ' '                # a space before the next one
        sb    $t1, 0($a0)
        addiu $a0, $a0, 1
        b     jword
jdone:  sb    $zero, 0($a0)
        jr    $ra

# longest: the longest of the $a1 words whose addresses are at $a0 -> $v0 =
# its address, $v1 = its length (the first one, if several are equal; a leaf)
longest:
        li    $v0, 0
        li    $v1, -1
lnext:  beqz  $a1, ldone
        lw    $t0, 0($a0)             # this word's length
        move  $t1, $t0
llen:   lbu   $t2, 0($t1)
        beqz  $t2, lcmp
        addiu $t1, $t1, 1
        b     llen
lcmp:   subu  $t1, $t1, $t0
        ble   $t1, $v1, lstep         # longer than the best?
        move  $v0, $t0
        move  $v1, $t1
lstep:  addiu $a0, $a0, 4
        addiu $a1, $a1, -1
        b     lnext
ldone:  jr    $ra
