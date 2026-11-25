# MIPS by Example in eMIPS, episode 13 — Splitting a line into words: split
# walks the line once, skipping spaces; at each word it stores the word's
# address in an array and, where the word ends, writes a 0 over the space:
# the words become strings of their own, in place (like C's strtok).
        .data
buf:    .space 64
        .align 2
words:  .space 64                     # up to 16 addresses
ask:    .asciiz "A line? "
nwords: .asciiz " words:\n"
sep:    .asciiz ": "
        .text
main:   addiu $sp, $sp, -40           # main's frame: 36 $ra, 32 $fp,
        sw    $ra, 36($sp)            # 28-20 $s0-$s2, 0-15 the home area
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        move  $fp, $sp
        li    $v0, 4                  # ask, read a line
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
        move  $s2, $v0                # the number of words
        li    $v0, 1
        move  $a0, $s2
        syscall
        li    $v0, 4
        la    $a0, nwords
        syscall
        li    $s0, 1                  # word number $s0 ...
        la    $s1, words              # ... its address at $s1
show:   bgt   $s0, $s2, end
        li    $v0, 1                  # "n: word"
        move  $a0, $s0
        syscall
        li    $v0, 4
        la    $a0, sep
        syscall
        lw    $a0, 0($s1)
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        addiu $s0, $s0, 1
        addiu $s1, $s1, 4
        b     show
end:    move  $sp, $fp                # main's epilogue
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        li    $v0, 10
        syscall

# split: cut the line at $a0 into words, their addresses into the array at
# $a1 (at most $a2 of them) -> $v0 = the number of words (a leaf; a space,
# the newline or the 0 ends a word)
split:  li    $v0, 0
        li    $t1, ' '
        li    $t2, '\n'
skip:   lbu   $t0, 0($a0)             # skip the spaces before a word
        bne   $t0, $t1, wstart
        addiu $a0, $a0, 1
        b     skip
wstart: beqz  $t0, sdone              # the end of the line: no more words
        beq   $t0, $t2, sdone
        beq   $v0, $a2, sdone         # the array is full
        sw    $a0, 0($a1)             # a word starts here
        addiu $a1, $a1, 4
        addiu $v0, $v0, 1
inword: addiu $a0, $a0, 1             # to the end of the word
        lbu   $t0, 0($a0)
        beqz  $t0, sdone
        beq   $t0, $t2, cutnl
        bne   $t0, $t1, inword
        sb    $zero, 0($a0)           # the space after it becomes its end
        addiu $a0, $a0, 1
        b     skip
cutnl:  sb    $zero, 0($a0)           # so does the newline
sdone:  jr    $ra
