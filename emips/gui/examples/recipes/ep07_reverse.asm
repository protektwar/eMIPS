# MIPS by Example in eMIPS, episode 7 — Reverse in place and palindromes:
# two pointers, one at each end, walking towards each other — swapping the
# characters to reverse a string, comparing them to test a palindrome.
        .data
text:   .asciiz "stressed"
words:  .word racecar, level, mips, 0
racecar:.asciiz "racecar"
level:  .asciiz "level"
mips:   .asciiz "MIPS"
yes:    .asciiz ": a palindrome\n"
no:     .asciiz ": not a palindrome\n"
        .text
main:   addiu $sp, $sp, -40           # main's frame: 36 $ra, 32 $fp,
        sw    $ra, 36($sp)            # 28-20 $s0-$s2, 0-15 the home area
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        move  $fp, $sp
        la    $a0, text               # reverse(text) and print it
        jal   reverse
        li    $v0, 4
        la    $a0, text
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        la    $s0, words              # each word: a palindrome?
next:   lw    $s1, 0($s0)
        beqz  $s1, end
        move  $a0, $s1
        jal   palindrome
        move  $s2, $v0                # 1 or 0
        li    $v0, 4
        move  $a0, $s1
        syscall
        la    $a0, yes
        bnez  $s2, say
        la    $a0, no
say:    syscall
        addiu $s0, $s0, 4
        b     next
end:    move  $sp, $fp                # main's epilogue
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        li    $v0, 10
        syscall

# last: $a0 = a string -> $v0 = the address of its last character (a leaf)
last:   move  $v0, $a0
last1:  lbu   $t0, 1($v0)             # is the next one the 0?
        beqz  $t0, last2
        addiu $v0, $v0, 1
        b     last1
last2:  jr    $ra

# reverse: reverse the string at $a0 in place (it calls last: a frame)
reverse:
        addiu $sp, $sp, -32           # the frame: 28 $ra, 24 $fp, 0-15 home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        move  $fp, $sp
        sw    $a0, 32($fp)            # the string, in the caller's home area:
        jal   last                    # a call may change $a0
        lw    $t1, 32($fp)            # left: the first character
        move  $t2, $v0                # right: the last
swap:   bgeu  $t1, $t2, swapped       # met in the middle: done
        lbu   $t3, 0($t1)
        lbu   $t4, 0($t2)
        sb    $t4, 0($t1)
        sb    $t3, 0($t2)
        addiu $t1, $t1, 1
        addiu $t2, $t2, -1
        b     swap
swapped:
        move  $sp, $fp                # restore, in reverse
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# palindrome: $a0 = a string -> $v0 = 1 if it reads the same backwards
palindrome:
        addiu $sp, $sp, -32           # the frame: 28 $ra, 24 $fp, 0-15 home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        move  $fp, $sp
        sw    $a0, 32($fp)            # the string, kept across the call
        jal   last
        lw    $t1, 32($fp)            # left
        move  $t2, $v0                # right
same:   li    $v0, 1
        bgeu  $t1, $t2, decided       # met: every pair was equal
        lbu   $t3, 0($t1)
        lbu   $t4, 0($t2)
        li    $v0, 0
        bne   $t3, $t4, decided       # a pair differs: no
        addiu $t1, $t1, 1
        addiu $t2, $t2, -1
        b     same
decided:
        move  $sp, $fp                # restore, in reverse
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra
