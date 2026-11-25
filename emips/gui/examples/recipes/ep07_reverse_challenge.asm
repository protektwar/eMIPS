# MIPS by Example in eMIPS, episode 7, the challenge solved.  reverse_words
# reverses each word of a sentence in place: it finds a word's first and
# last character and hands them to reverse_range, episode 7's two-pointer
# swap with the ends given.  palindrome_loose ignores case and spaces: both
# pointers skip what is not a letter, and the letters are compared folded
# to small letters (episode 4's challenge).
        .data
sentence: .asciiz "MIPS by Example"
phrases:.word never, nolemon, notone, 0
never:  .asciiz "Never odd or even"
nolemon:.asciiz "No lemon, no melon"
notone: .asciiz "Not a palindrome"
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
        la    $a0, sentence           # reverse_words(sentence), printed
        jal   reverse_words
        li    $v0, 4
        la    $a0, sentence
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        la    $s0, phrases            # each phrase: a palindrome?
next:   lw    $s1, 0($s0)
        beqz  $s1, end
        move  $a0, $s1
        jal   palindrome_loose
        move  $s2, $v0
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

# reverse_range: swap from $a0 (left) to $a1 (right) inwards (a leaf)
reverse_range:
        bgeu  $a0, $a1, rrdone        # met or crossed: done
        lbu   $t0, 0($a0)
        lbu   $t1, 0($a1)
        sb    $t1, 0($a0)
        sb    $t0, 0($a1)
        addiu $a0, $a0, 1
        addiu $a1, $a1, -1
        b     reverse_range
rrdone: jr    $ra

# reverse_words: reverse each word of the string at $a0 in place (it calls
# reverse_range: a frame; $s0 walks the string, kept across the calls)
reverse_words:
        addiu $sp, $sp, -32           # the frame: 28 $ra, 24 $fp, 20 $s0,
        sw    $ra, 28($sp)            # 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        move  $s0, $a0
word:   lbu   $t0, 0($s0)             # skip the spaces
        beqz  $t0, rwdone
        li    $t1, ' '
        bne   $t0, $t1, inword
        addiu $s0, $s0, 1
        b     word
inword: move  $a0, $s0                # the word's first character
wend:   lbu   $t0, 1($s0)             # walk to its last: the next byte is
        beqz  $t0, wlast              # a space or the 0
        li    $t1, ' '
        beq   $t0, $t1, wlast
        addiu $s0, $s0, 1
        b     wend
wlast:  move  $a1, $s0
        jal   reverse_range
        addiu $s0, $s0, 1             # on, after the word
        b     word
rwdone: move  $sp, $fp
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# letter: $a0 = a byte -> $v0 = it as a small letter, or 0 if it is not a
# letter (a leaf)
letter: ori   $v0, $a0, 0x20          # A-Z -> a-z, a-z unchanged
        addiu $t0, $v0, -97           # a small letter? (unsigned: c - 'a' < 26)
        sltiu $t0, $t0, 26
        bnez  $t0, ldone
        move  $v0, $zero              # not a letter
ldone:  jr    $ra

# palindrome_loose: $a0 = a string -> $v0 = 1 if its letters read the same
# backwards, ignoring case and anything else (it calls letter: a frame;
# $s0 = left, $s1 = right, $s2 = left's letter, kept across the calls)
palindrome_loose:
        addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28-20 $s0-$s2,
        sw    $ra, 36($sp)            # 0-15 the home area
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        move  $fp, $sp
        move  $s0, $a0                # left
        move  $s1, $a0
plast:  lbu   $t0, 1($s1)             # right: the last character
        beqz  $t0, pair
        addiu $s1, $s1, 1
        b     plast
pair:   li    $v0, 1
        bgeu  $s0, $s1, pdone         # met: every pair of letters was equal
        lbu   $a0, 0($s0)
        jal   letter
        bnez  $v0, rightl
        addiu $s0, $s0, 1             # not a letter: skip it
        b     pair
rightl: move  $s2, $v0
        lbu   $a0, 0($s1)
        jal   letter
        bnez  $v0, cmp
        addiu $s1, $s1, -1            # not a letter: skip it
        b     pair
cmp:    bne   $s2, $v0, pno           # two letters differ: no
        addiu $s0, $s0, 1
        addiu $s1, $s1, -1
        b     pair
pno:    move  $v0, $zero
pdone:  move  $sp, $fp
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra
