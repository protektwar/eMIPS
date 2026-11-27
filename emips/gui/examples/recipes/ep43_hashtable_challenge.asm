# MIPS by Example in eMIPS, episode 43 — the challenge, solved: a good hash
# and a bad one.  The same words counted twice: with h = h * 31 + c, and
# with the sum of the letters — which gives every anagram (cat, act, tac)
# the same slot.  probes counts every slot a lookup looks at: the bad hash
# makes the table walk.  (The episode: a hash table.)
        .data
text1:  .asciiz "the cat and the dog and the bird saw the cat act tac god was"
text2:  .asciiz "the cat and the dog and the bird saw the cat act tac god was"
        .eqv  SLOTS, 16
        .align 2                      # (the table holds words: aligned)
keys:   .space 4 * SLOTS              # a pointer to the word, 0 = empty
counts: .space 4 * SLOTS              # how often it came
sumhash: .word 0                      # 1: the sum of the letters as the hash
probes: .word 0                       # slots looked at
lslot:  .asciiz "slot "
colon:  .asciiz ": "
lgood:  .asciiz "h = h * 31 + c\n"
lbad:   .asciiz "the sum of the letters\n"
lprobe: .asciiz "probes: "

        .text
main:   addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        la    $a0, text1              # words(text, sum?, label)
        li    $a1, 0
        la    $a2, lgood
        jal   words
        la    $a0, text2
        li    $a1, 1
        la    $a2, lbad
        jal   words
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# words(text, sum, label): counts the text's words in an empty table with
# that hash, prints the label, the table and the probes
words:  addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28 $s2, 24 $s1, 20 $s0
        sw    $ra, 36($sp)
        sw    $fp, 32($sp)
        sw    $s2, 28($sp)
        sw    $s1, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        move  $s0, $a0                # $s0: the text, read word by word
        move  $s2, $a2                # $s2: the label
        sw    $a1, sumhash
        sw    $zero, probes
        li    $t0, 0                  # an empty table
clear:  sw    $zero, keys($t0)
        sw    $zero, counts($t0)
        addiu $t0, $t0, 4
        blt   $t0, 4 * SLOTS, clear
word:   lbu   $t0, 0($s0)
        beqz  $t0, table              # the end of the text
        move  $s1, $s0                # $s1: the word's start
scan:   lbu   $t0, 0($s0)             # find its end: a space or the end
        beqz  $t0, found
        beq   $t0, ' ', split
        addiu $s0, $s0, 1
        b     scan
split:  sb    $zero, 0($s0)           # the space becomes the word's end (0)
        addiu $s0, $s0, 1
found:  move  $a0, $s1                # count(word)
        jal   count
        b     word
table:  li    $v0, 4                  # the label
        move  $a0, $s2
        syscall
        li    $s0, 0                  # print every slot in use
show:   sll   $t0, $s0, 2
        lw    $t1, keys($t0)
        beqz  $t1, snext
        li    $v0, 4                  # "slot n: word count"
        la    $a0, lslot
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 4
        la    $a0, colon
        syscall
        li    $v0, 4
        move  $a0, $t1
        syscall
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        lw    $a0, counts($t0)
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
snext:  addiu $s0, $s0, 1
        blt   $s0, SLOTS, show
        li    $v0, 4
        la    $a0, lprobe
        syscall
        li    $v0, 1
        lw    $a0, probes
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp
        lw    $s0, 20($sp)
        lw    $s1, 24($sp)
        lw    $s2, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra

# count(word): one more for this word — its slot found by hashing and
# probing, an empty slot taken when it is new
count:  addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s1, 16 $s0
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s1, 20($sp)
        sw    $s0, 16($sp)
        move  $fp, $sp
        move  $s0, $a0                # $s0: the word
        jal   hash
        andi  $s1, $v0, SLOTS - 1     # $s1: the slot = the hash's low 4 bits
probe:  lw    $t0, probes          # one slot more looked at
        addiu $t0, $t0, 1
        sw    $t0, probes
        sll   $t0, $s1, 2
        lw    $a1, keys($t0)
        beqz  $a1, new                # empty: the word is new
        move  $a0, $s0
        jal   streq                   # this slot's word?
        bnez  $v0, again
        addiu $s1, $s1, 1             # a collision: the next slot,
        andi  $s1, $s1, SLOTS - 1     # around the table
        b     probe
new:    sw    $s0, keys($t0)          # the word in the empty slot
again:  sll   $t0, $s1, 2
        lw    $t1, counts($t0)        # one more
        addiu $t1, $t1, 1
        sw    $t1, counts($t0)
        move  $sp, $fp
        lw    $s0, 16($sp)
        lw    $s1, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# hash(s): h = h * 31 + c over the characters, or (sumhash = 1) h = h + c
# (a leaf)
hash:   li    $v0, 0
        lw    $t2, sumhash
hloop:  lbu   $t0, 0($a0)
        beqz  $t0, hend
        bnez  $t2, hsum
        sll   $t1, $v0, 5             # h * 31 = h * 32 - h
        subu  $v0, $t1, $v0
hsum:   addu  $v0, $v0, $t0
        addiu $a0, $a0, 1
        b     hloop
hend:   jr    $ra

# streq(a, b): 1 when the two strings are equal, else 0 (a leaf)
streq:  lbu   $t0, 0($a0)
        lbu   $t1, 0($a1)
        bne   $t0, $t1, sdiff
        beqz  $t0, ssame              # both ended together
        addiu $a0, $a0, 1
        addiu $a1, $a1, 1
        b     streq
ssame:  li    $v0, 1
        jr    $ra
sdiff:  li    $v0, 0
        jr    $ra
