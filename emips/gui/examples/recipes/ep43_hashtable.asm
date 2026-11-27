# MIPS by Example in eMIPS, episode 43 — A hash table: counting words.  A
# hash function turns a word into a number (h = h * 31 + c, for every
# character); its low 4 bits pick one of 16 slots.  Two words may want the
# same slot — a collision: then the next free slot is taken (linear
# probing), and a lookup walks the same way until it finds the word or an
# empty slot.  Finding a word costs a few steps, not a walk over all words.
        .data
text:   .asciiz "the cat and the dog and the bird saw the cat"
        .eqv  SLOTS, 16
        .align 2                      # (the table holds words: aligned)
keys:   .space 4 * SLOTS              # a pointer to the word, 0 = empty
counts: .space 4 * SLOTS              # how often it came
lslot:  .asciiz "slot "
colon:  .asciiz ": "
words:  .asciiz "different words: "

        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s1, 16 $s0, 0-15 the home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s1, 20($sp)
        sw    $s0, 16($sp)
        move  $fp, $sp
        la    $s0, text               # $s0: the text, read word by word
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
table:  li    $s0, 0                  # print every slot in use
        li    $s1, 0                  # (and how many there are)
show:   sll   $t0, $s0, 2
        lw    $t1, keys($t0)
        beqz  $t1, snext
        addiu $s1, $s1, 1
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
        la    $a0, words
        syscall
        li    $v0, 1
        move  $a0, $s1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s0, 16($sp)
        lw    $s1, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

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
probe:  sll   $t0, $s1, 2
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

# hash(s): h = h * 31 + c over the characters (a leaf)
hash:   li    $v0, 0
hloop:  lbu   $t0, 0($a0)
        beqz  $t0, hend
        sll   $t1, $v0, 5             # h * 31 = h * 32 - h
        subu  $v0, $t1, $v0
        addu  $v0, $v0, $t0
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
