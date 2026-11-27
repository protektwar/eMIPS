# MIPS by Example in eMIPS, episode 43, the RISC-V ending: the hash table on
# RV32.  h * 31 is h * 32 - h here too (slli, sub); RISC-V has no index-plus-
# label loads, so each table access adds its slot offset to the table's
# address first.
# eMIPS: --arch rv32
        .data
text:   .asciz "the cat and the dog and the bird saw the cat"
        .eqv  SLOTS, 16
        .align 2
keys:   .space 4 * SLOTS              # a pointer to the word, 0 = empty
counts: .space 4 * SLOTS              # how often it came
lslot:  .asciz "slot "
colon:  .asciz ": "
words:  .asciz "different words: "

        .text
main:   addi  sp, sp, -16             # 12 ra, 8 s0 (the frame pointer), 4 s1, 0 s2
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        sw    s1, 4(sp)
        sw    s2, 0(sp)
        addi  s0, sp, 16
        la    s1, text                # s1: the text, read word by word
word:   lbu   t0, 0(s1)
        beqz  t0, table               # the end of the text
        mv    s2, s1                  # s2: the word's start
scan:   lbu   t0, 0(s1)               # its end: a space or the end
        beqz  t0, found
        li    t1, ' '
        beq   t0, t1, split
        addi  s1, s1, 1
        j     scan
split:  sb    zero, 0(s1)             # the space becomes the word's end
        addi  s1, s1, 1
found:  mv    a0, s2                  # count(word)
        call  count
        j     word
table:  li    s1, 0                   # every slot in use
        li    s2, 0                   # (and how many)
show:   slli  t0, s1, 2
        la    t1, keys
        add   t1, t1, t0
        lw    t1, 0(t1)
        beqz  t1, snext
        addi  s2, s2, 1
        la    a0, lslot               # "slot n: word count"
        li    a7, 4
        ecall
        mv    a0, s1
        li    a7, 1
        ecall
        la    a0, colon
        li    a7, 4
        ecall
        mv    a0, t1
        li    a7, 4
        ecall
        li    a0, ' '
        li    a7, 11
        ecall
        slli  t0, s1, 2
        la    t1, counts
        add   t1, t1, t0
        lw    a0, 0(t1)
        li    a7, 1
        ecall
        li    a0, '\n'
        li    a7, 11
        ecall
snext:  addi  s1, s1, 1
        li    t0, SLOTS
        blt   s1, t0, show
        la    a0, words
        li    a7, 4
        ecall
        mv    a0, s2
        li    a7, 1
        ecall
        li    a0, '\n'
        li    a7, 11
        ecall
        lw    s2, 0(sp)
        lw    s1, 4(sp)
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        li    a7, 10
        ecall

# count(word): one more for this word
count:  addi  sp, sp, -16             # 12 ra, 8 s0, 4 s1, 0 s2
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        sw    s1, 4(sp)
        sw    s2, 0(sp)
        addi  s0, sp, 16
        mv    s1, a0                  # s1: the word
        call  hash
        andi  s2, a0, SLOTS - 1       # s2: the slot
probe:  slli  t0, s2, 2
        la    t1, keys
        add   t1, t1, t0
        lw    a1, 0(t1)
        beqz  a1, new                 # empty: the word is new
        mv    a0, s1
        call  streq                   # this slot's word?
        bnez  a0, again
        addi  s2, s2, 1               # a collision: the next slot,
        andi  s2, s2, SLOTS - 1       # around the table
        j     probe
new:    slli  t0, s2, 2
        la    t1, keys
        add   t1, t1, t0
        sw    s1, 0(t1)
again:  slli  t0, s2, 2
        la    t1, counts
        add   t1, t1, t0
        lw    t2, 0(t1)
        addi  t2, t2, 1
        sw    t2, 0(t1)
        lw    s2, 0(sp)
        lw    s1, 4(sp)
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        ret

# hash(s): h = h * 31 + c (a leaf)
hash:   li    t2, 0
hloop:  lbu   t0, 0(a0)
        beqz  t0, hend
        slli  t1, t2, 5               # h * 31 = h * 32 - h
        sub   t2, t1, t2
        add   t2, t2, t0
        addi  a0, a0, 1
        j     hloop
hend:   mv    a0, t2
        ret

# streq(a, b): 1 when equal (a leaf)
streq:  lbu   t0, 0(a0)
        lbu   t1, 0(a1)
        bne   t0, t1, sdiff
        beqz  t0, ssame
        addi  a0, a0, 1
        addi  a1, a1, 1
        j     streq
ssame:  li    a0, 1
        ret
sdiff:  li    a0, 0
        ret
