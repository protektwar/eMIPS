# MIPS by Example in eMIPS, episode 6, the challenge solved.  swapcase flips
# every letter: a capital and its small letter differ only in bit 5 (0x20),
# so one xori does both directions.  title makes every word start with a
# capital and go on in small letters: a letter right after the start or a
# space gets bit 5 cleared, any other letter gets it set.
        .data
buf:    .space 64
ask:    .asciiz "Your name? "
swp:    .asciiz "Swapped: "
ttl:    .asciiz "\nTitle: "
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        li    $v0, 4
        la    $a0, ask
        syscall
        li    $v0, 8                  # read_string(buf, 64)
        la    $a0, buf
        li    $a1, 64
        syscall
        la    $a0, buf
        jal   chomp
        la    $a0, buf                # swapcase(buf), printed
        jal   swapcase
        li    $v0, 4
        la    $a0, swp
        syscall
        la    $a0, buf
        syscall
        la    $a0, buf                # title(buf), printed
        jal   title
        li    $v0, 4
        la    $a0, ttl
        syscall
        la    $a0, buf
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# chomp: episode 6's (a leaf)
chomp:  lbu   $t0, 0($a0)
        beqz  $t0, chdone
        li    $t1, '\n'
        beq   $t0, $t1, cut
        addiu $a0, $a0, 1
        b     chomp
cut:    sb    $zero, 0($a0)
chdone: jr    $ra

# swapcase: flip the case of every letter at $a0, in place (a leaf)
swapcase:
        lbu   $t0, 0($a0)
        beqz  $t0, swdone
        ori   $t1, $t0, 0x20          # a letter? its small form - 'a' < 26
        addiu $t1, $t1, -97
        sltiu $t1, $t1, 26
        beqz  $t1, swnext
        xori  $t0, $t0, 0x20          # bit 5 flipped: A <-> a
        sb    $t0, 0($a0)
swnext: addiu $a0, $a0, 1
        b     swapcase
swdone: jr    $ra

# title: every word's first letter a capital, the others small (a leaf;
# $t2 = 1 while the next letter starts a word)
title:  li    $t2, 1                  # the start: a word begins
tinext: lbu   $t0, 0($a0)
        beqz  $t0, tidone
        li    $t1, ' '
        bne   $t0, $t1, tletter
        li    $t2, 1                  # a space: the next letter begins a word
        b     tistep
tletter:
        ori   $t1, $t0, 0x20          # a letter?
        addiu $t1, $t1, -97
        sltiu $t1, $t1, 26
        beqz  $t1, tistep
        ori   $t0, $t0, 0x20          # small ...
        beqz  $t2, tstore
        xori  $t0, $t0, 0x20          # ... or, at a word's start, capital
tstore: sb    $t0, 0($a0)
        move  $t2, $zero              # inside a word now
tistep: addiu $a0, $a0, 1
        b     tinext
tidone: jr    $ra
