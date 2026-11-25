# MIPS by Example in eMIPS, episode 3 — Strings in memory: length and the
# byte order.  A string is bytes in memory ending with a 0 byte; strlen
# counts them.  Then the same four bytes read as one word: what number is
# "MIPS"?  (Big-endian: 0x4d495053; with --little: 0x5350494d.)
        .data
word:   .asciiz "MIPS"
has:    .asciiz " has "
chars:  .asciiz " characters\nas a word: "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp, 20 $s0,
        sw    $ra, 28($sp)            # 0-15 the home area of strlen
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        la    $a0, word               # strlen(word)
        jal   strlen
        move  $s0, $v0                # keep the length
        li    $v0, 4                  # print the word
        la    $a0, word
        syscall
        li    $v0, 4
        la    $a0, has
        syscall
        li    $v0, 1                  # print the length
        move  $a0, $s0
        syscall
        li    $v0, 4
        la    $a0, chars
        syscall
        lw    $a0, word               # the four bytes as one word
        li    $v0, 34                 # print it in hex
        syscall
        li    $a0, '\n'
        li    $v0, 11
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10                 # exit
        syscall

# strlen: $a0 = the address of a string -> $v0 = the number of characters
# (a leaf: no frame)
strlen: move  $v0, $zero              # count = 0
loop:   lbu   $t0, 0($a0)             # the next byte
        beqz  $t0, done               # the 0 at the end: done
        addiu $a0, $a0, 1             # the next address
        addiu $v0, $v0, 1             # count it
        b     loop
done:   jr    $ra
