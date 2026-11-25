# Lab 95 — A library file: compare three pairs of words with strcmp from
# strlib.asm (.include) and print -, 0 or + for each, as C's strcmp
# would order them.  It prints the wrong signs: find the bug in the library.
# Run:  python3 -m emips debug docs/guide/labs/strlib_main.asm
        .text
main:   la    $s0, pairs
next:   lw    $a0, 0($s0)             # the next pair (0: the end)
        beqz  $a0, done
        lw    $a1, 4($s0)
        jal   strcmp
        li    $a0, '0'
        beqz  $v0, show
        li    $a0, '-'
        bltz  $v0, show
        li    $a0, '+'
show:   li    $v0, 11
        syscall
        li    $a0, ' '
        syscall
        addiu $s0, $s0, 8
        b     next
done:   li    $v0, 10
        syscall

        .include "strlib.asm"

        .data
apple:  .asciiz "apple"
apply:  .asciiz "apply"
pear:   .asciiz "pear"
peach:  .asciiz "peach"
pairs:  .word apple, apple, apple, apply, pear, peach, 0
