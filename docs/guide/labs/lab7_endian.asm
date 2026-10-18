# Lab 7 — which byte of a word is at the lowest address?
        .data
word:   .word 0x11223344
        .text
main:   la    $t0, word
        lbu   $a0, 0($t0)       # the byte at the word's own address
        li    $v0, 34           # print_hex
        syscall
        li    $v0, 10
        syscall
