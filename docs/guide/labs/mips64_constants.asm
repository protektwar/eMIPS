# Lab 59 — How long is a constant? (Phase 5, 5.12)
# Each line loads one constant and prints it in hex (16 digits on a
# MIPS64 core).  How many instructions does each take?  See them with
#   python3 -m emips asm docs/guide/labs/mips64_constants.asm
        .text
main:   dli     $a0, 100
        jal     hex
        dli     $a0, -100
        jal     hex
        dli     $a0, 0x12340000
        jal     hex
        dli     $a0, 0xffffffff80000000
        jal     hex
        dli     $a0, 0x80000000
        jal     hex
        dli     $a0, 0xffffffff
        jal     hex
        dli     $a0, 0x0000123400000000
        jal     hex
        dli     $a0, 0x123456789abcdef0
        jal     hex
        li      $a0, 0xffffffff
        jal     hex
        la      $a0, main
        jal     hex
        li      $v0, 10
        syscall

hex:    li      $v0, 34             # print $a0 in hex, then a newline
        syscall
        li      $a0, '\n'
        li      $v0, 11
        syscall
        jr      $ra
# eMIPS: --arch mips64
