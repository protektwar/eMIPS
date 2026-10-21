# Lab 58 — Unsigned words in a 64-bit register (Phase 5, 5.12)
# The sizes of five files, in bytes: unsigned 32-bit words, each up to
# 4 GB.  Their total does not fit in 32 bits, so it is added up in a
# 64-bit register — and printed with print_int, which prints the whole
# register on a MIPS64 core.  It should print 9500000000.
        .data
sizes:  .word   3000000000, 2500000000, 1500000000, 2000000000, 500000000
        .text
main:   la      $t0, sizes          # the next size
        li      $t1, 5              # how many are left
        move    $t2, $zero          # the total (64 bits)
loop:   lw      $t3, 0($t0)         # one size
        daddu   $t2, $t2, $t3       # total += size
        addiu   $t0, $t0, 4
        addiu   $t1, $t1, -1
        bnez    $t1, loop
        move    $a0, $t2
        li      $v0, 1              # print_int: the total
        syscall
        li      $v0, 10
        syscall
# eMIPS: --arch mips64
