# Lab 64: SAVE and RESTORE.  fact(n) = n * fact(n - 1), recursively; it
# should print 120.
        .text
main:   li      $a0, 5
        balc    fact
        li      $t4, 1              # print_int $a0
        syscall
        li      $t4, 10             # exit
        syscall

# fact: $a0 = n -> $a0 = n!  ($s0 must survive a call: the p32 ABI)
fact:   save    16, $ra             # push $ra, make a 16-byte frame
        move    $s0, $a0
        bltic   $a0, 2, one
        addiu   $a0, $a0, -1
        balc    fact
        mul     $a0, $a0, $s0
        restore.jrc 16, $ra         # pop $ra, free the frame, return
one:    li      $a0, 1
        restore.jrc 16, $ra
# eMIPS: --arch nanomips
