# Lab 4 — this program reads a word at an address that is not a multiple
# of 4.  Write the exception handler (.ktext 0x80000180) so that it prints
# "bad address " and the address (BadVAddr), skips the instruction and
# lets the program finish.
        .data
data:   .word 0x11223344, 0x55667788
        .text
main:   la    $t0, data
        lw    $t1, 2($t0)       # misaligned: Address Error (AdEL)
        li    $v0, 10
        syscall
