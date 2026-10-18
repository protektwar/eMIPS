# Three pages of data: each first touch of a page pair is a TLB refill.
# Run it with --mmu tlb (the boot firmware's refill handler maps them).
        .data
a:      .word 1
        .space 8188             # b is 8 KB further: the next page pair
b:      .word 2
        .text
main:   la    $t0, a
        lw    $t1, 0($t0)       # refill: the pair of a
        la    $t0, b
        lw    $t2, 0($t0)       # refill: the pair of b
        lw    $t3, 4($t0)       # same page: no refill
        addu  $a0, $t1, $t2
        li    $v0, 1            # print_int
        syscall
        li    $v0, 10
        syscall
