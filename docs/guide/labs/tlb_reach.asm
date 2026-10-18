# Lab: the reach of the TLB.
# Read one word in each of PAIRS page pairs (8 KB apart), twice.  A TLB
# entry maps one pair, so a TLB of N entries reaches N x 8 KB: with more
# pairs than entries the second pass misses again.  Run it with
# --mmu tlb:16 and --mmu tlb:32 and compare the TLB statistics.
        .eqv  PAIRS, 24

        .text
main:   li    $s0, 0x10100000       # PAIRS page pairs from here (kuseg)
        li    $s1, 2                # two passes
pass:   move  $t0, $s0
        li    $t1, PAIRS
loop:   lw    $t2, 0($t0)           # one word of the pair
        addiu $t0, $t0, 8192        # the next page pair
        addiu $t1, $t1, -1
        bnez  $t1, loop
        addiu $s1, $s1, -1
        bnez  $s1, pass
        li    $v0, 10
        syscall
