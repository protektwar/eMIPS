# Lab: measure with the performance counters (CP0 register 25).
# Counter 0 counts completed instructions, counter 1 load misses in the
# data cache, both in kernel mode (K): the program runs in kernel mode.
# Run it without caches, then with --dcache 1K:1:16 and --dcache 4K:1:16.
        .eqv  N, 512                 # 512 words = 2 KB
        .data
nums:   .space 4 * N
        .text
main:   mtc0  $zero, $25, 1          # PerfCnt0 = 0
        mtc0  $zero, $25, 3          # PerfCnt1 = 0
        li    $t0, (1 << 5) | 0x2    # PerfCtl0: event 1 (instructions), K
        mtc0  $t0, $25, 0
        li    $t0, (13 << 5) | 0x2   # PerfCtl1: event 13 (load misses), K
        mtc0  $t0, $25, 2
        li    $t2, 2                 # sum the array twice
pass:   la    $t0, nums
        li    $t1, N
loop:   lw    $t3, 0($t0)
        addu  $v1, $v1, $t3
        addiu $t0, $t0, 4
        addiu $t1, $t1, -1
        bnez  $t1, loop
        addiu $t2, $t2, -1
        bnez  $t2, pass
        mtc0  $zero, $25, 0          # stop both counters
        mtc0  $zero, $25, 2
        mfc0  $a0, $25, 1            # instructions
        li    $v0, 1
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        mfc0  $a0, $25, 3            # load misses
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall
