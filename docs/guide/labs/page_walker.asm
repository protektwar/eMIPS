# The hardware page table walker (run with --mmu tlb+pw): the kernel
# builds a two-level page table that maps the first page of .data onto
# the frame of its second page, turns the walker on and loads `first`.
# With the walker the TLB is refilled from the table (prints 2); without
# one, the boot firmware's refill handler maps the page as usual (prints 1).
        .data
first:  .word 1
        .space 4092
second: .word 2
        .ktext 0x80001000
main:   li    $t0, 0x80100000           # the directory (kseg0)
        mtc0  $t0, $5, 5                # PWBase
        li    $t1, (22 << 12) | (12 << 6) | 6
        mtc0  $t1, $5, 6                # PWField: MDI 22, PTI 12, PTEI 6
        li    $t1, (10 << 12) | (10 << 6)
        mtc0  $t1, $5, 7                # PWSize: MDW 10, PTW 10 (4-byte PTEs)
        li    $t1, 0x80200000           # a page table
        sw    $t1, 0x100($t0)           # directory[0x10010000 >> 22] = it
        li    $t2, 0x500117c0           # PTE: frame 0x50011000, C 3, D, V, G
        sw    $t2, 0x40($t1)            # page 0x10010000 (PT index 0x10)
        sw    $t2, 0x44($t1)            # page 0x10011000 (PT index 0x11)
        li    $t3, 1 << 31
        mtc0  $t3, $6, 6                # PWCtl.PWEn: walker on
        ehb
        lw    $a0, first
        li    $v0, 1                    # print_int
        syscall
        li    $v0, 10
        syscall
