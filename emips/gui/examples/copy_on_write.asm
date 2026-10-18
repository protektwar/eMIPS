# Copy-on-write with the TLB Modified exception.
# Needs a TLB MMU (the example switches Settings > Processor > MMU to
# "TLB, 32 entries"); watch the TLB pane while stepping.
#
# Two virtual pages, the parent's (0x00600000) and the child's
# (0x00700000), share one physical frame, both read-only (D = 0). The
# first store to the child's page takes a TLB Modified exception: the
# kernel copies the frame, maps the child's page to the copy with D = 1,
# rewrites the TLB entry (TLBP, TLBWI) and returns; the store is retried
# and lands in the child's own copy.  The parent still sees the old value.

        .eqv  PT, 0x90000000        # the page table (16 bytes per page pair)
        .eqv  PARENT, 0x00600000
        .eqv  CHILD, 0x00700000

        .ktext 0x80000000           # TLB Refill: load the pair from the table
refill: mfc0  $k1, $4               # Context: PTEBase | BadVPN2 << 4
        lw    $k0, 0($k1)
        lw    $k1, 4($k1)
        mtc0  $k0, $2               # EntryLo0
        mtc0  $k1, $3               # EntryLo1
        ehb
        tlbwr
        eret

        .ktext 0x80000180           # the other exceptions
general:
        mfc0  $k0, $13              # Cause
        andi  $k0, $k0, 0x7c        # ExcCode << 2
        li    $k1, 4                # ExcCode 1 = TLB Modified
        bne   $k0, $k1, fatal
        la    $k0, save
        sw    $t0, 0($k0)
        sw    $t1, 4($k0)
        sw    $t2, 8($k0)
        sw    $t3, 12($k0)
        mfc0  $t0, $4               # Context: the page pair's entry
        mfc0  $t1, $8               # BadVAddr: bit 12 = even / odd page
        srl   $t1, $t1, 10
        andi  $t1, $t1, 4
        addu  $t0, $t0, $t1         # the page's EntryLo in the table
        lw    $t2, 0($t0)
        srl   $t2, $t2, 6           # PFN
        sll   $t2, $t2, 12          # the shared frame's PA ...
        lui   $t1, 0x8000
        or    $t2, $t2, $t1         # ... as a kseg0 address (PA < 512 MB)
        la    $k0, free
        lw    $t3, 0($k0)           # a free frame (kseg0 address)
        addiu $t1, $t3, 4096
        sw    $t1, 0($k0)
        move  $k0, $t3
        addiu $k1, $t2, 4096
copy:   lw    $t1, 0($t2)           # copy the 4 KB page
        sw    $t1, 0($k0)
        addiu $t2, $t2, 4
        addiu $k0, $k0, 4
        bne   $t2, $k1, copy
        sll   $t3, $t3, 3           # the copy's PA (drop the kseg0 bits) ...
        srl   $t3, $t3, 9
        sll   $t3, $t3, 6           # ... as EntryLo.PFN
        ori   $t3, $t3, 0x1f        # C = 3, D = 1 (writable), V, G
        sw    $t3, 0($t0)           # the page table: the child's own frame
        tlbp                        # Index = the entry of EntryHi (the page)
        ehb
        mfc0  $t0, $4
        lw    $t1, 0($t0)
        mtc0  $t1, $2
        lw    $t1, 4($t0)
        mtc0  $t1, $3
        ehb
        tlbwi                       # rewrite that entry
        la    $k0, save
        lw    $t0, 0($k0)
        lw    $t1, 4($k0)
        lw    $t2, 8($k0)
        lw    $t3, 12($k0)
        eret                        # retry the store
fatal:  li    $v0, 10               # any other exception: stop
        syscall

        .kdata 0x90002000           # VPN2 0x00200: the code
        .word 0x0101001f, 0x0101005f
        .kdata 0x90003000           # VPN2 0x00300: the parent, read-only
        .word 0x0040401b, 0x00000001   # -> the shared frame (D = 0); odd: invalid
        .kdata 0x90003800           # VPN2 0x00380: the child, read-only too
        .word 0x0040401b, 0x00000001
        .kdata 0x90100000           # the shared frame (PA 0x10100000)
shared: .word 1
        .kdata 0x90200000           # kernel variables
free:   .word 0x90101000            # the next free frame
save:   .space 16

        .ktext 0x80001000           # the kernel starts here
main:   li    $t0, PT
        mtc0  $t0, $4               # Context.PTEBase = the page table
        la    $t0, user
        jr    $t0

        .text                       # VA 0x00400000
user:   li    $s0, PARENT
        li    $s1, CHILD
        lw    $a0, 0($s1)           # the child reads the shared page: 1
        jal   print
        li    $t0, 2
        sw    $t0, 0($s1)           # the child writes: TLB Modified -> copy
        lw    $a0, 0($s0)           # the parent still reads 1
        jal   print
        lw    $a0, 0($s1)           # the child reads its copy: 2
        jal   print
        li    $v0, 10
        syscall

print:  li    $v0, 1                # print $a0 and a space
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        jr    $ra
# eMIPS: --mmu tlb    (the settings this program runs with)
