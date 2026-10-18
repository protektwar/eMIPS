# The TLB refill handler of a small kernel: it reads the page table.
# Needs a TLB MMU (the example switches Settings > Processor > MMU to
# "TLB, 32 entries"); watch the TLB pane while stepping.
#
# Context (CP0 4) = PTEBase | BadVPN2 << 4: after a TLB miss it points at
# the 16-byte page table entry of the missing page pair (Vol. III 9.9);
# here an entry holds EntryLo0 (even page) and EntryLo1 (odd page).

        .eqv  PT, 0x90000000        # the page table: kseg0, PA 0x10000000

        .ktext 0x80000000           # TLB Refill vector (EBase + 0x000)
refill: mfc0  $k1, $4               # Context: the entry of the missing pair
        lw    $k0, 0($k1)           # EntryLo0 (even page)
        lw    $k1, 4($k1)           # EntryLo1 (odd page)
        mtc0  $k0, $2
        mtc0  $k1, $3
        ehb
        tlbwr                       # into a random entry (Wired .. 31)
        eret                        # retry the access

        # Page table entries (EntryLo: PFN << 6 | C << 3 | D V G): PFN =
        # PA >> 12; C = 3 cacheable; D = writable, V = valid, G = global.
        .kdata 0x90002000           # VPN2 0x00200: VA 0x00400000-0x00401fff
        .word 0x0101001f, 0x0101005f   # -> PA 0x40400000, 0x40401000 (the code)
        .kdata 0x90003000           # VPN2 0x00300: VA 0x00600000-0x00601fff
        .word 0x0140041f, 0x0140045f   # -> PA 0x50010000: the data, a 2nd time
        .kdata 0x90080080           # VPN2 0x08008: VA 0x10010000-0x10011fff
        .word 0x0140041f, 0x0140045f   # -> PA 0x50010000, 0x50011000 (the data)

        .ktext 0x80001000           # the kernel starts here
main:   li    $t0, PT
        mtc0  $t0, $4               # Context.PTEBase = the page table
        la    $t0, user
        jr    $t0                   # run the program (kuseg: mapped)

        .data                       # VA 0x10010000
value:  .word 42

        .text                       # VA 0x00400000
user:   lw    $a0, value            # a TLB miss: the handler maps the data
        li    $v0, 1
        syscall                     # print 42
        li    $a0, ' '
        li    $v0, 11
        syscall
        la    $t0, value
        andi  $t0, $t0, 0x1fff      # its offset in the page pair ...
        lui   $t1, 0x0060
        addu  $t0, $t0, $t1         # ... at VA 0x00600000: the same frame
        li    $t2, 99
        sw    $t2, 0($t0)           # write through the second mapping
        lw    $a0, value            # and read through the first: 99
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall
# eMIPS: --mmu tlb    (the settings this program runs with)
