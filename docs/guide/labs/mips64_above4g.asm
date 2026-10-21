# Lab 61 — Above 4 GB (Phase 5, 5.12)
# A kernel program (it runs in kseg0) stores two values in xuseg, 4 GB
# apart: 1 at 0x100000000, 2 at 0x200000000.  Its XTLB refill handler maps
# them through a page table, as a 64-bit operating system does: XContext
# holds the address of the missing page pair's two entries.  It should
# print "1 2 2": the two values, then how many refills there were.
        .kdata
refills: .word  0                       # how many times xrefill ran
        .ktext  0x80000080              # the XTLB Refill vector (EBase + 0x080)
xrefill:
        mfc0    $k1, $20                # XContext: where the missing pair's entries are
        ld      $k0, 0($k1)             # EntryLo0: the even page
        ld      $k1, 8($k1)             # EntryLo1: the odd page
        dmtc0   $k0, $2
        dmtc0   $k1, $3
        ehb
        tlbwr                           # into a random TLB entry
        la      $k0, refills
        lw      $k1, 0($k0)
        addiu   $k1, $k1, 1
        sw      $k1, 0($k0)
        eret

        .ktext  0x80000180              # any other exception: say which, and stop
        mfc0    $a0, $13                # Cause
        srl     $a0, $a0, 2
        andi    $a0, $a0, 0x1f          # ExcCode
        li      $v0, 1
        syscall
        li      $a0, ' '
        li      $v0, 11
        syscall
        dmfc0   $a0, $8                 # BadVAddr
        li      $v0, 34
        syscall
        li      $v0, 10
        syscall

        .ktext  0x80001000              # the program: kernel code in kseg0
main:   dli     $t0, 0x9800000000000000 # the page table: in xkphys (cached), from PA 0
        dmtc0   $t0, $20                # XContext.PTEBase = its address, bits 63:33
        # one entry pair (16 bytes) for each page pair, at PTEBase + VPN2 x 16:
        dli     $t1, 0x9800000000800000 # VA 4 GB: VPN2 = 0x80000
        li      $t2, (0x300 << 6) | 0x1f    # even page -> PA 0x300000 (C 3, D, V, G)
        sd      $t2, 0($t1)
        li      $t2, (0x301 << 6) | 0x1f    # odd page -> PA 0x301000
        sd      $t2, 8($t1)
        dli     $t1, 0x9800000001000000 # VA 8 GB: VPN2 = 0x100000
        li      $t2, (0x302 << 6) | 0x1f    # even page -> PA 0x302000
        sd      $t2, 0($t1)
        li      $t2, (0x303 << 6) | 0x1f    # odd page -> PA 0x303000
        sd      $t2, 8($t1)

        dli     $t0, 0x100000000        # 4 GB (xuseg)
        dli     $t1, 0x200000000        # 8 GB
        li      $t2, 1
        sw      $t2, 0($t0)
        li      $t2, 2
        sw      $t2, 0($t1)
        lw      $a0, 0($t0)
        li      $v0, 1
        syscall
        li      $a0, ' '
        li      $v0, 11
        syscall
        lw      $a0, 0($t1)
        li      $v0, 1
        syscall
        li      $a0, ' '
        li      $v0, 11
        syscall
        la      $t0, refills
        lw      $a0, 0($t0)
        li      $v0, 1
        syscall
        li      $v0, 10
        syscall
# eMIPS: --arch mips64 --mmu tlb
