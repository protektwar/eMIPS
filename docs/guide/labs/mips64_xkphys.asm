# Lab 60 — Memory without the TLB: xkphys (Phase 5, 5.12)
# A kernel program stores a word through one xkphys address and reads it
# back through another one of the same physical address (0x00201000):
# attribute 2 (uncached) and attribute 3 (cached, write-back).
        .ktext  0x80000180              # an exception: say which, and stop
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

        .text
main:   dli     $t0, 0x9800000000201000 # xkphys, attribute 3: cached
        dli     $t1, 0x9000000000201000 # xkphys, attribute 2: uncached
        li      $t2, 0x1234
        sw      $t2, 0($t0)             # store through the cache
        lw      $a0, 0($t1)             # read the memory itself
        jal     hex
        lw      $a0, 0($t0)             # read through the cache
        jal     hex
        li      $v0, 10
        syscall

hex:    li      $v0, 34                 # print $a0 in hex, then a newline
        syscall
        li      $a0, '\n'
        li      $v0, 11
        syscall
        jr      $ra
# eMIPS: --arch mips64 --mmu tlb --caches
