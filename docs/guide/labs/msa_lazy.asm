# MSA on first use: main clears Config5.MSAEn, as an operating system does
# for a new process, then uses MSA.  The first MSA instruction raises MSA
# Disabled (ExcCode 21); the handler enables MSA and returns.  As handed
# out the program prints 7, not 5 + 7 = 12: find the handler's bug and
# fix it (python3 -m emips check docs/guide/labs/msa_lazy.asm).
        .text
main:   mfc0    $t0, $16, 5         # Config5
        lui     $t1, 0x0800         # bit 27: MSAEn
        nor     $t1, $t1, $zero
        and     $t0, $t0, $t1
        mtc0    $t0, $16, 5         # MSA off: the next MSA instruction traps
        ehb
        ldi.w   $w1, 5              # 5 in every word (MSA Disabled here)
        ldi.w   $w2, 7
        addv.w  $w3, $w1, $w2
        copy_s.w $a0, $w3[0]
        li      $v0, 1              # print element 0
        syscall
        li      $v0, 10
        syscall

        .ktext  0x80000180          # the general exception vector
handler:
        mfc0    $k0, $13            # Cause
        srl     $k0, $k0, 2
        andi    $k0, $k0, 0x1f      # ExcCode
        li      $k1, 21             # MSA Disabled?
        bne     $k0, $k1, other
        mfc0    $k0, $16, 5         # yes: enable MSA (Config5.MSAEn = 1)
        lui     $k1, 0x0800
        or      $k0, $k0, $k1
        mtc0    $k0, $16, 5
        mfc0    $k0, $14            # EPC ...
        addiu   $k0, $k0, 4         # ... + 4, as after a system call
        mtc0    $k0, $14
        ehb
        eret
other:  li      $v0, 10             # anything else: stop
        syscall
# eMIPS: --release R5
