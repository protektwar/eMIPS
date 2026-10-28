# Lab 83 — Interrupts as memory writes (guide 5.13, "The AIA").
# Two harts, each with an IMSIC.  Hart 1 waits for interrupts and writes
# down every identity it claims; hart 0 sends it three by storing to hart
# 1's interrupt file, then makes the APLIC send one (genmsi), then points
# the APLIC's MSI address somewhere else and sends another.
        .data
ready:  .word 0                     # hart 1 is set up
count:  .word 0                     # identities hart 1 has claimed
took:   .word 0, 0, 0, 0, 0, 0, 0, 0
msg1:   .asciz "hart 1 took:"
msg2:   .asciz "\nword at 0x80101000: "
        .text
main:   csrr    t0, mhartid
        bnez    t0, hart1

# ---------------------------------------------------------------- hart 0
        la      s1, ready
1:      lw      t0, 0(s1)           # wait for hart 1
        beqz    t0, 1b
        li      s0, 0x24001000      # hart 1's M file (seteipnum_le)
        li      t0, 7
        sw      t0, 0(s0)
        li      t0, 3
        sw      t0, 0(s0)
        li      t0, 5
        sw      t0, 0(s0)
        li      a0, 3
        jal     wait_for
        li      s2, 0x0c000000      # the APLIC's machine-level domain
        li      t0, 0x100           # domaincfg: IE (DM reads 1: MSI mode)
        sw      t0, 0(s2)
        li      t0, 0x3000          # genmsi: hart index 1, EIID 9
        add     t0, s2, t0
        li      t1, 1 << 18 | 9
        sw      t1, 0(t0)
        li      a0, 4
        jal     wait_for
        li      t0, 0x1bc0          # mmsiaddrcfg: the MSI base page
        add     t0, s2, t0
        li      t1, 0x80100         # PPN: 0x80100000
        sw      t1, 0(t0)
        li      t0, 0x3000
        add     t0, s2, t0
        li      t1, 1 << 18 | 9     # the same MSI again
        sw      t1, 0(t0)
        la      a0, msg1            # print what hart 1 took
        li      a7, 4
        ecall
        la      s3, took
        la      t0, count
        lw      s4, 0(t0)
2:      beqz    s4, 3f
        li      a0, ' '
        li      a7, 11
        ecall
        lw      a0, 0(s3)
        li      a7, 1
        ecall
        addi    s3, s3, 4
        addi    s4, s4, -1
        j       2b
3:      la      a0, msg2
        li      a7, 4
        ecall
        li      t0, 0x80101000
        lw      a0, 0(t0)
        li      a7, 1
        ecall
        li      a7, 10
        ecall

wait_for:                           # until hart 1 has claimed a0 identities
        la      t0, count
1:      lw      t1, 0(t0)
        blt     t1, a0, 1b
        ret

# ---------------------------------------------------------------- hart 1
hart1:  li      t0, 0x70            # eidelivery = 1
        csrw    miselect, t0
        li      t0, 1
        csrw    mireg, t0
        li      t0, 0xc0            # eie0: identities 1-63
        csrw    miselect, t0
        li      t0, -2
        csrw    mireg, t0
        la      t0, handler
        csrw    mtvec, t0
        li      t0, 0x800           # MEIE
        csrs    mie, t0
        csrsi   mstatus, 8          # MIE
        la      t0, ready
        li      t1, 1
        sw      t1, 0(t0)
4:      wfi
        j       4b

        .align  2                   # (mtvec: a 4-byte aligned handler)
handler:
        csrrw   t0, mtopei, zero    # claim the top identity
        srli    t0, t0, 16
        la      t1, count
        lw      t2, 0(t1)
        slli    t3, t2, 2
        la      t4, took
        add     t4, t4, t3
        sw      t0, 0(t4)           # took[count] = identity
        addi    t2, t2, 1
        sw      t2, 0(t1)
        mret
# eMIPS: --arch rv64 --rv-intc imsic --cores 2
