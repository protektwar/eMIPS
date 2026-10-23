# RISC-V traps: an M-mode trap handler (mtvec), an illegal instruction it
# skips (mepc + 4), and a software interrupt (mip.SSIP, mie.SSIE,
# mstatus.MIE) it clears — mcause tells them apart (bit XLEN-1: interrupt).
        .text
main:   la   t0, handler
        csrw mtvec, t0          # direct mode: every trap to handler
        .option push
        .option norvc           # (the 32-bit unimp: csrrw zero, cycle, zero —
        unimp                   # a write to a read-only CSR, an illegal instruction)
        .option pop
        csrsi mie, 2            # SSIE
        csrsi mstatus, 8        # MIE
        csrsi mip, 2            # SSIP: pending and enabled -> taken now
        li   a0, 0
        mv   a0, s1             # how many traps the handler saw
        li   a7, 1
        ecall
        li   a7, 10
        ecall

handler:
        addi s1, s1, 1
        csrr t1, mcause
        bltz t1, 1f             # an interrupt
        csrr t2, mepc           # an exception: skip the instruction
        addi t2, t2, 4
        csrw mepc, t2
        mret
1:      csrci mip, 2            # clear SSIP
        mret
# eMIPS: --arch rv32
