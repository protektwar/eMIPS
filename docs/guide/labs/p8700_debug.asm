# Lab 89 — Hybrid Debug: a breakpoint with no debugger (guide 5.14,
# "Hybrid Debug").  Hart 1 sends hart 0 a debug interrupt; hart 0's debug
# handler (RESET_BASE + 0x480) puts an instruction breakpoint on f and
# counts its hits; the program prints the hits, then the Debug Mode entries.
        .option norelax           # (no linker relaxation: .org 0x480 stays put)
        .data
hits:   .dword 0
entries: .dword 0
go:     .dword 0
        .text
main:   csrr t0, mhartid
        andi t0, t0, 15
        bnez t0, other
        la   t0, go
wait:   ld   t1, 0(t0)            # until the handler has set the breakpoint
        beqz t1, wait
        li   s1, 0
        li   s2, 10
loop:   call f
        addi s1, s1, 1
        blt  s1, s2, loop
        la   t0, hits
        ld   a0, 0(t0)
        li   a7, 1
        ecall
        li   a0, ' '
        li   a7, 11
        ecall
        la   t0, entries
        ld   a0, 0(t0)
        li   a7, 1
        ecall
        li   a7, 10
        ecall
f:      addi a0, a0, 1
        ret
other:  li   t1, 0x38a040         # CPC.Core[0].DBG_DBRK_REG (GCR_BASE 0x380000)
        li   t2, 1
        sd   t2, 0(t1)            # DINT bit 0: hart 0
park:   j    park

        .org 0x480                # the debug vector: RESET_BASE + 0x480
debug:  csrw dscratch0, t0
        csrw dscratch1, t1
        la   t0, entries
        ld   t1, 0(t0)
        addi t1, t1, 1
        sd   t1, 0(t0)
        csrr t1, dcsr
        srli t1, t1, 6
        andi t1, t1, 7            # dcsr.cause
        li   t0, 3
        beq  t1, t0, first        # 3: a halt request (the DINT)
        li   t0, 2
        beq  t1, t0, hit          # 2: a trigger (the breakpoint)
        li   t0, 0xffffffffff301118   # 4: the single step — IBC0 = BE again
        li   t1, 1
        sd   t1, 0(t0)
        csrr t1, dcsr
        andi t1, t1, -5           # dcsr.step = 0
        csrw dcsr, t1
        j    out
first:  li   t0, 0xffffffffff301100   # IBA0 = f, IBM0 = 0, IBC0 = BE
        la   t1, f
        sd   t1, 0(t0)
        sd   zero, 8(t0)
        li   t1, 1
        sd   t1, 0x18(t0)
        la   t0, go
        li   t1, 1
        sd   t1, 0(t0)
        j    out
hit:    la   t0, hits
        ld   t1, 0(t0)
        addi t1, t1, 1
        sd   t1, 0(t0)
        li   t0, 0xffffffffff301118   # IBC0 = 0: f may run once ...
        sd   zero, 0(t0)
        csrr t1, dcsr
        ori  t1, t1, 4            # ... a single step, then Debug Mode again
        csrw dcsr, t1
out:    csrr t0, dscratch0
        csrr t1, dscratch1
        dret
# eMIPS: --arch rv64 --core p8700 --vps 2
