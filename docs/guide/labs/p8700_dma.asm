# Lab 88 — DMA and the caches (guide 5.14, "The rest of the memory system")
# Fills 4 KB with 1 ... 512, has the AXI CDMA copy it, prints the last
# doubleword of the copy.  Run it with and without --iocu.
        .data
src:    .space 4096
dst:    .space 4096
        .text
main:   la   s0, src
        li   t0, 1
        li   t1, 512
fill:   sd   t0, 0(s0)
        addi s0, s0, 8
        addi t0, t0, 1
        ble  t0, t1, fill
        li   s1, 0xffff1000       # the AXI CDMA
        la   t0, src
        sd   t0, 0x18(s1)         # SA
        la   t0, dst
        sd   t0, 0x20(s1)         # DA
        li   t0, 4096
        sw   t0, 0x28(s1)         # BTT: start
wait:   lw   t0, 4(s1)
        andi t0, t0, 2            # CDMASR.Idle
        beqz t0, wait
        la   s0, dst
        li   t0, 4088
        add  s0, s0, t0
        ld   a0, 0(s0)
        li   a7, 1
        ecall
        li      a7, 10
        ecall
# eMIPS: --arch rv64 --core p8700
