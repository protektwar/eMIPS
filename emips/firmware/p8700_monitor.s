# eMIPS's debug monitor for the P8700's Debug Unit (LF16; emips.p8700dbu).
#
# The probe loads it into the DBU's RAM (dmxseg, mirrored at dmseg); with
# the probe's ProbTrap set a hart in Debug Mode starts at dmseg + 0x200.
# Each hart has a mailbox at dmseg + 0x1000 + 0x200 x slot (slot = core x 2
# + hart, from mhartid): the probe reads and writes it through the TAP.
#
#   +0x00 state     0 running, 1 in the monitor
#   +0x08 command   0 none, 1 resume, 2 step, 3 read, 4 write (the monitor
#                   clears it when done)
#   +0x10 address   +0x18 data (read: the doubleword, write: the value)
#   +0x20 dpc       where DRET returns (the probe may change it)
#   +0x28 error     DEXCCODE + 1 of a command that faulted, else 0
#   +0x30 entries   how many times the hart entered the monitor
#   +0x108 ...      x1 - x31 as the hart left them (the probe may change them)
#
# Position independent: assembled anywhere, copied to the RAM's start.
# An exception in Debug Mode re-enters the vector (MD00047 2.4.3): with the
# state already 1 that is a faulting command — reported, not a new entry.

        .equ mipsdebug, 0x7cf
        .text
        .org 0x200
entry:  csrw dscratch0, t0               # (two scratch registers: enough for the slot)
        csrw dscratch1, t1
        csrr t0, mhartid
        srli t1, t0, 4
        andi t1, t1, 0xFF                # (the core)
        slli t1, t1, 1
        andi t0, t0, 0xF                 # (the hart)
        add  t1, t1, t0
        slli t1, t1, 9
        lla  t0, mailboxes
        add  t0, t0, t1
        ld   t1, 0x00(t0)
        bnez t1, fault                   # (state 1: a command faulted)
        csrr t1, dscratch1
        sd   t1, 0x130(t0)               # x6  (t1)
        csrr t1, dscratch0
        sd   t1, 0x128(t0)               # x5  (t0)
        sd   x1, 0x108(t0)
        sd   x2, 0x110(t0)
        sd   x3, 0x118(t0)
        sd   x4, 0x120(t0)
        sd   x7, 0x138(t0)
        sd   x8, 0x140(t0)
        sd   x9, 0x148(t0)
        sd   x10, 0x150(t0)
        sd   x11, 0x158(t0)
        sd   x12, 0x160(t0)
        sd   x13, 0x168(t0)
        sd   x14, 0x170(t0)
        sd   x15, 0x178(t0)
        sd   x16, 0x180(t0)
        sd   x17, 0x188(t0)
        sd   x18, 0x190(t0)
        sd   x19, 0x198(t0)
        sd   x20, 0x1A0(t0)
        sd   x21, 0x1A8(t0)
        sd   x22, 0x1B0(t0)
        sd   x23, 0x1B8(t0)
        sd   x24, 0x1C0(t0)
        sd   x25, 0x1C8(t0)
        sd   x26, 0x1D0(t0)
        sd   x27, 0x1D8(t0)
        sd   x28, 0x1E0(t0)
        sd   x29, 0x1E8(t0)
        sd   x30, 0x1F0(t0)
        sd   x31, 0x1F8(t0)
        csrr t1, dpc
        sd   t1, 0x20(t0)
        ld   t1, 0x30(t0)
        addi t1, t1, 1
        sd   t1, 0x30(t0)
        sd   zero, 0x28(t0)
        li   t1, 1
        sd   t1, 0x00(t0)
        j    poll

fault:  csrr t1, mipsdebug               # DEXCCODE + 1 into the error word
        srli t1, t1, 10
        andi t1, t1, 0x3F
        addi t1, t1, 1
        sd   t1, 0x28(t0)
        sd   zero, 0x08(t0)              # (the command is over)

poll:   ld   t1, 0x08(t0)
        beqz t1, poll
        li   t2, 3
        beq  t1, t2, read
        li   t2, 4
        beq  t1, t2, write
        csrr t2, dcsr                    # 1 resume / 2 step: dcsr.step
        andi t2, t2, -5
        li   t3, 2
        bne  t1, t3, 1f
        ori  t2, t2, 4
1:      csrw dcsr, t2
        ld   t1, 0x20(t0)
        csrw dpc, t1
        sd   zero, 0x08(t0)
        sd   zero, 0x00(t0)
        ld   x1, 0x108(t0)
        ld   x2, 0x110(t0)
        ld   x3, 0x118(t0)
        ld   x4, 0x120(t0)
        ld   x7, 0x138(t0)
        ld   x8, 0x140(t0)
        ld   x9, 0x148(t0)
        ld   x10, 0x150(t0)
        ld   x11, 0x158(t0)
        ld   x12, 0x160(t0)
        ld   x13, 0x168(t0)
        ld   x14, 0x170(t0)
        ld   x15, 0x178(t0)
        ld   x16, 0x180(t0)
        ld   x17, 0x188(t0)
        ld   x18, 0x190(t0)
        ld   x19, 0x198(t0)
        ld   x20, 0x1A0(t0)
        ld   x21, 0x1A8(t0)
        ld   x22, 0x1B0(t0)
        ld   x23, 0x1B8(t0)
        ld   x24, 0x1C0(t0)
        ld   x25, 0x1C8(t0)
        ld   x26, 0x1D0(t0)
        ld   x27, 0x1D8(t0)
        ld   x28, 0x1E0(t0)
        ld   x29, 0x1E8(t0)
        ld   x30, 0x1F0(t0)
        ld   x31, 0x1F8(t0)
        ld   t1, 0x130(t0)
        ld   t0, 0x128(t0)
        dret

read:   ld   t2, 0x10(t0)
        ld   t2, 0(t2)
        sd   t2, 0x18(t0)
        sd   zero, 0x08(t0)
        j    poll
write:  ld   t2, 0x10(t0)
        ld   t3, 0x18(t0)
        sd   t3, 0(t2)
        sd   zero, 0x08(t0)
        j    poll

        .org 0x1000
mailboxes:
