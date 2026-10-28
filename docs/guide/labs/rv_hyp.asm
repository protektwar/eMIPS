# Lab 81: a hypervisor in M-mode runs a guest in VS-mode.  It builds the
# G-stage page table (Sv39x4: guest physical -> physical): the program's code
# and data where they are, and guest page 0x20000000 on a page of its own.
# The guest reads that page, then a guest physical address nothing maps.
        .data
        .align 14
root:   .space 16384                # the G-stage root table: 2048 entries
l1:     .space 4096                 # GPAs 0 - 1 GB, 2 MB each
l0:     .space 4096                 # GPAs 0x20000000 - 0x201fffff, 4 KB each
page:   .dword 42                   # what the guest sees at GPA 0x20000000
        .space 4088
reads:  .asciz "guest reads "
fault:  .asciz "\nguest-page fault: mcause "
gpa:    .asciz ", GPA "
        .text
main:   li      t2, 0xdf            # a leaf: V R W X U A D (every G-stage page is U)
        la      t0, root
        la      t1, l1              # root[0] -> l1 (a pointer: V only)
        srli    t1, t1, 12
        slli    t1, t1, 10
        ori     t1, t1, 1
        sd      t1, 0(t0)
        la      t0, l1
        li      t1, 0x100000        # l1[2]: GPA 0x400000 (the code) -> PA 0x400000
        or      t1, t1, t2
        sd      t1, 2*8(t0)
        li      t1, 0x4000000       # l1[0x80]: GPA 0x10000000 (the data) -> PA 0x10000000
        or      t1, t1, t2
        sd      t1, 0x80*8(t0)
        la      t1, l0              # l1[0x100]: GPA 0x20000000 -> l0
        srli    t1, t1, 12
        slli    t1, t1, 10
        ori     t1, t1, 1
        li      t3, 0x100*8         # (beyond a 12-bit offset)
        add     t3, t0, t3
        sd      t1, 0(t3)
        la      t0, l0
        la      t1, page            # l0[0]: GPA 0x20000000 -> page
        srli    t1, t1, 12
        slli    t1, t1, 10
        or      t1, t1, t2
        sd      t1, 0(t0)
        la      t0, root            # hgatp = Sv39x4 (8), VMID 1, the root's PPN
        srli    t0, t0, 12
        li      t1, 8 << 60 | 1 << 44
        or      t0, t0, t1
        csrw    hgatp, t0
        hfence.gvma                 # (forget older G-stage translations)
        la      t0, hyp
        csrw    mtvec, t0           # the guest's traps come here (none delegated)
        li      t0, 3 << 11
        csrc    mstatus, t0
        li      t0, 1 << 11 | 1 << 39
        csrs    mstatus, t0         # MPP = S, MPV = 1: mret enters VS-mode
        la      t0, guest
        csrw    mepc, t0
        mret
guest:  la      a0, reads           # (VS-mode: every access translates twice —
        li      a7, 4               #  vsatp is Bare, so GVA = GPA)
        ecall
        li      t0, 0x20000000
        ld      a0, 0(t0)           # the hypervisor's page
        li      a7, 1
        ecall
        li      t0, 0x30000000
        ld      a0, 0(t0)           # nothing maps it: a guest-page fault
        li      a7, 10
        ecall
        .align 2
hyp:    la      a0, fault
        li      a7, 4
        ecall
        csrr    a0, mcause          # 21: load guest-page fault
        li      a7, 1
        ecall
        la      a0, gpa
        li      a7, 4
        ecall
        csrr    a0, mtval2          # the GPA >> 2 (mtval: the guest's VA)
        slli    a0, a0, 2
        li      a7, 34
        ecall
        li      a7, 10
        ecall
# eMIPS: --arch rv64 --rv-ext imafdch_zicsr_zifencei_zicntr
