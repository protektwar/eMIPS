# Lab 75 — pages: an Sv32 page table built by M-mode, used by S-mode.
# Run:  python3 -m emips run docs/guide/labs/rv_pages.asm
# The S-mode code reads two words through the 4 KB pages at 0x80000000 and
# 0x80001000.  The second read faults: map that page too.
        .data
        .balign 4096
root:   .space  4096                # the first-level table: 1024 PTEs
leaf:   .space  4096                # a second-level table
page0:  .word   1111                # what VA 0x80000000 shows
        .balign 4096
page1:  .word   2222                # what VA 0x80001000 should show
        .text
main:   la      t0, handler
        csrw    mtvec, t0
        la      s0, root
        # megapages (4 MB, level 1 leaves): the code and the data, VA = PA
        li      t0, 0x00400000 >> 2 | 0xcb      # PPN | D A X R V (0xcb)
        sw      t0, 1 * 4(s0)                   # root[0x00400000 >> 22]
        li      t0, 0x10000000 >> 2 | 0xc7      # PPN | D A W R V
        sw      t0, 0x40 * 4(s0)                # root[0x10000000 >> 22]
        # VA 0x80000000: root[0x200] points to the leaf table
        la      t1, leaf
        srli    t0, t1, 2
        ori     t0, t0, 0x01                    # V only: a pointer, not a leaf
        li      t2, 0x200 * 4
        add     t2, s0, t2
        sw      t0, 0(t2)
        la      t2, page0                       # leaf[0] -> page0
        srli    t0, t2, 2
        ori     t0, t0, 0xc7
        sw      t0, 0(t1)
        # turn paging on for S-mode and go there
        srli    t0, s0, 12
        li      t1, 1 << 31                     # MODE = Sv32
        or      t0, t0, t1
        csrw    satp, t0
        sfence.vma
        li      t0, 0x1800
        csrc    mstatus, t0
        li      t0, 0x0800
        csrs    mstatus, t0                     # MPP = S
        la      t0, smode
        csrw    mepc, t0
        mret

smode:  li      s1, 0x80000000
        lw      a0, 0(s1)                       # through leaf[0]
        li      a7, 1
        ecall
        li      a0, ' '
        li      a7, 11
        ecall
        li      s2, 0x80001000
        lw      a0, 0(s2)                       # through leaf[1]
        li      a7, 1
        ecall
        li      a7, 10
        ecall

        .balign 4
handler:
        csrr    a0, mcause                      # 13: load page fault
        li      a7, 1
        ecall
        li      a0, ' '
        li      a7, 11
        ecall
        csrr    a0, mtval                       # the address that faulted
        li      a7, 34
        ecall
        li      a7, 10
        ecall
# eMIPS: --arch rv32
