# Lab 84 — Control-flow integrity: a ROP return and a JOP jump (guide 5.13).
# M-mode builds an Sv39 page table — the program and its data mapped
# one-to-one by two 1 GB pages, and one 4 KB shadow-stack page (xwr = 010)
# at 0xC0000000 — turns on the shadow stack (menvcfg.SSE) and landing pads
# (menvcfg.LPE) for S-mode, and runs `smain` in S-mode.  `victim` saves ra
# on the stack and on the shadow stack; with a0 = 1 a "buffer overflow"
# overwrites the saved ra with the address of `gadget`.
        .data
        .align  12
root:   .space  4096                # the page tables (4 KB aligned)
l1:     .space  4096
l0:     .space  4096
shadow: .space  4096                # the shadow-stack page's frame
msg1:   .asciz  "victim returned normally\n"
msg2:   .asciz  "gadget ran: the attacker won\n"
msg3:   .asciz  "trap: mcause "
msg4:   .asciz  ", mtval "
        .text
main:   la      t0, trap
        csrw    mtvec, t0
        la      t0, root
        li      t1, 0xcf            # V R W X A D: a 1 GB page at PA 0
        sd      t1, 0(t0)
        li      t1, 0x40000 << 10 | 0xcf    # and at PA 0x40000000 (the stack)
        sd      t1, 8(t0)
        la      t1, l1              # VA 0xC0000000: a 4 KB page through l1 / l0
        srli    t1, t1, 12
        slli    t1, t1, 10
        ori     t1, t1, 1
        sd      t1, 24(t0)
        la      t2, l0
        srli    t1, t2, 12
        slli    t1, t1, 10
        ori     t1, t1, 1
        la      t3, l1
        sd      t1, 0(t3)
        la      t1, shadow          # the shadow-stack page: xwr = 010 (V W A D)
        srli    t1, t1, 12
        slli    t1, t1, 10
        ori     t1, t1, 0xc5
        sd      t1, 0(t2)
        srli    t0, t0, 12          # satp: Sv39, the root table
        li      t1, 8
        slli    t1, t1, 60
        or      t0, t0, t1
        csrw    satp, t0
        sfence.vma
        li      t0, 0xc             # menvcfg: LPE (landing pads) and SSE (shadow stack)
        csrs    menvcfg, t0
        li      t0, 0xc0001000      # ssp: the top of the shadow-stack page
        csrw    ssp, t0
        li      t0, 0x1800          # MPP = S
        csrc    mstatus, t0
        li      t0, 0x0800
        csrs    mstatus, t0
        la      t0, smain
        csrw    mepc, t0
        mret

        .balign 4
smain:  li      a0, 0
        call    victim              # (1) an honest call
        li      a0, 1
        call    victim              # (2) the "buffer overflow"
        li      a7, 10
        ecall

        .balign 4
victim: addi    sp, sp, -16
        sd      ra, 8(sp)           # the return address on the stack ...
        sspush  ra                  # ... and on the shadow stack
        beqz    a0, 1f
        la      t0, gadget          # the overflow overwrites the saved ra
        sd      t0, 8(sp)
1:      la      a0, msg1
        li      a7, 4
        ecall
        ld      ra, 8(sp)
        addi    sp, sp, 16
        sspopchk ra                 # ra must equal the shadow copy
        ret

        .balign 4
gadget: la      a0, msg2
        li      a7, 4
        ecall
        li      a7, 10
        ecall

        .balign 4
trap:   la      a0, msg3
        li      a7, 4
        ecall
        csrr    a0, mcause
        li      a7, 1
        ecall
        la      a0, msg4
        li      a7, 4
        ecall
        csrr    a0, mtval
        li      a7, 1
        ecall
        li      a0, '\n'
        li      a7, 11
        ecall
        li      a7, 10
        ecall
# eMIPS: --arch rv64 --rv-ext imafdc_zicsr_zifencei_zicntr_zicfilp_zicfiss_zcmop
