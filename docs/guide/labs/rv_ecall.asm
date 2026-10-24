# Lab 74 — a service of your own: an M-mode handler for a U-mode ecall.
# Run:  python3 -m emips debug docs/guide/labs/rv_ecall.asm
# It should print 42 and stop — it never does.  Find out why and fix it.
        .text
main:   la      t0, handler
        csrw    mtvec, t0
        li      t0, 0x1800
        csrc    mstatus, t0         # MPP = U
        la      t0, user
        csrw    mepc, t0
        mret                        # into U-mode

user:   li      a0, 21
        li      a7, 100             # service 100: not RARS's — the handler's
        ecall                       # (environment call from U-mode: cause 8)
        li      a7, 1               # print_int: RARS's
        ecall
        li      a7, 10
        ecall

        .balign 4
handler:
        csrr    t0, mcause
        li      t1, 8
        bne     t0, t1, fatal       # only U-mode ecalls are expected
        li      t1, 100
        bne     a7, t1, fatal
        add     a0, a0, a0          # service 100 doubles a0
        mret
fatal:  li      a7, 10
        ecall
# eMIPS: --arch rv32
