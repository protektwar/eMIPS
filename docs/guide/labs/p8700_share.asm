# Lab 76 — two cores, one cache line: false sharing on the P8700.
# Run:  python3 -m emips debug docs/guide/labs/p8700_share.asm -c "run" -c "cm"
# Each core adds 1 to its own counter 100 times; core 0 prints the total.
        .eqv    APART, 8            # bytes between the two counters (try 64)
        .data
        .balign 64
c0:     .dword  0
        .space  APART - 8
c1:     .dword  0
done:   .dword  0
        .text
main:   csrr    t0, mhartid         # 0x00: core 0, 0x10: core 1
        la      s0, c0
        beqz    t0, count
        la      s0, c1
count:  li      t1, 100
loop:   ld      t2, 0(s0)
        addi    t2, t2, 1
        sd      t2, 0(s0)
        addi    t1, t1, -1
        bnez    t1, loop
        csrr    t0, mhartid
        bnez    t0, finish
wait:   la      t3, done            # core 0: wait for core 1
        ld      t4, 0(t3)
        beqz    t4, wait
        la      t3, c0
        ld      a0, 0(t3)
        la      t3, c1
        ld      t4, 0(t3)
        add     a0, a0, t4
        li      a7, 1               # prints 200
        ecall
        li      a7, 10
        ecall
finish: la      t3, done            # core 1: say so, then stop
        li      t4, 1
        sd      t4, 0(t3)
idle:   wfi
        j       idle
# eMIPS: --core p8700 --cores 2
