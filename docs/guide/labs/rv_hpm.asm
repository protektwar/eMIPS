# Lab 82: counting with the hardware performance monitor (Zihpm).  The
# program sums a 32 x 32 matrix of words twice — row by row, then column by
# column — and has mhpmcounter3 / 4 count the L1 D-cache's read accesses and
# read misses (SBI PMU events 0x10000 / 0x10001 in mhpmevent3 / 4).
        .data
        .align 6
matrix: .space 4096                 # 32 rows of 32 words (128 bytes a row)
rows:   .asciz "row by row:    "
cols:   .asciz "column by column: "
loads:  .asciz "loads "
misses: .asciz ", misses "
        .text
main:   li      t0, 0x10000         # L1D read access
        csrw    mhpmevent3, t0
        li      t0, 0x10001         # L1D read miss
        csrw    mhpmevent4, t0
        la      a0, rows
        li      a7, 4
        ecall
        csrw    mhpmcounter3, zero
        csrw    mhpmcounter4, zero
        la      t1, matrix          # row by row: one word after the other
        li      t2, 1024
        li      s0, 0
1:      lw      t3, 0(t1)
        add     s0, s0, t3
        addi    t1, t1, 4
        addi    t2, t2, -1
        bnez    t2, 1b
        jal     report
        la      a0, cols
        li      a7, 4
        ecall
        csrw    mhpmcounter3, zero
        csrw    mhpmcounter4, zero
        la      t4, matrix          # column by column: 128 bytes apart
        li      t5, 32              # columns
2:      mv      t1, t4
        li      t2, 32              # rows
3:      lw      t3, 0(t1)
        add     s0, s0, t3
        addi    t1, t1, 128
        addi    t2, t2, -1
        bnez    t2, 3b
        addi    t4, t4, 4
        addi    t5, t5, -1
        bnez    t5, 2b
        jal     report
        li      a7, 10
        ecall

report: csrr    s1, mhpmcounter3    # (read first: the printing loads count too)
        csrr    s2, mhpmcounter4
        la      a0, loads
        li      a7, 4
        ecall
        mv      a0, s1
        li      a7, 1
        ecall
        la      a0, misses
        li      a7, 4
        ecall
        mv      a0, s2
        li      a7, 1
        ecall
        li      a0, '\n'
        li      a7, 11
        ecall
        ret
# eMIPS: --arch rv64 --rv-ext imafdc_zicsr_zifencei_zicntr_zihpm_sscofpmf --icache 2K:2:32 --dcache 2K:2:32
