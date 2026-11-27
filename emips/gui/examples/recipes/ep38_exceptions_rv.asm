# MIPS by Example in eMIPS, episode 38, the RISC-V ending: the same tries on
# RV32 — and only one of them traps.  RISC-V's add never traps (the sum wraps
# around), a division by zero gives -1 (no trap: the quotient is all ones),
# and this hart does a misaligned load in hardware (the spec lets it choose).
# What traps here: an illegal instruction — unimp, the all-zero halfword,
# illegal on purpose.  The trap goes to mtvec with mcause (the reason), mepc (the
# instruction), mtval (here: the instruction's bits); MRET goes back to mepc.
# eMIPS: --arch rv32
        .data
plus:   .asciz " + "
divide: .asciz " / "
equals: .asciz " = "
skip:   .asciz "skipped\n"
wordat: .asciz "word at "
illeg:  .asciz "an illegal instruction = "
total:  .asciz "exceptions handled: "
head:   .asciz "[exception "
colon:  .asciz ": "
nill:   .asciz "illegal instruction "
nother: .asciz "unexpected"
tail:   .asciz "] "
        .align 2
saved:  .word 0, 0, 0                 # a0, a7, t0 of the interrupted code
failed: .word 0
count:  .word 0

        .text
main:   addi  sp, sp, -16
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        addi  s0, sp, 16              # s0 = the frame pointer
        la    t0, handler             # where traps go: mtvec
        csrw  mtvec, t0
        li    a0, 7
        li    a1, 5
        call  try_add
        li    a0, 0x7fffffff          # the largest int + 1: wraps around
        li    a1, 1
        call  try_add
        li    a0, 100
        li    a1, 7
        call  try_div
        li    a0, 42                  # 42 / 0: -1, no trap
        li    a1, 0
        call  try_div
        li    a0, 0x10010001          # a word at an odd address: done in hardware
        call  try_load
        call  try_illegal             # the one that traps
        la    a0, total
        li    a7, 4
        ecall
        lw    a0, count
        li    a7, 1
        ecall
        li    a0, '\n'
        li    a7, 11
        ecall
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        li    a7, 10
        ecall

# print_op(a, text, b): "a text b = " (a leaf)
print_op:
        mv    t0, a1
        mv    t1, a2
        li    a7, 1                   # a
        ecall
        mv    a0, t0                  # " + " / " / "
        li    a7, 4
        ecall
        mv    a0, t1                  # b
        li    a7, 1
        ecall
        la    a0, equals
        li    a7, 4
        ecall
        ret

# try_add(a, b): "a + b = sum" — add: never a trap
try_add:
        addi  sp, sp, -16
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        sw    s1, 4(sp)
        addi  s0, sp, 16
        add   s1, a0, a1              # the sum (wraps around)
        mv    a2, a1
        la    a1, plus
        call  print_op
        mv    a0, s1
        call  result
        lw    s1, 4(sp)
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        ret

# try_div(a, b): "a / b = quotient" — div: by zero, -1
try_div:
        addi  sp, sp, -16
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        sw    s1, 4(sp)
        addi  s0, sp, 16
        div   s1, a0, a1
        mv    a2, a1
        la    a1, divide
        call  print_op
        mv    a0, s1
        call  result
        lw    s1, 4(sp)
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        ret

# try_load(address): "word at address = the word" (a leaf until result)
try_load:
        addi  sp, sp, -16
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        sw    s1, 4(sp)
        addi  s0, sp, 16
        mv    s1, a0
        la    a0, wordat
        li    a7, 4
        ecall
        mv    a0, s1                  # the address in hex
        li    a7, 34
        ecall
        la    a0, equals
        li    a7, 4
        ecall
        sw    zero, failed, t0
        lw    a0, 0(s1)               # misaligned: this hart reads it anyway
        call  result
        lw    s1, 4(sp)
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        ret

# try_illegal(): "an illegal instruction = 0", or "skipped"
try_illegal:
        addi  sp, sp, -16
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        addi  s0, sp, 16
        la    a0, illeg
        li    a7, 4
        ecall
        sw    zero, failed, t0
        li    a0, 0
        unimp                         # an illegal instruction (GNU as's: c.unimp, 0x0000)
        call  result
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        ret

# result(value): the value, or "skipped" when the try failed (a leaf)
result: lw    t0, failed
        bnez  t0, rskip
        li    a7, 1
        ecall
        li    a0, '\n'
        li    a7, 11
        ecall
        ret
rskip:  la    a0, skip
        li    a7, 4
        ecall
        ret

# the trap handler (M-mode, interrupts off): says what happened, skips the
# instruction (mepc + its size: 2 or 4 bytes) and returns with MRET
        .align 2
handler:
        sw    a0, saved, t6           # (t6: the address; saved first, below)
        sw    a7, saved+4, t6
        sw    t0, saved+8, t6
        la    a0, head
        li    a7, 4
        ecall
        csrr  t0, mcause              # the reason
        mv    a0, t0
        li    a7, 1
        ecall
        la    a0, colon
        li    a7, 4
        ecall
        li    a0, 2                   # 2 = illegal instruction
        bne   t0, a0, hother
        la    a0, nill
        li    a7, 4
        ecall
        csrr  a0, mtval               # its bits
        li    a7, 34
        ecall
        j     hdone
hother: la    a0, nother
        li    a7, 4
        ecall
hdone:  la    a0, tail
        li    a7, 4
        ecall
        li    t0, 1
        sw    t0, failed, a0
        lw    t0, count
        addi  t0, t0, 1
        sw    t0, count, a0
        csrr  t0, mtval               # the instruction: 2 or 4 bytes? (C: compressed
        andi  t0, t0, 3               #  ones are 2 — the low bits 11 = 4 bytes)
        li    a0, 3
        li    a7, 2
        bne   t0, a0, hskip
        li    a7, 4
hskip:  csrr  t0, mepc                # skip the instruction that trapped
        add   t0, t0, a7
        csrw  mepc, t0
        lw    t0, saved+8
        lw    a7, saved+4
        lw    a0, saved
        mret
