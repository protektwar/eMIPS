# RISC-V: a recursive function — jal / ret (jalr), the stack, callee-saved
# registers (the psABI: ra and s0 saved by the callee).  RV64: 64-bit
# registers, ld / sd for the saved doublewords.
        .text
main:   li   a0, 20
        call fact               # (jal ra, fact — or auipc + jalr when far)
        li   a7, 1              # PrintInt: 20! fits in 64 bits, RARS prints
        ecall                   # the low 32 bits of a0 as an int
        li   a7, 11
        li   a0, 10
        ecall
        li   a0, 5
        call fact
        li   a7, 1
        ecall
        li   a7, 10
        ecall

# fact(n) = n <= 1 ? 1 : n * fact(n - 1)
fact:   li   t0, 1
        bgt  a0, t0, 1f
        li   a0, 1
        ret
1:      addi sp, sp, -16
        sd   ra, 8(sp)
        sd   s0, 0(sp)
        mv   s0, a0
        addi a0, a0, -1
        call fact
        mul  a0, a0, s0
        ld   ra, 8(sp)
        ld   s0, 0(sp)
        addi sp, sp, 16
        ret
# eMIPS: --arch rv64
