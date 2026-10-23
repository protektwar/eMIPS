# RISC-V: the sum of an array — a loop, loads, RARS's ecall services.
# Run it in pipeline mode to watch the load-use stall (lw, then add) and
# the branch decided in ID (bnez: one bubble when it is taken).
        .data
array:  .word 3, 1, 4, 1, 5, 9, 2, 6
msg:    .asciz "sum = "
        .text
main:   la   t0, array          # the array's address (auipc + addi)
        li   t1, 8              # elements left
        li   t2, 0              # the sum
loop:   lw   t3, 0(t0)
        add  t2, t2, t3         # (needs the load's value: load-use)
        addi t0, t0, 4
        addi t1, t1, -1
        bnez t1, loop
        la   a0, msg
        li   a7, 4              # PrintString
        ecall
        mv   a0, t2
        li   a7, 1              # PrintInt
        ecall
        li   a7, 10             # Exit
        ecall
# eMIPS: --arch rv32
