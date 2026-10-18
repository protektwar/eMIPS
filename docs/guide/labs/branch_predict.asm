# Lab: branch prediction on the MIPS 24K (run with --model 24k).
# The loop branch is taken 99 times, then falls through: a 2-bit counter
# learns it at once.  The inner branch alternates: taken, not taken,
# taken ... — a 2-bit counter guesses it wrong about half the time.
        .text
main:   li    $t0, 100              # iterations
        li    $t1, 0                # alternates 0, 1, 0, 1 ...
loop:   xori  $t1, $t1, 1
        beq   $t1, $zero, even      # taken every other time
        nop
        addiu $t2, $t2, 1           # odd iterations
even:   addiu $t0, $t0, -1
        bnez  $t0, loop             # taken 99 times
        nop
        li    $v0, 10
        syscall
# eMIPS: --model 24k --pipeline    (the settings this program runs with)
