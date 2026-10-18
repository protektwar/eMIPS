# mul holds EX for its latency; mult works in the background (HI/LO).
        .text
main:   li    $t0, 6
        mul   $t1, $t0, $t0     # EX for 4 cycles
        addu  $t2, $t1, $t1     # waits for it
        mult  $t0, $t0          # the multiply unit, in parallel
        addiu $t3, $zero, 1     # independent: goes on
        mflo  $t4               # needs LO: waits for the multiply unit
        li    $v0, 10
        syscall
