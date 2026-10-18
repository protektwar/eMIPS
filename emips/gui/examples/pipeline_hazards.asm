# Pipeline hazards.  Choose "5-stage pipeline" and step one clock cycle
# at a time; watch the Datapath and the Pipeline chart.  Then switch
# forwarding off and compare the number of cycles (CPI in the status bar).
        .data
a:      .word 5
b:      .word 7
        .text
main:   la    $t0, a
        lw    $t1, 0($t0)
        addu  $t2, $t1, $t1     # needs $t1 from the load: load-use stall
        lw    $t3, 4($t0)
        addiu $t4, $zero, 1     # independent: hides the load delay
        addu  $t5, $t3, $t2     # $t3 forwarded from MEM/WB (no stall)
        subu  $t6, $t5, $t1     # $t5 forwarded from EX/MEM
        or    $t7, $t6, $t5     # $t6 from EX/MEM, $t5 from MEM/WB
        beq   $t7, $zero, done  # branch compares in ID: waits for $t7
        mul   $s0, $t7, $t7     # the multiplier holds EX for extra cycles
        addu  $s1, $s0, $s0     # ... and this one waits for its result
done:   li    $v0, 10
        syscall
# eMIPS: --pipeline    (the settings this program runs with)
