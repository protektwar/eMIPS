# Lab: who gets the issue slots?  Two threads count as fast as they can;
# TC 0 stops after 3000 rounds and prints its count and TC 1's.  The
# weighted round-robin policy manager (--mt-pm wrr) puts each TC in a
# group (TCSchedule.GRP, 0-3): change the groups and compare.
        .eqv    GROUP0, 1               # TC 0's group
        .eqv    GROUP1, 0               # TC 1's group
        .text
main:   li      $t0, 1
        mtc0    $t0, $1, 1              # VPEControl.TargTC = 1
        la      $t1, counter
        mttc0   $t1, $2, 3              # TC 1: TCRestart = counter
        li      $t1, GROUP1
        mttc0   $t1, $2, 6              # TC 1: TCSchedule.GRP
        li      $t1, 0x2000
        mttc0   $t1, $2, 1              # TC 1: TCStatus.A
        mttc0   $zero, $2, 4            # TC 1: TCHalt = 0
        li      $t1, GROUP0
        mtc0    $t1, $2, 6              # TC 0: TCSchedule.GRP
        mtc0    $zero, $1, 5            # VPESchedule.GPO = 0: the groups take turns
        emt
        evpe

        li      $s0, 0
        li      $t2, 3000
        .set    noreorder
count0: addiu   $t2, $t2, -1            # TC 0: 3000 rounds of 3 instructions
        bnez    $t2, count0
        addiu   $s0, $s0, 1
        .set    reorder
        dmt                             # only TC 0 now
        mftgpr  $s1, $s0                # TC 1's count
        move    $a0, $s0
        li      $v0, 1
        syscall
        li      $a0, ' '
        li      $v0, 11
        syscall
        move    $a0, $s1
        li      $v0, 1
        syscall
        li      $v0, 10
        syscall

        .set    noreorder
counter:
        b       counter                 # TC 1: 2 instructions a round
        addiu   $s0, $s0, 1
        .set    reorder
# eMIPS: --mt 1:2 --mt-pm wrr
