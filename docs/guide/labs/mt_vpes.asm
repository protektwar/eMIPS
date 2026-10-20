# Lab: two VPEs, two processors.  TC 0 starts TC 1 at second; TC 1
# takes a trap.  The handler notes which processor (EBase.CPUNum) took it
# and how far TC 0 counted meanwhile.  Run it with --mt 2:2 (TC 1 is
# VPE 1's) and with --mt 1:2 (both TCs in VPE 0).
        .data
cpu:    .word   -1, -1                  # EBase.CPUNum as TC 0 / TC 1 read it
taken:  .word   -1                      # the CPUNum of the handler's processor
moved:  .word   -1                      # how far TC 0 counted during the handler
count:  .word   0                       # TC 0's counter
done:   .word   0
        .text
main:   li      $t0, 1
        mtc0    $t0, $1, 1              # VPEControl.TargTC = 1
        mftc0   $t1, $1, 2              # TC 1's VPEConf0 ...
        ori     $t1, $t1, 1
        mttc0   $t1, $1, 2              # ... VPA: its VPE may run
        mftc0   $t1, $12                # TC 1's Status: EXL = ERL = 0
        li      $t2, ~0x6
        and     $t1, $t1, $t2
        mttc0   $t1, $12
        la      $t1, second
        mttc0   $t1, $2, 3              # TCRestart
        li      $t1, 0x2000
        mttc0   $t1, $2, 1              # TCStatus.A
        mttc0   $zero, $2, 4            # TCHalt = 0
        emt                             # this VPE: several TCs (--mt 1:2)
        evpe                            # every VPE runs (--mt 2:2)

        mfc0    $t0, $15, 1             # EBase: CPUNum in bits 9:0
        andi    $t0, $t0, 0x3ff
        sw      $t0, cpu
        la      $t3, count
spin:   lw      $t4, 0($t3)             # count until TC 1 is done
        addiu   $t4, $t4, 1
        sw      $t4, 0($t3)
        lw      $t5, done
        beqz    $t5, spin
        lw      $a0, cpu
        jal     print
        lw      $a0, cpu + 4
        jal     print
        lw      $a0, taken
        jal     print
        lw      $a0, moved
        jal     print
        li      $v0, 10
        syscall

second: mfc0    $t0, $15, 1             # TC 1: which processor am I?
        andi    $t0, $t0, 0x3ff
        sw      $t0, cpu + 4
        teq     $zero, $zero            # a trap: the exception handler
        li      $t0, 1
        sw      $t0, done
idle:   wait
        b       idle

print:  li      $v0, 1                  # print $a0 and a space
        syscall
        li      $a0, ' '
        li      $v0, 11
        syscall
        jr      $ra

        .ktext  0x80000180
        mfc0    $k0, $15, 1
        andi    $k0, $k0, 0x3ff
        sw      $k0, taken              # this processor's CPUNum
        lw      $k1, count
        li      $k0, 100                # take a while
1:      addiu   $k0, $k0, -1
        bnez    $k0, 1b
        lw      $k0, count
        subu    $k1, $k0, $k1
        sw      $k1, moved              # what TC 0 did meanwhile
        mfc0    $k0, $14
        addiu   $k0, $k0, 4             # past the teq
        mtc0    $k0, $14
        ehb
        eret
# eMIPS: --mt 2:2
