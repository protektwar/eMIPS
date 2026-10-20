# Lab: FORK.  TC 0 forks three threads; each adds a quarter of the
# numbers 1-100, TC 0 the last quarter, and then it adds the four sums.
# A FORK needs a free thread context: run it with --mt 1:3 too.
        .data
nums:   .word   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
        .word   21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40
        .word   41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60
        .word   61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80
        .word   81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100
sums:   .word   0, 0, 0, 0              # one per quarter
left:   .word   3                       # threads still running
        .text
main:   mfc0    $t0, $2, 1              # TCStatus of this TC
        ori     $t0, $t0, 0x8000
        mtc0    $t0, $2, 1              # DA: dynamically allocatable
        li      $t1, 1                  # TCs 1, 2, 3: free (DA, not activated, not halted)
free:   mtc0    $t1, $1, 1              # VPEControl.TargTC
        li      $t0, 0x8000
        mttc0   $t0, $2, 1              # its TCStatus: DA
        mttc0   $zero, $2, 4            # its TCHalt = 0
        addiu   $t1, $t1, 1
        slti    $t2, $t1, 4
        bnez    $t2, free
        emt                             # VPEControl.TE: several TCs issue
        evpe                            # MVPControl.EVP

        la      $t2, worker
        li      $s0, 0                  # quarter 0, 1, 2 for the new threads
spawn:  fork    $a0, $t2, $s0           # a free TC starts at worker, its $a0 = $s0
        addiu   $s0, $s0, 1
        slti    $t3, $s0, 3
        bnez    $t3, spawn
        li      $a0, 3                  # the last quarter here
        jal     quarter

join:   lw      $t3, left               # wait until the three are done
        bnez    $t3, join
        la      $t4, sums
        lw      $a0, 0($t4)
        lw      $t5, 4($t4)
        addu    $a0, $a0, $t5
        lw      $t5, 8($t4)
        addu    $a0, $a0, $t5
        lw      $t5, 12($t4)
        addu    $a0, $a0, $t5
        li      $v0, 1
        syscall
        li      $v0, 10
        syscall

worker: jal     quarter
1:      ll      $t0, left               # one thread less (atomically)
        addiu   $t0, $t0, -1
        sc      $t0, left
        beqz    $t0, 1b
        yield   $zero                   # done: this TC is free again

quarter:                                # sums[$a0] = the 25 numbers of quarter $a0
        sll     $t0, $a0, 2
        sll     $t1, $a0, 6
        sll     $t9, $a0, 5
        addu    $t1, $t1, $t9
        addu    $t1, $t1, $t0           # $t1 = $a0 * 100: the quarter's offset
        la      $t9, nums
        addu    $t1, $t1, $t9
        li      $v0, 0
        li      $t9, 25
2:      lw      $t8, 0($t1)
        addu    $v0, $v0, $t8
        addiu   $t1, $t1, 4
        addiu   $t9, $t9, -1
        bnez    $t9, 2b
        la      $t9, sums
        addu    $t9, $t9, $t0
        sw      $v0, 0($t9)
        jr      $ra

        .ktext  0x80000180              # an exception: print why, then stop
        mfc0    $k0, $13                # Cause
        srl     $a0, $k0, 2
        andi    $a0, $a0, 31            # ExcCode (25: Thread)
        li      $v0, 1
        syscall
        li      $a0, ' '
        li      $v0, 11
        syscall
        mfc0    $k0, $1, 1              # VPEControl
        srl     $a0, $k0, 16
        andi    $a0, $a0, 7             # EXCPT: which Thread exception
        li      $v0, 1
        syscall
        li      $v0, 10
        syscall
# eMIPS: --mt 1:4
