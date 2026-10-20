# Lab: hiding a cache miss with a second thread (the 34K pipeline).
# The program adds the first word of 64 cache lines: every load misses
# the data cache.  With THREADS = 2, TC 1 adds the second half while
# TC 0 adds the first; with THREADS = 1, TC 0 adds all 64.
        .eqv    THREADS, 2
        .eqv    LINES, 64
        .data
        .align  5
table:  .space  LINES * 32              # 64 lines of 32 bytes
part:   .word   0                       # TC 1's sum
done:   .word   0
        .text
main:   la      $t0, table              # the first word of line i = i + 1
        li      $t1, 1
fill:   sw      $t1, 0($t0)
        cache   0x15, 0($t0)            # Hit Writeback Invalidate D: out of the cache
        addiu   $t0, $t0, 32
        addiu   $t1, $t1, 1
        slti    $t2, $t1, LINES + 1
        bnez    $t2, fill
        sync
        mfc0    $s7, $9                 # Count (one tick every 2 cycles): the start
        la      $a0, table
        li      $a1, LINES
        li      $t9, THREADS
        li      $t0, 1
        beq     $t9, $t0, go            # one thread: TC 0 alone
        li      $a1, LINES / 2
        mtc0    $t0, $1, 1              # VPEControl.TargTC = 1
        la      $t1, worker
        mttc0   $t1, $2, 3              # TC 1: TCRestart = worker
        addiu   $t1, $a0, LINES / 2 * 32
        mttgpr  $t1, $a0                # its $a0: the second half
        mttgpr  $a1, $a1                # its $a1: 32 lines
        li      $t1, 0x2000
        mttc0   $t1, $2, 1              # TCStatus.A
        mttc0   $zero, $2, 4            # TCHalt = 0
        emt
        evpe
go:     jal     sum
        move    $s0, $v0
        li      $t0, 1
        beq     $t9, $t0, print
wait:   lw      $t3, done               # two threads: wait for TC 1
        beqz    $t3, wait
        lw      $t4, part
        addu    $s0, $s0, $t4
print:  mfc0    $t5, $9                 # Count: the end
        subu    $t5, $t5, $s7
        sll     $s1, $t5, 1             # cycles
        move    $a0, $s0                # the sum (2080)
        li      $v0, 1
        syscall
        li      $a0, ' '
        li      $v0, 11
        syscall
        move    $a0, $s1                # the cycles it took
        li      $v0, 1
        syscall
        li      $v0, 10
        syscall

worker: jal     sum
        sw      $v0, part
        li      $t0, 1
        sw      $t0, done
idle:   wait
        b       idle

sum:    li      $v0, 0                  # $v0 = the first words of $a1 lines at $a0
1:      lw      $t8, 0($a0)             # a miss
        addu    $v0, $v0, $t8           # needs it at once
        addiu   $a1, $a1, -1
        addiu   $a0, $a0, 32
        bnez    $a1, 1b
        jr      $ra
# eMIPS: --mt 1:2 --model 24k --caches --pipeline    (the settings this program runs with)
