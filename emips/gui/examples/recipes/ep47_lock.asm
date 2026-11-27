# MIPS by Example in eMIPS, episode 47 — Two cores and a lock.  Both cores
# run this same program, each from main, each with its own stack; they
# share memory.  Each adds 1 to count 1000 times.  count = count + 1 is a
# load, an add and a store: two cores doing it at once can both load 7 and
# both store 8 — one addition lost.  A lock makes it one core at a time:
# ll (load linked) reads the lock and starts watching it, sc (store
# conditional) writes only if nobody wrote it since, and says whether it did.
# CP0's EBase (register 15, select 1) tells a core its number.
        .eqv  N, 1000                 # additions per core
        .data
lock:   .word 0                       # 0 free, 1 taken
count:  .word 0
done:   .word 0                       # core 1 sets it when it has finished
lcount: .asciiz "count = "
lwant:  .asciiz " (2 cores x 1000)\n"

        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s1, 16 $s0, 0-15 the home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s1, 20($sp)
        sw    $s0, 16($sp)
        move  $fp, $sp
        mfc0  $s0, $15, 1             # EBase: CPUNum in bits 9..0
        andi  $s0, $s0, 0x3ff         # $s0: which core am I?
        li    $s1, N                  # $s1: additions left
again:  la    $a0, lock               # lock(&lock)
        jal   acquire
        lw    $t0, count              # the critical section: one core at a time
        addiu $t0, $t0, 1
        sw    $t0, count
        la    $a0, lock               # unlock(&lock)
        jal   release
        addiu $s1, $s1, -1
        bnez  $s1, again
        bnez  $s0, finish             # core 1: say so, and stop
wait1:  lw    $t0, done               # core 0: wait for core 1, then print
        beqz  $t0, wait1
        li    $v0, 4
        la    $a0, lcount
        syscall
        li    $v0, 1
        lw    $a0, count
        syscall
        li    $v0, 4
        la    $a0, lwant
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s0, 16($sp)
        lw    $s1, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10                 # the program ends
        syscall
finish: sync                          # (count's last store before done's)
        li    $t0, 1
        sw    $t0, done
park:   wait                          # core 1 sleeps for good
        b     park

# acquire(lock): spins until it changes the lock from 0 to 1 itself (a leaf)
acquire:
        ll    $t0, 0($a0)             # the lock, watched from now on
        bnez  $t0, acquire            # taken: try again
        li    $t0, 1
        sc    $t0, 0($a0)             # 1, if nobody wrote it since the ll
        beqz  $t0, acquire            # $t0 = 0: someone did — again
        sync                          # nothing of the section moves above
        jr    $ra

# release(lock): (a leaf)
release:
        sync                          # nothing of the section moves below
        sw    $zero, 0($a0)
        jr    $ra
# eMIPS: --cores 2
