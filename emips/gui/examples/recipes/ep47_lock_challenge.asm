# MIPS by Example in eMIPS, episode 47 — the challenge, solved: no lock at
# all.  An atomic add: ll the counter itself, add, sc it back; if the other
# core wrote it in between, sc fails and the add is tried again.  Each core
# counts its failed sc's.  (The episode: two cores and a lock.)
        .eqv  N, 1000                 # additions per core
        .data
count:  .word 0
done:   .word 0                       # core 1 sets it when it has finished
fails1: .word 0                       # core 1's failed sc's
lcount: .asciiz "count = "
lwant:  .asciiz " (2 cores x 1000)\n"
lfail0: .asciiz "failed sc: core 0 "
lfail1: .asciiz ", core 1 "

        .text
main:   addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28 $s2, 24 $s1, 20 $s0, 0-15 home
        sw    $ra, 36($sp)
        sw    $fp, 32($sp)
        sw    $s2, 28($sp)
        sw    $s1, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        mfc0  $s1, $15, 1             # EBase: CPUNum in bits 9..0
        andi  $s1, $s1, 0x3ff         # $s1: which core am I?
        li    $s2, 0                  # $s2: my failed sc's
        li    $s0, N                  # $s0: additions left
again:  la    $a0, count              # atomic_add(&count, 1) -> its failures
        li    $a1, 1
        jal   atomic_add
        addu  $s2, $s2, $v0
        addiu $s0, $s0, -1
        bnez  $s0, again
        bnez  $s1, finish             # core 1: its failures, done, and stop
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
        li    $v0, 4                  # "failed sc: core 0 n, core 1 m"
        la    $a0, lfail0
        syscall
        li    $v0, 1
        move  $a0, $s2
        syscall
        li    $v0, 4
        la    $a0, lfail1
        syscall
        li    $v0, 1
        lw    $a0, fails1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s0, 20($sp)
        lw    $s1, 24($sp)
        lw    $s2, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        li    $v0, 10                 # the program ends
        syscall
finish: sw    $s2, fails1
        sync                          # (fails1 and count before done)
        li    $t0, 1
        sw    $t0, done
park:   wait                          # core 1 sleeps for good
        b     park

# atomic_add(p, n) -> $v0: *p += n as one indivisible step; $v0 = how many
# times sc failed (a leaf)
atomic_add:
        li    $v0, 0
aa:     ll    $t0, 0($a0)             # the counter, watched from now on
        addu  $t0, $t0, $a1
        sc    $t0, 0($a0)             # stored only if nobody wrote it since
        bnez  $t0, aadone
        addiu $v0, $v0, 1             # it failed: count it, and again
        b     aa
aadone: jr    $ra
# eMIPS: --cores 2
