# Lab: coherence by hand (run with --cores 4 --caches).
# The boot maps kuseg coherent (CCA 5); this program first makes it
# cacheable but NOT coherent (Config.KU = 3), as on cores without a
# Coherence Manager. Then the write-back L1s keep stale copies: each
# writer pushes its line to memory (CACHE Hit Writeback D), the reader
# drops its stale copy before every read (CACHE Hit Invalidate D). Each
# core's sum and flag have a 32-byte line of their own: a writeback writes
# the whole line, so cores sharing one would overwrite each other's words.
        .eqv  LINE, 32
        .data
        .align 5                     # 32-byte aligned: one line per core
slot:   .space 4 * LINE              # slot[cpu]: +0 the sum, +4 done
        .text
main:   mfc0  $t0, $16               # Config
        li    $t1, ~(7 << 25)
        and   $t0, $t0, $t1
        li    $t1, 3 << 25
        or    $t0, $t0, $t1
        mtc0  $t0, $16               # KU = 3: kuseg cacheable, not coherent
        ehb
        mfc0  $s0, $15, 1            # EBase
        andi  $s0, $s0, 0x3ff        # CPUNum: 0..3
        li    $t0, 25
        mul   $t1, $s0, $t0          # first = 25 * cpu + 1
        addiu $t1, $t1, 1
        addu  $t2, $t1, $t0          # last + 1
        li    $t3, 0
sum:    addu  $t3, $t3, $t1
        addiu $t1, $t1, 1
        bne   $t1, $t2, sum
        la    $t7, slot
        sll   $t4, $s0, 5
        addu  $t7, $t7, $t4          # this core's line
        sw    $t3, 0($t7)            # the sum
        li    $t5, 1
        sw    $t5, 4($t7)            # done
        cache 0x19, 0($t7)           # Hit Writeback D: the line to memory
        sync
        bnez  $s0, rest              # cores 1..3 are finished
        li    $a0, 0                 # core 0: add up the four sums
        li    $t4, 0
next:   la    $t7, slot
        addu  $t7, $t7, $t4
wait:   cache 0x11, 0($t7)           # Hit Invalidate D: forget the copy
        lw    $t5, 4($t7)            # so this load reads memory
        beqz  $t5, wait
        lw    $t5, 0($t7)
        addu  $a0, $a0, $t5
        addiu $t4, $t4, LINE
        li    $t6, 4 * LINE
        bne   $t4, $t6, next
        li    $v0, 1                 # print 5050
        syscall
        li    $v0, 10
        syscall
rest:   wait
        b     rest
