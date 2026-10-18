# Lab: four cores sum 1..100 together (run with --cores 4).
# The boot firmware starts every core at main; each one reads its number
# (EBase.CPUNum), sums its quarter and stores it; core 0 waits for the
# three others' results, adds them up and prints 5050.
        .data
part:   .word 0, 0, 0, 0             # each core's sum
done:   .word 0, 0, 0, 0             # set when its sum is there
        .text
main:   mfc0  $s0, $15, 1            # EBase
        andi  $s0, $s0, 0x3ff        # CPUNum: 0..3
        li    $t0, 25
        mul   $t1, $s0, $t0          # first = 25 * cpu + 1
        addiu $t1, $t1, 1
        addu  $t2, $t1, $t0          # last + 1
        li    $t3, 0
sum:    addu  $t3, $t3, $t1
        addiu $t1, $t1, 1
        bne   $t1, $t2, sum
        sll   $t4, $s0, 2
        sw    $t3, part($t4)         # part[cpu] = the sum
        li    $t5, 1
        sw    $t5, done($t4)         # then done[cpu] = 1
        bnez  $s0, rest              # cores 1..3 are finished
        li    $t4, 4                 # core 0: wait for done[1..3]
        li    $t6, 16
wait:   lw    $t5, done($t4)
        beqz  $t5, wait
        addiu $t4, $t4, 4
        bne   $t4, $t6, wait
        lw    $a0, part
        lw    $t5, part + 4
        addu  $a0, $a0, $t5
        lw    $t5, part + 8
        addu  $a0, $a0, $t5
        lw    $t5, part + 12
        addu  $a0, $a0, $t5
        li    $v0, 1                 # print 5050
        syscall
        li    $v0, 10
        syscall
rest:   wait                         # nothing to do: wait (for an interrupt)
        b     rest
