# Sum an array twice: the first pass misses in the data cache, the
# second hits.  Run it with and without --caches (guide 5.8).
        .data
array:  .word 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        .text
main:   li    $s1, 2               # two passes
pass:   la    $t0, array
        li    $t1, 16
        li    $v0, 0
loop:   lw    $t2, 0($t0)
        addiu $t0, $t0, 4
        addu  $v0, $v0, $t2
        addiu $t1, $t1, -1
        bnez  $t1, loop
        addiu $s1, $s1, -1
        bnez  $s1, pass
        move  $a0, $v0
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall
