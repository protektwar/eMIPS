# Sum a 2 KB array twice.  In a 1 KB data cache the second pass finds
# nothing left from the first: the array does not fit
# (guide 5.8 "Watching the caches", Lab 8).
        .data
array:  .space 2048                # 512 words
        .text
main:   li    $s1, 2               # two passes
pass:   la    $t0, array
        li    $t1, 512
loop:   lw    $t2, 0($t0)
        addu  $v0, $v0, $t2
        addiu $t0, $t0, 4
        addiu $t1, $t1, -1
        bnez  $t1, loop
        addiu $s1, $s1, -1
        bnez  $s1, pass
        li    $v0, 10
        syscall
# eMIPS: --dcache 1K:1:16    (the settings this program runs with)
