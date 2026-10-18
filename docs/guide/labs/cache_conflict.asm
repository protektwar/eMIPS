# Add two arrays element by element.  They are exactly 1 KB apart, so in
# a 1 KB direct-mapped data cache a[i] and b[i] want the same line
# (guide 5.8 "Watching the caches", Lab 8).
        .data
a:      .space 1024                # 256 words
b:      .space 1024
        .text
main:   la    $t0, a
        la    $t1, b
        li    $t2, 256
        li    $v0, 0
loop:   lw    $t3, 0($t0)          # a[i]
        lw    $t4, 0($t1)          # b[i]: same set as a[i]
        addu  $v0, $v0, $t3
        addu  $v0, $v0, $t4
        addiu $t0, $t0, 4
        addiu $t1, $t1, 4
        addiu $t2, $t2, -1
        bnez  $t2, loop
        li    $v0, 10
        syscall
# eMIPS: --dcache 1K:1:16    (the settings this program runs with)
