# strlib.asm — a small string library, included by strlib_main.asm
# (Lab 95).  Each function takes its strings in $a0 / $a1 and returns $v0.

# strlen: $a0 = a string -> $v0 = its length
strlen: move  $v0, $zero
1:      lbu   $t0, 0($a0)
        beqz  $t0, 2f
        addiu $a0, $a0, 1
        addiu $v0, $v0, 1
        b     1b
2:      jr    $ra

# strcmp: $a0, $a1 = two strings -> $v0 < 0, 0 or > 0, as C's strcmp
strcmp: lbu   $t0, 0($a0)
        lbu   $t1, 0($a1)
        bne   $t0, $t1, 3f            # they differ here
        beqz  $t0, 3f                 # both ended: equal
        addiu $a0, $a0, 1
        addiu $a0, $a0, 1
        b     strcmp
3:      subu  $v0, $t0, $t1
        jr    $ra
