# MIPS by Example in eMIPS, episode 32's challenge, solved — a second
# library file: arr_find, with the same contract as ep32_arrays_lib.asm.
#   arr_find(a, n, x) -> $v0  the index of the first word equal to x, or -1
        .text
arr_find:
        li    $v0, 0                  # the index (a leaf: no frame)
afnext: beq   $v0, $a1, afnone        # all n looked at
        lw    $t0, 0($a0)
        beq   $t0, $a2, afdone        # found: $v0 is its index
        addiu $a0, $a0, 4
        addiu $v0, $v0, 1
        b     afnext
afnone: li    $v0, -1
afdone: jr    $ra
