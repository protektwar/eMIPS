# MIPS32 Release 6 (this example switches the release to R6):
# compact branches (bnezc, bc, balc, ...) have NO delay slot, and the
# instruction after a not-taken conditional compact branch is its
# "forbidden slot" (it may not be another branch).
        .text
main:   li    $t0, 0
        li    $t1, 5
loop:   addiu $t0, $t0, 10
        addiu $t1, $t1, -1
        bnezc $t1, loop         # compact: no delay slot
        lsa   $t2, $t0, $t0, 2  # R6 (and R5 MSA): $t2 = ($t0 << 2) + $t0
        li    $v0, 1
        move  $a0, $t2
        syscall                 # prints 250
        li    $v0, 10
        syscall
# eMIPS: --release R6    (the settings this program runs with)
