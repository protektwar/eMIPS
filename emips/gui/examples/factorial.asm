# Function calls: jal / jr $ra, argument in $a0, result in $v0, and the
# stack ($sp) to save $ra and $s0 across a recursive call.
        .text
main:   li    $a0, 5
        jal   fact              # $v0 = fact(5)
        move  $a0, $v0
        li    $v0, 1
        syscall                 # prints 120
        li    $v0, 10
        syscall

# int fact(int n) { return n <= 0 ? 1 : n * fact(n - 1); }
fact:   addiu $sp, $sp, -8      # push $ra and $s0
        sw    $ra, 4($sp)
        sw    $s0, 0($sp)
        move  $s0, $a0
        li    $v0, 1
        blez  $s0, done         # fact(0) = 1
        addiu $a0, $s0, -1
        jal   fact              # $v0 = fact(n - 1)
        mul   $v0, $v0, $s0
done:   lw    $s0, 0($sp)       # pop
        lw    $ra, 4($sp)
        addiu $sp, $sp, 8
        jr    $ra
