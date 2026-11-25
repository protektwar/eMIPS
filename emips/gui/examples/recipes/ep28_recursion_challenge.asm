# MIPS by Example in eMIPS, episode 28, the challenge solved.  power by
# squaring: x^n = (x^(n/2))², times x once more when n is odd — 13 needs 5
# calls, not 13 multiplications.  permute prints every order of "abc": put
# each letter first in turn (a swap), arrange the rest, swap back.
        .data
word:   .asciiz "abc"
powerw: .asciiz "3^13 = "
permw:  .asciiz "permutations:"
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        li    $a0, 3                  # power(3, 13)
        li    $a1, 13
        jal   power
        move  $t0, $v0
        li    $v0, 4
        la    $a0, powerw
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $v0, 4                  # every order of "abc"
        la    $a0, permw
        syscall
        li    $a0, 0                  # from place 0 ...
        li    $a1, 3                  # ... of 3
        jal   permute
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# power: $a0 = x, $a1 = n -> $v0 = x^n (it calls itself: a frame, x in $s0
# and n in $s1)
power:  addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 16 $s1
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        li    $v0, 1                  # x^0 = 1
        beqz  $a1, pwdone
        move  $s0, $a0
        move  $s1, $a1
        srl   $a1, $a1, 1             # power(x, n / 2)
        jal   power
        mul   $v0, $v0, $v0           # squared
        andi  $t0, $s1, 1
        beqz  $t0, pwdone             # n odd: once more x
        mul   $v0, $v0, $s0
pwdone: move  $sp, $fp
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# permute: $a0 = i, $a1 = n: print every order of word[i .. n - 1] after
# word[0 .. i - 1] (it calls itself: a frame, i, n and j in $s0-$s2)
permute:
        addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28-20 $s0-$s2
        sw    $ra, 36($sp)
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        move  $fp, $sp
        move  $s0, $a0
        move  $s1, $a1
        bne   $s0, $s1, pmore         # i = n: one order, complete
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 4
        la    $a0, word
        syscall
        b     pmdone
pmore:  move  $s2, $s0                # j = i ... n - 1
pnext:  move  $a0, $s0                # swap word[i], word[j]: word[j] first
        move  $a1, $s2
        jal   swap
        addiu $a0, $s0, 1             # the rest, every way
        move  $a1, $s1
        jal   permute
        move  $a0, $s0                # swap back
        move  $a1, $s2
        jal   swap
        addiu $s2, $s2, 1
        bne   $s2, $s1, pnext
pmdone: move  $sp, $fp
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra

# swap: word[$a0] <-> word[$a1] (a leaf)
swap:   lbu   $t0, word($a0)
        lbu   $t1, word($a1)
        sb    $t1, word($a0)
        sb    $t0, word($a1)
        jr    $ra
