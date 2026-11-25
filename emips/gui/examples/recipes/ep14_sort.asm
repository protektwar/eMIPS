# MIPS by Example in eMIPS, episode 14 — Sorting an array of strings: the
# list is an array of addresses; isort, an insertion sort, compares the
# strings with episode 4's strcmp but moves only their addresses: the
# strings themselves never move.
        .data
pear:   .asciiz "pear"
apple:  .asciiz "apple"
mips:   .asciiz "MIPS"
zebra:  .asciiz "zebra"
banana: .asciiz "banana"
apple2: .asciiz "Apple"
        .align 2
names:  .word pear, apple, mips, zebra, banana, apple2
before: .asciiz "before:"
after:  .asciiz "after: "
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        li    $v0, 4
        la    $a0, before
        syscall
        la    $a0, names              # show(names, 6)
        li    $a1, 6
        jal   show
        la    $a0, names              # isort(names, 6)
        li    $a1, 6
        jal   isort
        li    $v0, 4
        la    $a0, after
        syscall
        la    $a0, names
        li    $a1, 6
        jal   show
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# show: print the $a1 strings whose addresses are at $a0, each after a
# space, and a newline (a leaf: system calls)
show:   move  $t0, $a0
shnext: beqz  $a1, shdone
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 4
        lw    $a0, 0($t0)
        syscall
        addiu $t0, $t0, 4
        addiu $a1, $a1, -1
        b     shnext
shdone: li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra

# isort: sort the $a1 addresses at $a0 so their strings are in strcmp order
# (it calls strcmp: a frame; $s0 = the array, $s1 = n, $s2 = i, $s3 = the
# key, the address being placed, $s4 = j)
isort:  addiu $sp, $sp, -48           # 44 $ra, 40 $fp, 36-20 $s0-$s4,
        sw    $ra, 44($sp)            # 0-15 the home area
        sw    $fp, 40($sp)
        sw    $s0, 36($sp)
        sw    $s1, 32($sp)
        sw    $s2, 28($sp)
        sw    $s3, 24($sp)
        sw    $s4, 20($sp)
        move  $fp, $sp
        move  $s0, $a0
        move  $s1, $a1
        li    $s2, 1                  # i = 1: a[0] alone is sorted
outer:  bge   $s2, $s1, idone
        sll   $t0, $s2, 2
        addu  $t0, $s0, $t0
        lw    $s3, 0($t0)             # key = a[i]
        addiu $s4, $s2, -1            # j = i - 1
inner:  bltz  $s4, place              # past the front: place the key at 0
        sll   $t0, $s4, 2
        addu  $t0, $s0, $t0
        lw    $a0, 0($t0)             # strcmp(a[j], key)
        move  $a1, $s3
        jal   strcmp
        blez  $v0, place              # a[j] <= key: the key goes after it
        sll   $t0, $s4, 2             # a[j + 1] = a[j]: one place up
        addu  $t0, $s0, $t0
        lw    $t1, 0($t0)
        sw    $t1, 4($t0)
        addiu $s4, $s4, -1
        b     inner
place:  addiu $t0, $s4, 1             # a[j + 1] = key
        sll   $t0, $t0, 2
        addu  $t0, $s0, $t0
        sw    $s3, 0($t0)
        addiu $s2, $s2, 1
        b     outer
idone:  move  $sp, $fp
        lw    $s4, 20($sp)
        lw    $s3, 24($sp)
        lw    $s2, 28($sp)
        lw    $s1, 32($sp)
        lw    $s0, 36($sp)
        lw    $fp, 40($sp)
        lw    $ra, 44($sp)
        addiu $sp, $sp, 48
        jr    $ra

# strcmp: episode 4's (a leaf)
strcmp: lbu   $t0, 0($a0)
        lbu   $t1, 0($a1)
        bne   $t0, $t1, differ
        beqz  $t0, differ
        addiu $a0, $a0, 1
        addiu $a1, $a1, 1
        b     strcmp
differ: subu  $v0, $t0, $t1
        jr    $ra
