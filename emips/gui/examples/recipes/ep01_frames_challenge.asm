# MIPS by Example in eMIPS, episode 1, the challenge solved: max3(a, b, c)
# calls max2 twice.  It must save $ra (its jal overwrites it) and $fp (its
# frame pointer); c is still needed after the first call, and a call may
# change $a2, so c goes in $s0 — saved first: it is the caller's.  b is not
# needed after the first call: nothing keeps it.  max3_home keeps c in the
# caller's home area instead, as GCC does: no $s register at all.
        .data
text1:  .asciiz "max3(3, 9, 5) = "
text2:  .asciiz "max3_home(3, 9, 5) = "
text3:  .asciiz "max3(-2, -7, -4) = "
text4:  .asciiz "max3_home(-2, -7, -4) = "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 0-15 the home area of what it calls
        sw    $fp, 24($sp)
        move  $fp, $sp
        li    $a0, 3                  # max3(3, 9, 5)
        li    $a1, 9
        li    $a2, 5
        jal   max3
        la    $a0, text1
        move  $a1, $v0
        jal   print_result
        li    $a0, 3                  # max3_home(3, 9, 5)
        li    $a1, 9
        li    $a2, 5
        jal   max3_home
        la    $a0, text2
        move  $a1, $v0
        jal   print_result
        li    $a0, -2                 # max3(-2, -7, -4)
        li    $a1, -7
        li    $a2, -4
        jal   max3
        la    $a0, text3
        move  $a1, $v0
        jal   print_result
        li    $a0, -2                 # max3_home(-2, -7, -4)
        li    $a1, -7
        li    $a2, -4
        jal   max3_home
        la    $a0, text4
        move  $a1, $v0
        jal   print_result
        move  $sp, $fp                # main's epilogue
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10                 # exit
        syscall

# max2: $a0, $a1 -> $v0 = the larger (a leaf: no frame)
max2:   move  $v0, $a0
        bge   $a0, $a1, max2done
        move  $v0, $a1
max2done:
        jr    $ra

# max3: $a0, $a1, $a2 -> $v0 = the largest (it calls max2: a frame)
max3:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 0-15 home area
        sw    $ra, 28($sp)            # max2's jal overwrites $ra
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)            # we use $s0: the caller's, saved
        move  $fp, $sp
        move  $s0, $a2                # c must survive the first call
        jal   max2                    # max2(a, b): $a0, $a1 are already there
        move  $a0, $v0                # max2(that, c)
        move  $a1, $s0
        jal   max2
        move  $sp, $fp
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# max3_home: the same, c kept in the caller's home area (as GCC does)
max3_home:
        addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        sw    $a2, 32($fp)            # c into the caller's home area: 24 + 8
        jal   max2                    # max2(a, b)
        move  $a0, $v0
        lw    $a1, 32($fp)            # c, back from the home area
        jal   max2
        move  $sp, $fp
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        jr    $ra

# print_result: $a0 = a text, $a1 = a number (a leaf: only system calls)
print_result:
        li    $v0, 4
        syscall
        li    $v0, 1
        move  $a0, $a1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra
