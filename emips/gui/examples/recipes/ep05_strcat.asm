# MIPS by Example in eMIPS, episode 5 — Copy, concatenate, and the
# overflow: strcpy copies a string into a buffer, strcat appends another at
# its end.  The buffer has room for 15 characters and the 0; the word right
# after it in memory (visits) shows whether anything was written past it.
        .data
buf:    .space 16                     # 15 characters + the 0
visits: .word 7                       # the next thing in memory
hello:  .asciiz "Hello, "
world:  .asciiz "world!"
says:   .asciiz "\nvisits = "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 0-15 the home area
        sw    $fp, 24($sp)
        move  $fp, $sp
        la    $a0, buf                # strcpy(buf, hello)
        la    $a1, hello
        jal   strcpy
        la    $a0, buf                # strcat(buf, world)
        la    $a1, world
        jal   strcat
        li    $v0, 4                  # print buf
        la    $a0, buf
        syscall
        li    $v0, 4
        la    $a0, says
        syscall
        lw    $a0, visits             # still 7?
        li    $v0, 1
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# strcpy: copy the string at $a1, its 0 too, to $a0 (a leaf: no frame)
strcpy: lbu   $t0, 0($a1)
        sb    $t0, 0($a0)
        addiu $a0, $a0, 1
        addiu $a1, $a1, 1
        bnez  $t0, strcpy             # until the 0 is copied
        jr    $ra

# strcat: append the string at $a1 to the string at $a0 (it calls strcpy:
# a frame)
strcat:
        addiu $sp, $sp, -32           # the frame: 28 $ra, 24 $fp, 0-15 home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        move  $fp, $sp
find:   lbu   $t0, 0($a0)             # find the 0 at the end of $a0's string
        beqz  $t0, found
        addiu $a0, $a0, 1
        b     find
found:  jal   strcpy                  # copy $a1's string there
        move  $sp, $fp                # restore, in reverse
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra
