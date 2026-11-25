# MIPS by Example in eMIPS, episode 5, the challenge solved: strlcat(dst,
# src, size) appends src to the string in a buffer of size bytes, never
# writing past them, always ending with a 0; it returns the length it tried
# to make (strlen(dst) + strlen(src)): a result >= size means it was cut.
# (dst's string must already fit in its buffer.)
        .data
buf:    .space 16                     # 15 characters + the 0
visits: .word 7                       # the next thing in memory
hello:  .asciiz "Hello, "
more:   .asciiz "world, and more!"
says:   .asciiz "\nwanted = "
says2:  .asciiz "\nvisits = "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp, 20 $s0,
        sw    $ra, 28($sp)            # 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        la    $a0, buf                # strlcat(buf, hello, 16) into the
        la    $a1, hello              # empty buffer: a copy
        li    $a2, 16
        jal   strlcat
        la    $a0, buf                # strlcat(buf, more, 16): cut
        la    $a1, more
        li    $a2, 16
        jal   strlcat
        move  $s0, $v0                # the length it wanted
        li    $v0, 4                  # print buf
        la    $a0, buf
        syscall
        li    $v0, 4
        la    $a0, says
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 4
        la    $a0, says2
        syscall
        lw    $a0, visits             # still 7
        li    $v0, 1
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# strlcat: $a0 = dst, $a1 = src, $a2 = size -> $v0 = strlen(dst) +
# strlen(src) (a leaf: no frame).  $t2 = the address of the buffer's last
# byte, the only place left for the 0 once the buffer is full.
strlcat:
        move  $v0, $a0                # remember dst
        addu  $t2, $a0, $a2
        addiu $t2, $t2, -1            # dst + size - 1
find:   lbu   $t0, 0($a0)             # the end of dst's string
        beqz  $t0, copy
        addiu $a0, $a0, 1
        b     find
copy:   subu  $v0, $a0, $v0           # $v0 = strlen(dst) so far
copy1:  lbu   $t0, 0($a1)             # the next byte of src
        beqz  $t0, end
        addiu $v0, $v0, 1             # counted, whether it fits or not
        addiu $a1, $a1, 1
        beq   $a0, $t2, copy1         # the buffer is full: count, do not store
        sb    $t0, 0($a0)             # room: store it
        addiu $a0, $a0, 1
        b     copy1
end:    sb    $zero, 0($a0)           # always end with the 0
        jr    $ra
