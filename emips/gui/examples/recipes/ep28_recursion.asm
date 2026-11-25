# MIPS by Example in eMIPS, episode 28 — Recursion: from n to 1 and back,
# all binary patterns.  updown prints n on the way down and again on the
# way back up — each call keeps its own n in $s0, saved in its frame.
# patterns writes 0, then 1, into place i and recurses for i + 1; at the
# end of the buffer it prints: 2 to the n patterns.
        .data
buf:    .space 4                      # n digits and the 0 after them
updw:   .asciiz "updown(3):"
patw:   .asciiz "patterns(3):"
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        li    $v0, 4                  # updown(3)
        la    $a0, updw
        syscall
        li    $a0, 3
        jal   updown
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $v0, 4                  # every pattern of 3 binary digits
        la    $a0, patw
        syscall
        sb    $zero, buf + 3          # the buffer ends after 3 digits
        li    $a0, 0                  # from place 0 ...
        li    $a1, 3                  # ... of 3
        jal   patterns
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# updown: print n, n - 1, ..., 1, ..., n - 1, n (each after a space)
updown: addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 0-15 home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)            # the caller's $s0: ours to restore
        move  $fp, $sp
        move  $s0, $a0                # this call's n
        jal   print_num               # on the way down
        beq   $s0, 1, udone           # 1: the bottom, turn back
        addiu $a0, $s0, -1
        jal   updown                  # n - 1 ... 1 ... n - 1
        move  $a0, $s0                # and n again, on the way back up
        jal   print_num
udone:  move  $sp, $fp
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# patterns: $a0 = i, $a1 = n: fill buf[i .. n - 1] with every mix of 0 and
# 1, printing each complete buffer (it calls itself: a frame, i and n in
# $s0 and $s1)
patterns:
        addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 16 $s1
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        move  $s0, $a0
        move  $s1, $a1
        bne   $s0, $s1, pfill         # i = n: a complete pattern
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 4
        la    $a0, buf
        syscall
        b     pdone
pfill:  li    $t0, '0'                # buf[i] = '0', the rest after it
        sb    $t0, buf($s0)
        addiu $a0, $s0, 1
        move  $a1, $s1
        jal   patterns
        li    $t0, '1'                # buf[i] = '1', the rest after it
        sb    $t0, buf($s0)
        addiu $a0, $s0, 1
        move  $a1, $s1
        jal   patterns
pdone:  move  $sp, $fp
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# print_num: a space, then $a0 (a leaf)
print_num:
        move  $t0, $a0
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        jr    $ra
