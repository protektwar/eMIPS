# MIPS by Example in eMIPS, episode 9 — Finding a substring: find tries
# every starting place in the text; at each, two pointers walk the text and
# the needle together until the needle ends (found) or a character differs
# (try the next place).  It returns the place, or -1.
        .data
text:   .asciiz "she sells sea shells by the sea shore"
sea:    .asciiz "sea"
shell:  .asciiz "shell"
shore:  .asciiz "shore"
sure:   .asciiz "sure"
needles:.word sea, shell, shore, sure, 0
colon:  .asciiz ": "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $s0, needles            # the next needle
next:   lw    $s1, 0($s0)
        beqz  $s1, end
        la    $a0, text               # find(text, needle)
        move  $a1, $s1
        jal   find
        move  $t0, $v0                # (no call until it is printed)
        li    $v0, 4                  # needle: place
        move  $a0, $s1
        syscall
        la    $a0, colon
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        addiu $s0, $s0, 4
        b     next
end:    move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# find: $a0 = a text, $a1 = a needle -> $v0 = where the needle first starts
# in the text, or -1 (a leaf; $t0 = the starting place being tried)
find:   move  $t0, $a0
start:  move  $t1, $t0                # walk the text from the starting place
        move  $t2, $a1                # and the needle from its start
match:  lbu   $t4, 0($t2)
        beqz  $t4, found              # the needle ended: every character matched
        lbu   $t3, 0($t1)
        bne   $t3, $t4, again         # a difference (the text's 0 is one too)
        addiu $t1, $t1, 1
        addiu $t2, $t2, 1
        b     match
again:  lbu   $t3, 0($t0)             # the next starting place, if the text
        beqz  $t3, notfound           # has not ended
        addiu $t0, $t0, 1
        b     start
found:  subu  $v0, $t0, $a0           # the place: an offset from the text's start
        jr    $ra
notfound:
        li    $v0, -1
        jr    $ra
