# MIPS by Example in eMIPS, episode 9, the challenge solved.  all prints
# every place a needle starts: after each find it searches again from one
# character past the last place.  last returns the last place the same way:
# it keeps the latest one find returned.  Both reuse episode 9's find.
        .data
text:   .asciiz "she sells sea shells by the sea shore"
sea:    .asciiz "sea"
ll:     .asciiz "ll"
s:      .asciiz "s"
colon:  .asciiz ":"
lasts:  .asciiz "last s: "
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        la    $a0, sea                # all(sea)
        jal   all
        la    $a0, ll                 # all(ll)
        jal   all
        la    $a0, text               # last(text, s)
        la    $a1, s
        jal   last
        move  $t0, $v0
        li    $v0, 4
        la    $a0, lasts
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# all: print "needle: p1 p2 ..." for every place the needle at $a0 starts
# in text (it calls find: a frame; $s0 = the needle, $s1 = where to search
# from)
all:    addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20-16 $s0-$s1,
        sw    $ra, 28($sp)            # 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        move  $s0, $a0
        li    $v0, 4                  # the needle and a colon
        syscall
        la    $a0, colon
        syscall
        la    $s1, text
anext:  move  $a0, $s1                # find(from, needle)
        move  $a1, $s0
        jal   find
        bltz  $v0, adone              # no more
        addu  $s1, $s1, $v0           # the place itself ...
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1                  # ... printed as an offset from text
        la    $t0, text
        subu  $a0, $s1, $t0
        syscall
        addiu $s1, $s1, 1             # search again from one past it
        b     anext
adone:  li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# last: $a0 = a text, $a1 = a needle -> $v0 = the last place it starts, or
# -1 (it calls find: a frame; $s0 = the text, $s1 = the needle, $s2 = where
# to search from, $s3 = the latest place)
last:   addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28-16 $s0-$s3,
        sw    $ra, 36($sp)            # 0-15 the home area
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        sw    $s3, 16($sp)
        move  $fp, $sp
        move  $s0, $a0
        move  $s1, $a1
        move  $s2, $a0
        li    $s3, -1                 # none yet
lnext:  move  $a0, $s2
        move  $a1, $s1
        jal   find
        bltz  $v0, ldone
        addu  $s2, $s2, $v0           # the place itself
        subu  $s3, $s2, $s0           # the latest, as an offset
        addiu $s2, $s2, 1
        b     lnext
ldone:  move  $v0, $s3
        move  $sp, $fp
        lw    $s3, 16($sp)
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra

# find: episode 9's (a leaf)
find:   move  $t0, $a0
start:  move  $t1, $t0
        move  $t2, $a1
match:  lbu   $t4, 0($t2)
        beqz  $t4, found
        lbu   $t3, 0($t1)
        bne   $t3, $t4, again
        addiu $t1, $t1, 1
        addiu $t2, $t2, 1
        b     match
again:  lbu   $t3, 0($t0)
        beqz  $t3, notfound
        addiu $t0, $t0, 1
        b     start
found:  subu  $v0, $t0, $a0
        jr    $ra
notfound:
        li    $v0, -1
        jr    $ra
