# MIPS by Example in eMIPS, episode 26 — Balanced brackets with the stack.
# Each opening bracket is pushed on the real stack ($sp down 4, sw); each
# closing one pops the last open one and must match it.  How many are
# pushed depends on the text, so the epilogue restores $sp from $fp.
        .data
t1:     .asciiz "(a[b]{c})"
t2:     .asciiz "(a[b)c]"
t3:     .asciiz "((x)"
t4:     .asciiz "x)"
tests:  .word t1, t2, t3, t4
okw:    .asciiz " balanced\n"
errw:   .asciiz " error at "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $s0, tests
        li    $s1, 4
tnext:  lw    $a0, 0($s0)             # the text
        jal   balanced
        move  $t0, $v0
        li    $v0, 4                  # print it ...
        lw    $a0, 0($s0)
        syscall
        bgez  $t0, terr               # ... and the verdict
        li    $v0, 4
        la    $a0, okw
        syscall
        b     tdone
terr:   li    $v0, 4
        la    $a0, errw
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
tdone:  addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, tnext
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# balanced: $a0 = a text -> $v0 = -1 if its brackets match, else the
# index of the first problem (the text's length when some are left open).
# The open brackets go on the stack, below this function's frame.
balanced:
        addiu $sp, $sp, -24           # the usual frame: 20 $ra, 16 $fp
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp                # the stack is empty when $sp = $fp
        li    $v0, -1
        move  $t9, $a0                # the start, for the index
bnext:  lbu   $t0, 0($a0)
        beqz  $t0, bend
        beq   $t0, '(', bpush
        beq   $t0, '[', bpush
        beq   $t0, '{', bpush
        beq   $t0, ')', bclose
        beq   $t0, ']', bclose
        beq   $t0, '}', bclose
        b     bstep                   # not a bracket
bpush:  addiu $sp, $sp, -4            # push it
        sw    $t0, 0($sp)
        b     bstep
bclose: beq   $sp, $fp, bbad          # nothing open: an extra closing one
        lw    $t1, 0($sp)             # pop the last open one
        addiu $sp, $sp, 4
        addiu $t2, $t0, -1            # ')' - 1 = '('
        beq   $t0, ')', bmatch
        addiu $t2, $t0, -2            # ']' - 2 = '[', '}' - 2 = '{'
bmatch: bne   $t1, $t2, bbad          # the wrong kind
bstep:  addiu $a0, $a0, 1
        b     bnext
bend:   beq   $sp, $fp, bdone         # the end: anything still open?
bbad:   subu  $v0, $a0, $t9           # the problem's index
bdone:  move  $sp, $fp                # whatever is left on the stack: gone
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        jr    $ra
