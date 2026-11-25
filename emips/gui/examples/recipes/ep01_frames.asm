# MIPS by Example in eMIPS, episode 1 — Functions done right: the stack
# frame.  main calls describe for each word; describe calls strlen.  A
# function that calls another (or keeps values in $s registers or in
# memory) builds a frame on the stack: its prologue saves $ra, the caller's
# $fp and the $s registers it uses; its epilogue restores them.  strlen calls
# nobody and uses only $t / $a / $v registers: a leaf, it needs no frame.
        .data
words:  .word mips, emips, assembly, 0
mips:   .asciiz "MIPS"
emips:  .asciiz "eMIPS"
assembly: .asciiz "assembly"
has:    .asciiz " has "
chars:  .asciiz " characters\n"

        .text
# main: describe every word of the list
main:   addiu $sp, $sp, -32           # main's frame (a multiple of 8):
        sw    $ra, 28($sp)            #   28 $ra, 24 $fp, 20 $s0,
        sw    $fp, 24($sp)            #   16 (unused), 0-15 the home area
        sw    $s0, 20($sp)            #   of the functions main calls
        move  $fp, $sp
        la    $s0, words              # $s0: kept across the calls
next:   lw    $a0, 0($s0)             # the next word (0: the end)
        beqz  $a0, end
        jal   describe
        addiu $s0, $s0, 4
        b     next
end:    move  $sp, $fp                # main's epilogue (for symmetry: it exits)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10                 # exit
        syscall

# describe: $a0 = a string -> prints "<string> has <n> characters"
describe:
        addiu $sp, $sp, -32           # the prologue: the frame
        sw    $ra, 28($sp)            # jal strlen will overwrite $ra
        sw    $fp, 24($sp)            # the caller's frame pointer
        sw    $s0, 20($sp)            # the caller's $s0: we use $s0 too
        move  $fp, $sp                # our frame starts here
        move  $s0, $a0                # the string, kept across the call
        jal   strlen                  # strlen($a0)
        sw    $v0, 16($fp)            # a local variable: n = the length
        li    $v0, 4                  # print the string
        move  $a0, $s0
        syscall
        la    $a0, has
        syscall
        li    $v0, 1                  # print n
        lw    $a0, 16($fp)
        syscall
        li    $v0, 4
        la    $a0, chars
        syscall
        move  $sp, $fp                # the epilogue: restore, in reverse
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# strlen: $a0 = a string -> $v0 = its length (a leaf: no frame)
strlen: move  $v0, $zero
len:    lbu   $t0, 0($a0)
        beqz  $t0, lendone
        addiu $a0, $a0, 1
        addiu $v0, $v0, 1
        b     len
lendone:
        jr    $ra
