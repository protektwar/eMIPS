# MIPS by Example in eMIPS, episode 3, the challenge solved.  strlen_bottom
# tests at the bottom of the loop: one branch a round (and one test before
# the loop, for the empty string).  strlen_ptr has no counter: the length is
# the address after the 0, minus the start, minus 1.  count_caps counts A to
# Z without a branch per character: c - 'A' < 26, compared unsigned, is 1
# exactly for A ... Z (anything below 'A' wraps round to a huge number).
        .data
text:   .asciiz "MIPS by Example in eMIPS"
empty:  .asciiz ""
t1:     .asciiz "strlen_bottom: "
t2:     .asciiz "strlen_ptr: "
t3:     .asciiz "strlen_bottom(\"\"): "
t4:     .asciiz "strlen_ptr(\"\"): "
t5:     .asciiz "capitals: "
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        la    $a0, text               # strlen_bottom(text)
        jal   strlen_bottom
        la    $a0, t1
        move  $a1, $v0
        jal   print_result
        la    $a0, text               # strlen_ptr(text)
        jal   strlen_ptr
        la    $a0, t2
        move  $a1, $v0
        jal   print_result
        la    $a0, empty              # both with the empty string
        jal   strlen_bottom
        la    $a0, t3
        move  $a1, $v0
        jal   print_result
        la    $a0, empty
        jal   strlen_ptr
        la    $a0, t4
        move  $a1, $v0
        jal   print_result
        la    $a0, text               # count_caps(text)
        jal   count_caps
        la    $a0, t5
        move  $a1, $v0
        jal   print_result
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10                 # exit
        syscall

# strlen_bottom: $a0 = a string -> $v0 = its length (a leaf; the test at the
# bottom: addiu, addiu, lbu, bnez + its slot = 5 instructions a character)
strlen_bottom:
        move  $v0, $zero
        lbu   $t0, 0($a0)
        beqz  $t0, sbdone             # the empty string: no round at all
sbnext: addiu $a0, $a0, 1
        addiu $v0, $v0, 1
        lbu   $t0, 0($a0)
        bnez  $t0, sbnext             # one branch a round
sbdone: jr    $ra

# strlen_ptr: no counter (a leaf; lbu, addiu, bnez + its slot = 4 a character)
strlen_ptr:
        move  $v0, $a0                # p = the start
spnext: lbu   $t0, 0($v0)
        addiu $v0, $v0, 1             # p moves on, past the 0 too
        bnez  $t0, spnext
        subu  $v0, $v0, $a0           # (the 0's address + 1) - the start
        addiu $v0, $v0, -1
        jr    $ra

# count_caps: $a0 = a string -> $v0 = how many of A ... Z (a leaf)
count_caps:
        move  $v0, $zero
ccnext: lbu   $t0, 0($a0)
        beqz  $t0, ccdone
        addiu $a0, $a0, 1
        addiu $t1, $t0, -65           # c - 'A'
        sltiu $t1, $t1, 26            # unsigned: 1 for A ... Z, else 0
        addu  $v0, $v0, $t1           # counted without a branch
        b     ccnext
ccdone: jr    $ra

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
