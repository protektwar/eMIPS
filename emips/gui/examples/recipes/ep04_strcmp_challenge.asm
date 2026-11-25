# MIPS by Example in eMIPS, episode 4, the challenge solved.  strncmp
# compares at most n characters: n reaching 0 ends the walk with "equal".
# strcasecmp compares ignoring case: each byte is folded to a small letter
# first — 32 added to A ... Z, chosen without a branch by episode 3's
# unsigned test: ((c - 'A') < 26) << 5 is 32 for a capital, else 0.
        .data
apple:  .asciiz "apple"
apply:  .asciiz "apply"
zebra:  .asciiz "Zebra"
mipsc:  .asciiz "MIPS"
mipss:  .asciiz "mips"
t1:     .asciiz "strncmp(apple, apply, 4): "
t2:     .asciiz "strncmp(apple, apply, 5): "
t3:     .asciiz "strcasecmp(Zebra, apple): "
t4:     .asciiz "strcasecmp(MIPS, mips): "
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        la    $a0, apple              # strncmp(apple, apply, 4)
        la    $a1, apply
        li    $a2, 4
        jal   strncmp
        la    $a0, t1
        move  $a1, $v0
        jal   print_sign
        la    $a0, apple              # strncmp(apple, apply, 5)
        la    $a1, apply
        li    $a2, 5
        jal   strncmp
        la    $a0, t2
        move  $a1, $v0
        jal   print_sign
        la    $a0, zebra              # strcasecmp(Zebra, apple)
        la    $a1, apple
        jal   strcasecmp
        la    $a0, t3
        move  $a1, $v0
        jal   print_sign
        la    $a0, mipsc              # strcasecmp(MIPS, mips)
        la    $a1, mipss
        jal   strcasecmp
        la    $a0, t4
        move  $a1, $v0
        jal   print_sign
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10                 # exit
        syscall

# strncmp: $a0, $a1 = two strings, $a2 = n -> $v0 < 0, 0 or > 0 (a leaf)
strncmp:
        move  $v0, $zero              # n = 0: equal
snnext: beqz  $a2, sndone             # n characters compared: equal
        lbu   $t0, 0($a0)
        lbu   $t1, 0($a1)
        subu  $v0, $t0, $t1
        bnez  $v0, sndone             # different: they decide
        beqz  $t0, sndone             # both ended: 0
        addiu $a0, $a0, 1
        addiu $a1, $a1, 1
        addiu $a2, $a2, -1            # one character fewer to go
        b     snnext
sndone: jr    $ra

# strcasecmp: $a0, $a1 = two strings -> $v0 < 0, 0 or > 0, ignoring case
# (a leaf)
strcasecmp:
        lbu   $t0, 0($a0)
        lbu   $t1, 0($a1)
        addiu $t2, $t0, -65           # a capital? (unsigned: c - 'A' < 26)
        sltiu $t2, $t2, 26
        sll   $t2, $t2, 5             # 32 if so, else 0
        addu  $t0, $t0, $t2           # 'A' + 32 = 'a'
        addiu $t3, $t1, -65           # the same for b's byte
        sltiu $t3, $t3, 26
        sll   $t3, $t3, 5
        addu  $t1, $t1, $t3
        bne   $t0, $t1, scdiff        # different: they decide
        beqz  $t0, scdiff             # both ended: 0
        addiu $a0, $a0, 1
        addiu $a1, $a1, 1
        b     strcasecmp
scdiff: subu  $v0, $t0, $t1
        jr    $ra

# print_sign: $a0 = a text, $a1 = a result -> prints the text and <, = or >
# (a leaf: only system calls)
print_sign:
        li    $v0, 4
        syscall
        li    $a0, '='
        beqz  $a1, psout
        li    $a0, '<'
        bltz  $a1, psout
        li    $a0, '>'
psout:  li    $v0, 11
        syscall
        li    $a0, '\n'
        syscall
        jr    $ra
