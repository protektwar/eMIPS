# MIPS by Example in eMIPS, episode 6 — Reading a line; upper and lower
# case: read_string (service 8) stores the line with its newline; chomp
# removes it.  upper and lower change only the letters: a capital and its
# small letter are 32 apart in ASCII.
        .data
buf:    .space 64                     # the line: up to 63 characters + the 0
ask:    .asciiz "Your name? "
hello:  .asciiz "Hello, "
bang:   .asciiz "!\n"
up:     .asciiz "UPPER: "
low:    .asciiz "\nlower: "
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        li    $v0, 4                  # ask
        la    $a0, ask
        syscall
        li    $v0, 8                  # read_string(buf, 64)
        la    $a0, buf
        li    $a1, 64
        syscall
        la    $a0, buf                # chomp(buf): the newline away
        jal   chomp
        li    $v0, 4                  # Hello, <name>!
        la    $a0, hello
        syscall
        la    $a0, buf
        syscall
        la    $a0, bang
        syscall
        la    $a0, buf                # upper(buf), printed
        jal   upper
        li    $v0, 4
        la    $a0, up
        syscall
        la    $a0, buf
        syscall
        la    $a0, buf                # lower(buf), printed
        jal   lower
        li    $v0, 4
        la    $a0, low
        syscall
        la    $a0, buf
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10                 # exit
        syscall

# chomp: cut the string at $a0 at its newline, if it has one (a leaf)
chomp:  lbu   $t0, 0($a0)
        beqz  $t0, chdone             # the end, and no newline
        li    $t1, '\n'
        beq   $t0, $t1, cut
        addiu $a0, $a0, 1
        b     chomp
cut:    sb    $zero, 0($a0)           # the newline becomes the end
chdone: jr    $ra

# upper: a-z -> A-Z in the string at $a0, in place (a leaf)
upper:  lbu   $t0, 0($a0)
        beqz  $t0, updone
        addiu $t1, $t0, -97           # a small letter? c - 'a' < 26
        sltiu $t1, $t1, 26            # (unsigned: episode 3's test)
        beqz  $t1, upnext
        addiu $t0, $t0, -32           # 'a' - 32 = 'A'
        sb    $t0, 0($a0)
upnext: addiu $a0, $a0, 1
        b     upper
updone: jr    $ra

# lower: A-Z -> a-z in the string at $a0, in place (a leaf)
lower:  lbu   $t0, 0($a0)
        beqz  $t0, lodone
        addiu $t1, $t0, -65           # a capital? c - 'A' < 26
        sltiu $t1, $t1, 26
        beqz  $t1, lonext
        addiu $t0, $t0, 32            # 'A' + 32 = 'a'
        sb    $t0, 0($a0)
lonext: addiu $a0, $a0, 1
        b     lower
lodone: jr    $ra
