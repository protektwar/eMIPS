# MIPS by Example in eMIPS, episode 31 — A menu-driven program with dialogs.
# MARS's dialogs ask (51 InputDialogInt, 50 ConfirmDialog) and answer
# (56 MessageDialogInt, 55 MessageDialog); the console keeps a log.  The
# choice picks a function from a table of addresses: one jalr instead of a
# chain of branches.  Each action returns $v0 = 1 to go on, 0 to stop.
        .data
menu:   .asciiz "1 add a number\n2 the sum\n3 the average\n0 quit"
asknum: .asciiz "A number:"
sumis:  .asciiz "The sum: "
avgis:  .asciiz "The average: "
nonums: .asciiz "No numbers yet."
notnum: .asciiz "Not a number."
nosuch: .asciiz "No such choice."
really: .asciiz "Quit?"
added:  .asciiz "added "
sumlog: .asciiz "sum "
avglog: .asciiz "average "
bye:    .asciiz "bye\n"
count:  .word 0                       # how many numbers
total:  .word 0                       # their sum
actions:.word quit, add, sum, average # the table: choice 0, 1, 2, 3
        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 0-15 the home area
        sw    $ra, 28($sp)            # (jalr overwrites $ra too)
        sw    $fp, 24($sp)
        move  $fp, $sp
loop:   li    $v0, 51                 # the menu: InputDialogInt
        la    $a0, menu
        syscall
        li    $t0, -2
        beq   $a1, $t0, cancel        # -2: Cancel = quit
        bnez  $a1, nonumber           # -1 not a number, -3 empty
        sltiu $t0, $a0, 4             # the range check: 0 <= choice < 4
        beqz  $t0, nochoice
        sll   $t0, $a0, 2             # choice * 4: its place in the table
        lw    $t0, actions($t0)       # the function's address
        jalr  $t0                     # call it
        bnez  $v0, loop               # 1: the menu again
        b     done
cancel: jal   quit                    # Cancel asks too
        bnez  $v0, loop
        b     done
nonumber:
        li    $v0, 55                 # MessageDialog, a warning (2)
        la    $a0, notnum
        li    $a1, 2
        syscall
        b     loop
nochoice:
        li    $v0, 55
        la    $a0, nosuch
        li    $a1, 2
        syscall
        b     loop
done:   move  $sp, $fp                # main's epilogue
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# add: ask for a number, add it to total, count it (a leaf: system calls only)
add:    li    $v0, 51
        la    $a0, asknum
        syscall
        bnez  $a1, addnot             # not a number: nothing added
        lw    $t0, total
        addu  $t0, $t0, $a0
        sw    $t0, total
        lw    $t0, count
        addiu $t0, $t0, 1
        sw    $t0, count
        move  $t1, $a0                # the log: "added n"
        li    $v0, 4
        la    $a0, added
        syscall
        li    $v0, 1
        move  $a0, $t1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $v0, 1                  # go on
        jr    $ra
addnot: li    $v0, 55
        la    $a0, notnum
        li    $a1, 2
        syscall
        li    $v0, 1
        jr    $ra

# sum: show the total (a leaf)
sum:    li    $v0, 56                 # MessageDialogInt: the text and $a1
        la    $a0, sumis
        lw    $a1, total
        syscall
        li    $v0, 4                  # the log: "sum n"
        la    $a0, sumlog
        syscall
        li    $v0, 1
        lw    $a0, total
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $v0, 1
        jr    $ra

# average: total / count, or a message when there are no numbers (a leaf)
average:
        lw    $t1, count
        bnez  $t1, avgok
        li    $v0, 55                 # an information message (1)
        la    $a0, nonums
        li    $a1, 1
        syscall
        li    $v0, 1
        jr    $ra
avgok:  lw    $t0, total
        div   $t0, $t1
        mflo  $t2
        li    $v0, 56
        la    $a0, avgis
        move  $a1, $t2
        syscall
        li    $v0, 4                  # the log: "average n"
        la    $a0, avglog
        syscall
        li    $v0, 1
        move  $a0, $t2
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $v0, 1
        jr    $ra

# quit: ask; Yes (0) stops, No / Cancel goes on (a leaf)
quit:   li    $v0, 50                 # ConfirmDialog: 0 Yes, 1 No, 2 Cancel
        la    $a0, really
        syscall
        li    $v0, 1
        bnez  $a0, qdone              # not Yes: go on
        li    $v0, 4
        la    $a0, bye
        syscall
        li    $v0, 0                  # stop
qdone:  jr    $ra
