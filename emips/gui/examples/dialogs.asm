# Dialogs (MARS's services 50-59): ask for a number and a name in input
# dialogs, answer in message dialogs, then ask whether to go on.  In the
# window they are real dialogs; on the command line the answers come from
# the program's input (!cancel = Cancel) and the dialogs are printed.
        .data
askage: .asciiz "How old are you?"
next:   .asciiz "Next year you will be "
askname:.asciiz "What is your name?"
hello:  .asciiz "Hello, "
again:  .asciiz "Shall I say it on the console too?"
name:   .space 32
        .text
main:   li    $v0, 51                 # InputDialogInt: $a0 the number, $a1 the status
        la    $a0, askage
        syscall
        bnez  $a1, nonum              # 0: a number (-1 not a number, -2 Cancel, -3 empty)
        addiu $a1, $a0, 1
        li    $v0, 56                 # MessageDialogInt: the text and $a1
        la    $a0, next
        syscall
nonum:  li    $v0, 54                 # InputDialogString: into name, at most 32 bytes
        la    $a0, askname
        la    $a1, name
        li    $a2, 32
        syscall
        li    $v0, 59                 # MessageDialogString: the text and the string
        la    $a0, hello
        la    $a1, name
        syscall
        li    $v0, 50                 # ConfirmDialog: 0 Yes, 1 No, 2 Cancel
        la    $a0, again
        syscall
        bnez  $a0, done
        li    $v0, 4                  # Yes: print it too
        la    $a0, hello
        syscall
        la    $a0, name               # (the name ends with its newline)
        syscall
done:   li    $v0, 10
        syscall
