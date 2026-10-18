# Console input and output with syscalls ($v0 = service number).
# The program waits for a number: type it in the Console input line.
        .data
ask:    .asciiz "Enter a number: "
res:    .asciiz "squared = "
        .text
main:   li    $v0, 4            # print_string($a0)
        la    $a0, ask
        syscall
        li    $v0, 5            # read_int -> $v0
        syscall
        mul   $t0, $v0, $v0
        li    $v0, 4
        la    $a0, res
        syscall
        li    $v0, 1            # print_int($a0)
        move  $a0, $t0
        syscall
        li    $v0, 11           # print_char($a0)
        li    $a0, 10           # '\n'
        syscall
        li    $v0, 10           # exit
        syscall
