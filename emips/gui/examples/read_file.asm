# Files: open hello.txt (given with the program: see the last line), read
# it in pieces of 32 bytes and print them.  Settings > Exceptions & I/O >
# Files lists the files a program may open; the Files pane shows them.
        .data
name:   .asciiz "hello.txt"
buf:    .space 33
        .text
main:   li    $v0, 13                 # open: $a0 the name, $a1 0 = read
        la    $a0, name
        li    $a1, 0
        syscall
        move  $s0, $v0                # the descriptor (-1: not there)
more:   li    $v0, 14                 # read: up to 32 bytes into buf
        move  $a0, $s0
        la    $a1, buf
        li    $a2, 32
        syscall
        blez  $v0, done               # 0: the end of the file
        sb    $zero, buf($v0)         # end the piece with a NUL
        li    $v0, 4                  # and print it
        la    $a0, buf
        syscall
        b     more
done:   li    $v0, 16                 # close
        move  $a0, $s0
        syscall
        li    $v0, 10
        syscall
# eMIPS: --file hello.txt
