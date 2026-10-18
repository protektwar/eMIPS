# Lab 28: a program that calls the debugger three times (sdbbp).  The
# debug handler is not in it: the probe serves it from its own memory
# (labs/probe_handler.asm, at the debug vector 0xff200200 in dmseg).
        .text
main:   li    $s0, 3
loop:   sdbbp                        # -> the probe's handler
        addiu $s0, $s0, -1
        bnez  $s0, loop
        li    $v0, 10
        syscall
