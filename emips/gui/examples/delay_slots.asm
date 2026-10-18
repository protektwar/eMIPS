# Branch delay slots (MIPS32 before Release 6): the instruction after a
# branch or jump ALWAYS executes.  With ".set noreorder" you fill the slot
# yourself; by default (".set reorder") the assembler puts a nop there.
        .set  noreorder
        .text
main:   li    $t0, 0
        li    $t1, 5
loop:   addiu $t1, $t1, -1
        bnez  $t1, loop
        addiu $t0, $t0, 10      # delay slot: runs on every iteration -> 50
        bnel  $t1, $zero, main  # branch-likely, not taken ($t1 = 0) ...
        addiu $t0, $t0, 1       # ... so its delay slot is squashed
        li    $v0, 1
        move  $a0, $t0
        syscall                 # prints 50
        li    $v0, 10
        syscall
