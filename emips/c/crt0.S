# crt0.S — the start of a C program on eMIPS (python3 -m emips cc), o32 and
# n64.  The boot firmware already set $sp and the CPU state; this sets $gp,
# sets $sp again (the reset boot leaves it 0), clears .bss, calls main() and
# ends the program with main's return value as the exit code (the exit2
# service, 17, as MARS).  Keep the code on its lines: see crt0.h.
#include "crt0.h"
        .set    noreorder
        .text
        .globl  __start
        .ent    __start
__start:
        LA($gp, _gp)
        LA($sp, __stack_top)
        LA($t0, __bss_start)
        LA($t1, _end)
1:      beq     $t0, $t1, 2f            # clear .bss, a word at a time
        nop
        sw      $zero, 0($t0)
        b       1b
        PTR_ADDIU $t0, $t0, 4
2:      jal     main
        nop
        move    $a0, $v0                # exit(main())
        li      $v0, 17
        syscall
        li      $v0, 10                 # (without the exit services: stop here)
        syscall
3:      b       3b
        nop
        .end    __start
