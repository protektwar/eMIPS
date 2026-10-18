# crt0.S — the start of a C program on eMIPS (python3 -m emips cc).
#
# The boot firmware (kernel / user boot) already set $sp and the CPU state;
# this sets $gp, sets $sp again (the reset boot leaves it 0), clears .bss,
# calls main() and ends the program with main's return value as the exit
# code (the exit2 service, 17, as MARS).
        .set    noreorder
        .text
        .globl  __start
        .ent    __start
__start:
        la      $gp, _gp
        la      $sp, __stack_top
        la      $t0, __bss_start
        la      $t1, _end
1:      beq     $t0, $t1, 2f            # clear .bss, a word at a time
        nop
        sw      $zero, 0($t0)
        b       1b
        addiu   $t0, $t0, 4
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
