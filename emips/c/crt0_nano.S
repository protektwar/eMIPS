# crt0_nano.S — the start of a nanoMIPS C program on eMIPS (python3 -m
# emips cc --arch nanomips; the p32 ABI, DN00179).  The boot firmware set
# the CPU state; this sets $gp (the start of the small data: the p32 ABI's
# gp-relative offsets are unsigned) and $sp, clears .bss, calls main() and
# ends the program with main's return value — already in $a0, where the
# p32 ABI returns it — as the exit code (the exit2 service, 17, in $t4 =
# $2: the MARS services keep their register numbers).  Keep the code on
# its lines: the prebuilt executables' line tables name them.
        .text
        .globl  __start
        .ent    __start
__start:
        la      $gp, _gp
        la      $sp, __stack_top
        la      $t0, __bss_start
        la      $t1, _end
1:      beqc    $t0, $t1, 2f            # clear .bss, a word at a time
        sw      $zero, 0($t0)
        addiu   $t0, $t0, 4
        bc      1b
2:      balc    main
        li      $t4, 17                 # exit(main())
        syscall
        li      $t4, 10                 # (without the exit services: stop here)
        syscall
3:      bc      3b
        .end    __start
