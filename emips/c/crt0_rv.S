# crt0_rv.S — the start of a RISC-V C program on eMIPS (python3 -m emips
# cc --arch rv32 / rv64).  The boot firmware set the hart's state (M-mode,
# mstatus.FS); this sets gp (__global_pointer$: the linker relaxed the
# program's gp-relative accesses against it — loaded without relaxation, as
# GNU's crt0 does) and sp, clears .bss, calls main() and ends the program
# with main's return value (in a0) as the exit code: RARS's Exit2 (93).
# Keep the code on its lines: the prebuilt executables' line tables name them.
        .text
        .globl  _start
        .type   _start, @function
_start:
        .option push
        .option norelax
        la      gp, __global_pointer$
        .option pop
        la      sp, __stack_top
        la      t0, __bss_start
        la      t1, _end
1:      bgeu    t0, t1, 2f              # clear .bss, a byte at a time
        sb      zero, 0(t0)
        addi    t0, t0, 1
        j       1b
2:      call    main
        li      a7, 93                  # exit(main())
        ecall
        li      a7, 10                  # (without the exit services: stop here)
        ecall
3:      j       3b
        .size   _start, . - _start
