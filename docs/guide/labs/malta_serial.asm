# A serial driver for the Malta (run with --board malta): write a string
# to tty0, the console, polling the 16550's Line Status Register.
        .text
main:   li    $s0, 0xb80003f8       # tty0: ISA port 0x3f8
        la    $s1, msg
next:   lbu   $t0, 0($s1)
        beqz  $t0, done
wait:   lbu   $t1, 5($s0)           # LSR
        andi  $t1, $t1, 0x20        # THRE: the transmitter takes a character
        beqz  $t1, wait
        sb    $t0, 0($s0)           # THR
        addiu $s1, $s1, 1
        b     next
done:   lbu   $t1, 5($s0)
        andi  $t1, $t1, 0x40        # TEMT: the last bit has left
        beqz  $t1, done
        li    $v0, 10
        syscall

        .data
msg:    .asciiz "Hello from tty0!\n"
# eMIPS: --board malta    (the settings this program runs with)
