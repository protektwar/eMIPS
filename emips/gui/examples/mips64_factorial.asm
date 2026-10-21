# MIPS64: 64-bit registers and the doubleword instructions.  fact(20) =
# 2432902008176640000 does not fit in 32 bits: daddiu / dmultu / sd / ld
# work on whole 64-bit registers (the Registers pane shows 16 digits; the
# names are the n64 ABI's: $a4-$a7, $t0-$t3).
        .text
main:   li      $a0, 20
        jal     fact              # $v0 = fact(20)
        move    $a0, $v0
        li      $v0, 1            # print_int prints the whole register
        syscall
        li      $v0, 10
        syscall

# long fact(long n): a loop; DMULTU's LO holds the low 64 bits of n * fact
fact:   daddiu  $sp, $sp, -16     # a 16-byte frame (the n64 stack alignment)
        sd      $s0, 0($sp)       # a doubleword store
        li      $v0, 1
        move    $s0, $a0
loop:   blez    $s0, done
        dmultu  $v0, $s0
        mflo    $v0
        daddiu  $s0, $s0, -1
        b       loop
done:   ld      $s0, 0($sp)       # a doubleword load
        daddiu  $sp, $sp, 16
        jr      $ra
# eMIPS: --arch mips64
