# Lab 66: a guard with the MPU.  fill writes the numbers 0, 1, 2 ... into
# buf, sixteen words — but its loop runs once too often and overwrites
# secret.  Make the MPU catch it: region 0 must forbid writes to secret.
        .data
buf:    .space  64                  # 16 words
secret: .word   1234
        .text
main:   # (set up MPU region 0 here)

        la      $a0, buf
        li      $a1, 0
fill:   sw      $a1, 0($a0)
        addiu   $a0, $a0, 4
        addiu   $a1, $a1, 1
        bltic   $a1, 17, fill       # 17 words: one too many
        lw      $a0, secret
        li      $t4, 1              # print_int: what is left of secret
        syscall
        li      $t4, 10
        syscall

        .ktext                      # the exception handler (0x80000180):
handler:                            # prints Cause.ExcCode and BadVAddr, stops
        mfc0    $k1, $13
        ext     $a0, $k1, 2, 5
        li      $t4, 1
        syscall
        li      $a0, ' '
        li      $t4, 11
        syscall
        mfc0    $a0, $8
        li      $t4, 34
        syscall
        li      $t4, 10
        syscall
# eMIPS: --arch nanomips --mmu mpu:8
