# Lab: the same function twice, in MIPS32 and in MIPS16e (16-bit
# instructions).  Both sum the words of an array; main calls the MIPS32
# one with jal and the MIPS16e one with jalx (which switches the ISA),
# and prints both sums.  Compare their sizes in the listing (asm) and
# the instructions each executes (debug: break, then step).
        .data
nums:   .word   3, 1, 4, 1, 5, 9, 2, 6, 5, 3
        .text
main:   la      $a0, nums
        li      $a1, 10
        jal     sum32               # MIPS32 -> MIPS32
        move    $s0, $v0
        la      $a0, nums
        li      $a1, 10
        jalx    sum16               # MIPS32 -> MIPS16e
        move    $s1, $v0
        move    $a0, $s0
        li      $v0, 1
        syscall
        li      $a0, ' '
        li      $v0, 11
        syscall
        move    $a0, $s1
        li      $v0, 1
        syscall
        li      $v0, 10
        syscall

sum32:  move    $v0, $zero          # $a0 = the array, $a1 = how many words
loop32: lw      $t0, 0($a0)
        addu    $v0, $v0, $t0
        addiu   $a0, $a0, 4
        addiu   $a1, $a1, -1
        bnez    $a1, loop32
        jr      $ra

        .set mips16                 # 16-bit instructions from here
sum16:  li      $v0, 0
loop16: lw      $v1, 0($a0)         # only $s0, $s1, $v0, $v1, $a0-$a3 here
        addu    $v0, $v0, $v1
        addiu   $a0, 4
        addiu   $a1, -1
        bnez    $a1, loop16         # no delay slot: a compact branch
        jr      $ra                 # jrc: back to MIPS32 ($ra bit 0 = 0)
