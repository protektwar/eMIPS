# Lab: the same function twice, in MIPS32 and in microMIPS (16- and
# 32-bit instructions).  Both sum the words of an array; main calls the
# MIPS32 one with jal and the microMIPS one with jalx (which switches the
# ISA), and prints both sums.  Compare their sizes in the listing (asm)
# and the instructions each executes (debug: break, then step).
        .data
nums:   .word   3, 1, 4, 1, 5, 9, 2, 6, 5, 3
        .text
main:   la      $a0, nums
        li      $a1, 10
        jal     sum32               # MIPS32 -> MIPS32
        move    $s0, $v0
        la      $a0, nums
        li      $a1, 10
        jalx    summm               # MIPS32 -> microMIPS
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

        .set micromips              # microMIPS from here: the same source
summm:  move    $v0, $zero
loopmm: lw      $t0, 0($a0)         # $t0: no 16-bit form (8 registers only)
        addu    $v0, $v0, $t0
        addiu   $a0, $a0, 4
        addiu   $a1, $a1, -1
        bnez    $a1, loopmm         # bnezc: compact, no delay slot
        jr      $ra                 # jrc: back to MIPS32 ($ra bit 0 = 0)
# eMIPS: --release R5
