# Lab: one microMIPS source, two releases.  Assemble it for Release 5
# (python3 -m emips asm labs/micromips_r6.asm --release R5) and for
# Release 6 (its header): the same instructions, other encodings.
# Release 6 microMIPS has no delay slots at all.  Both print 39 (and
# the largest word, 9, stays in $a2).
        .set micromips
        .data
nums:   .word   3, 1, 4, 1, 5, 9, 2, 6, 5, 3
        .text
main:   la      $a0, nums
        li      $a1, 10
        bal     sum                 # R5: bgezal + a 32-bit nop; R6: balc
        move    $a0, $v0
        li      $v0, 1
        syscall
        li      $v0, 10
        syscall

sum:    li      $v0, 0
        li      $a2, 0              # the biggest so far
loop:   lw      $a3, 0($a0)
        addu    $v0, $v0, $a3
        slt     $v1, $a2, $a3       # (a register of the 16-bit forms)
        beqz    $v1, small          # R5: beqzc (32 bits); R6: beqzc16
        move    $a2, $a3
small:  addiu   $a0, $a0, 4
        addiu   $a1, $a1, -1
        bne     $a1, $zero, loop    # R5: bnezc (32 bits); R6: bnezc16
        jr      $ra                 # R5: jrc; R6: jrc16
# eMIPS: --release R6
