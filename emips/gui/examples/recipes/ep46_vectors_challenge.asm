# MIPS by Example in eMIPS, episode 46 — the challenge, solved: brighter
# pixels.  Sixteen grey pixels, one byte each, are one MSA register; adding
# 0x40 to all sixteen is one instruction.  addv.b wraps: 0xf0 + 0x40 is 0x30,
# a bright pixel turns dark.  adds_u.b saturates: anything past 255 stays
# 255, white.  (The episode: vectors with MSA.)
        .data
        .align 4                      # (16 bytes: a vector's own alignment)
pixels: .byte 0, 32, 64, 96, 128, 160, 192, 200, 208, 216, 224, 232, 240, 248, 250, 255
wrapped: .space 16
bright: .space 16
lpix:   .asciiz "pixels:     "
lwrap:  .asciiz "wrapping:   "
lsat:   .asciiz "saturating: "

        .text
main:   addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        la    $t0, pixels
        ld.b  $w0, 0($t0)             # all sixteen pixels
        ldi.b $w1, 0x40               # sixteen times 0x40
        addv.b $w2, $w0, $w1          # + 0x40, modulo 256
        la    $t1, wrapped
        st.b  $w2, 0($t1)
        adds_u.b $w3, $w0, $w1        # + 0x40, at most 255
        la    $t1, bright
        st.b  $w3, 0($t1)
        li    $v0, 4
        la    $a0, lpix
        syscall
        la    $a0, pixels
        jal   print_bytes
        li    $v0, 4
        la    $a0, lwrap
        syscall
        la    $a0, wrapped
        jal   print_bytes
        li    $v0, 4
        la    $a0, lsat
        syscall
        la    $a0, bright
        jal   print_bytes
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# print_bytes(p): the 16 bytes at p as numbers 0-255, and a new line (a leaf)
print_bytes:
        move  $t0, $a0
        addiu $t1, $t0, 16            # the end
pb:     lbu   $a0, 0($t0)
        li    $v0, 1
        syscall
        addiu $t0, $t0, 1
        li    $v0, 11
        li    $a0, ' '                # a space between two, a new line after the last
        bne   $t0, $t1, pbs
        li    $a0, '\n'
pbs:    syscall
        bne   $t0, $t1, pb
        jr    $ra
# eMIPS: --release R5
