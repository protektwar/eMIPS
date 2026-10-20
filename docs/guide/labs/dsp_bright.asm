# Brighten a 16-pixel greyscale image by 0x40, saturating at white (0xff):
# bright_dsp does four pixels per instruction (addu_s.qb), bright32 one
# byte at a time with a branch that clamps.  .eqv DSP chooses which one
# runs; both print the image after it.
        .eqv    DSP, 1              # 1: bright_dsp, 0: bright32
        .data
image:  .byte   0x00, 0x10, 0x40, 0x7f, 0x80, 0xa0, 0xbe, 0xbf
        .byte   0xc0, 0xc1, 0xe0, 0xf0, 0xfe, 0xff, 0x33, 0x99
        .text
main:   la      $a0, image
        li      $a1, 16
        li      $t0, DSP
        beqz    $t0, plain
        jal     bright_dsp
        b       show
plain:  jal     bright32
show:   la      $s0, image          # print the 16 pixels, 4 to a line
        li      $s1, 4
row:    lw      $a0, 0($s0)
        li      $v0, 34
        syscall
        li      $a0, '\n'
        li      $v0, 11
        syscall
        addiu   $s0, $s0, 4
        addiu   $s1, $s1, -1
        bnez    $s1, row
        li      $v0, 10
        syscall

# bright_dsp($a0 = image, $a1 = pixels, a multiple of 4)
bright_dsp:
        repl.qb $t1, 0x40           # 0x40 in every byte
dloop:  lw      $t0, 0($a0)         # four pixels
        addu_s.qb $t0, $t0, $t1     # each + 0x40, at most 0xff
        sw      $t0, 0($a0)
        addiu   $a0, $a0, 4
        addiu   $a1, $a1, -4
        bnez    $a1, dloop
        jr      $ra

# bright32($a0 = image, $a1 = pixels)
bright32:
        li      $t2, 0xff
bloop:  lbu     $t0, 0($a0)         # one pixel
        addiu   $t0, $t0, 0x40
        sltu    $t3, $t2, $t0       # above 0xff?
        beqz    $t3, store
        move    $t0, $t2            # (only when it is): clamp to white
store:  sb      $t0, 0($a0)
        addiu   $a0, $a0, 1
        addiu   $a1, $a1, -1
        bnez    $a1, bloop
        jr      $ra
