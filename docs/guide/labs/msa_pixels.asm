# Brighten a 32-pixel greyscale image by 0x50 with MSA: sixteen pixels per
# instruction.  As handed out it uses addv.b, which wraps around: a bright
# pixel turns dark.  Fix it so every pixel stops at white (0xff), and run
# python3 -m emips check docs/guide/labs/msa_pixels.asm (msa_pixels.check).
        .data
        .align  4                   # (16 bytes: one vector)
image:  .byte   0x00, 0x10, 0x20, 0x40, 0x60, 0x80, 0x9f, 0xa0
        .byte   0xaf, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0, 0xfe, 0xff
        .byte   0x05, 0x15, 0x25, 0x35, 0x45, 0x55, 0x65, 0x75
        .byte   0x85, 0x95, 0xa5, 0xb5, 0xc5, 0xd5, 0xe5, 0xf5
        .text
main:   la      $a0, image
        ldi.b   $w1, 0x50           # 0x50 in all sixteen bytes
        ld.b    $w0, 0($a0)         # pixels 0-15
        addv.b  $w0, $w0, $w1       # + 0x50 each (wraps around past 0xff)
        st.b    $w0, 0($a0)
        ld.b    $w0, 16($a0)        # pixels 16-31
        addv.b  $w0, $w0, $w1
        st.b    $w0, 16($a0)
        la      $s0, image          # print the image, 4 pixels (a word) a line
        li      $s1, 8
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
# eMIPS: --release R5
