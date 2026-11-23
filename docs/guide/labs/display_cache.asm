# Lab 91 — The display and the caches: this program paints a 32 x 32-word
# gradient at $gp, through write-back data caches (--caches).  The display
# reads physical memory below the caches, as a real frame buffer does.
# Run:  python3 -m emips debug docs/guide/labs/display_cache.asm
#       (emips) run      then: display, cache stats
# Then set FLUSH to 1 and run it again.
        .eqv  FLUSH, 0
        .text
main:   move  $t0, $gp
        li    $t1, 0                  # word number 0 .. 1023
fill:   srl   $t2, $t1, 5
        sll   $t2, $t2, 19            # red = row * 8
        andi  $t3, $t1, 31
        sll   $t3, $t3, 3             # blue = column * 8
        or    $t2, $t2, $t3
        ori   $t2, $t2, 0x4000        # and a little green
        sw    $t2, 0($t0)
        addiu $t0, $t0, 4
        addiu $t1, $t1, 1
        blt   $t1, 1024, fill
        .if FLUSH
        move  $t0, $gp                # write every line of the picture back
        addiu $t1, $gp, 4096
wb:     cache 0x19, 0($t0)            # Hit Writeback D
        addiu $t0, $t0, 32
        bne   $t0, $t1, wb
        sync
        .endif
        li    $v0, 10
        syscall
# eMIPS: --display 256x256:8@gp --caches
