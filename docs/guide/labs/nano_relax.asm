# Lab 65: what the linker does.  Sum a table with a helper in another
# section, then report through a routine in kernel memory.
        .linkrelax
        .text
main:   li      $a0, 0
        li      $a1, 10
        la      $a2, table
loop:   lw      $a3, 0($a2)
        balc    add                 # a call into .text.lib
        addiu   $a2, $a2, 4
        addiu   $a1, $a1, -1
        bnezc   $a1, loop
        sw      $a0, total          # a $gp-relative store
        balc    report              # a call 2 GB away
        li      $t4, 10
        syscall

        .section .text.lib, "ax"
add:    addu    $a0, $a0, $a3
        jrc     $ra

        .ktext
report: lw      $a0, total
        li      $t4, 1
        syscall
        jrc     $ra

        .sdata
table:  .word   1, 2, 3, 4, 5, 6, 7, 8, 9, 10
total:  .word   0
# eMIPS: --arch nanomips
