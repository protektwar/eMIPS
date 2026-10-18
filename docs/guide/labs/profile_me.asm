# Where does the time go?  Profile it with the probe's PC sampling:
#   probe on / probe profile on / probe rate 0 / c / probe profile
        .text
main:   li    $a0, 100
        jal   count                  # a short count
        li    $a0, 2000
        jal   squares                # a long one
        li    $v0, 10
        syscall

count:  addiu $a0, $a0, -1
        bnez  $a0, count
        jr    $ra

squares:
        move  $v0, $zero
sq:     mul   $t0, $a0, $a0
        addu  $v0, $v0, $t0
        addiu $a0, $a0, -1
        bnez  $a0, sq
        jr    $ra
