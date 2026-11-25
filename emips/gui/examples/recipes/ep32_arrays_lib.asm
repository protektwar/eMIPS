# MIPS by Example in eMIPS, episode 32 — the library: arrays of words.
# A program takes these functions with .include "ep32_arrays_lib.asm".
# Each follows o32, the contract every caller counts on: arguments in
# $a0-$a3, the result in $v0, $s0-$s7, $fp, $sp and $ra as they were.
#   arr_sum(a, n) -> $v0      the sum of the n words at a
#   arr_max(a, n) -> $v0      the largest of them (n >= 1)
#   arr_print(a, n)           prints them: [2, 3, 5]
#   arr_reverse(a, n)         reverses them in place
# Its labels start with its own letters (as..., am..., ap..., ar...): a
# program's labels must not collide with them.
        .text
arr_sum:li    $v0, 0                  # a leaf: no frame
assum:  beqz  $a1, asdone
        lw    $t0, 0($a0)
        addu  $v0, $v0, $t0
        addiu $a0, $a0, 4
        addiu $a1, $a1, -1
        b     assum
asdone: jr    $ra

arr_max:lw    $v0, 0($a0)             # the first one: the largest so far
amnext: addiu $a1, $a1, -1
        beqz  $a1, amdone
        addiu $a0, $a0, 4
        lw    $t0, 0($a0)
        ble   $t0, $v0, amnext
        move  $v0, $t0                # a larger one
        b     amnext
amdone: jr    $ra

arr_print:                            # it calls ap_word: a frame
        addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 16 $s1
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        move  $s0, $a0                # the next word: kept across the calls
        move  $s1, $a1                # how many are left
        li    $v0, 11
        li    $a0, '['
        syscall
apnext: beqz  $s1, apdone
        lw    $a0, 0($s0)
        jal   ap_word
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        beqz  $s1, apdone
        li    $v0, 11                 # ", " between two
        li    $a0, ','
        syscall
        li    $a0, ' '
        syscall
        b     apnext
apdone: li    $v0, 11
        li    $a0, ']'
        syscall
        move  $sp, $fp
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

ap_word:li    $v0, 1                  # print_int $a0 (a leaf)
        syscall
        jr    $ra

arr_reverse:                          # two ends moving in (a leaf)
        sll   $t0, $a1, 2
        addu  $a1, $a0, $t0
        addiu $a1, $a1, -4            # the last word
arnext: bgeu  $a0, $a1, ardone        # they met
        lw    $t0, 0($a0)
        lw    $t1, 0($a1)
        sw    $t1, 0($a0)
        sw    $t0, 0($a1)
        addiu $a0, $a0, 4
        addiu $a1, $a1, -4
        b     arnext
ardone: jr    $ra
