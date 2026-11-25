# MIPS by Example in eMIPS, episode 25, the challenge solved.  insert puts
# a value into a sorted list where it belongs: walk while the next value is
# smaller, then link the new node in.  remove unlinks the first node with a
# value: the node before it skips over it.  Both return the head, which
# changes when the first node does.
        .data
values: .word 42, 7, 19, 3
sortw:  .asciiz "sorted:"
remw:   .asciiz "without 19:"
rem3w:  .asciiz "without 3:"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        li    $s0, 0                  # head: the empty list
        li    $s1, 0                  # i
inext:  la    $t0, values
        sll   $t1, $s1, 2
        addu  $t0, $t0, $t1
        move  $a0, $s0
        lw    $a1, 0($t0)
        jal   insert
        move  $s0, $v0
        addiu $s1, $s1, 1
        blt   $s1, 4, inext
        la    $a0, sortw
        move  $a1, $s0
        jal   print_list
        move  $a0, $s0                # remove 19: from the middle
        li    $a1, 19
        jal   remove
        move  $s0, $v0
        la    $a0, remw
        move  $a1, $s0
        jal   print_list
        move  $a0, $s0                # remove 3: the head itself
        li    $a1, 3
        jal   remove
        move  $s0, $v0
        la    $a0, rem3w
        move  $a1, $s0
        jal   print_list
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# insert: $a0 = head of a sorted list, $a1 = a value -> $v0 = the head
# (a leaf)
insert: move  $t0, $a0
        li    $a0, 8                  # the new node
        li    $v0, 9
        syscall
        sw    $a1, 0($v0)
        beqz  $t0, ifront             # an empty list ...
        lw    $t1, 0($t0)
        bgt   $t1, $a1, ifront        # ... or smaller than the first: in front
        move  $t2, $t0                # walk: $t2 = the node to insert after
iwalk:  lw    $t3, 4($t2)
        beqz  $t3, ilink              # the end
        lw    $t1, 0($t3)
        bgt   $t1, $a1, ilink         # the next one is larger: here
        move  $t2, $t3
        b     iwalk
ilink:  sw    $t3, 4($v0)             # new -> next
        sw    $v0, 4($t2)             # before -> new
        move  $v0, $t0                # the head did not change
        jr    $ra
ifront: sw    $t0, 4($v0)             # new -> the old head
        jr    $ra                     # the new node is the head

# remove: $a0 = head, $a1 = a value -> $v0 = the head without the first
# node holding it (a leaf; sbrk's memory is not given back)
remove: move  $v0, $a0
        beqz  $a0, rdone
        lw    $t0, 0($a0)
        bne   $t0, $a1, rwalk
        lw    $v0, 4($a0)             # the head itself: its next is the head
        jr    $ra
rwalk:  lw    $t1, 4($a0)             # $a0 = the node before
        beqz  $t1, rdone              # not there
        lw    $t0, 0($t1)
        beq   $t0, $a1, runlink
        move  $a0, $t1
        b     rwalk
runlink:lw    $t2, 4($t1)
        sw    $t2, 4($a0)             # skip over it
rdone:  jr    $ra

# print_list: the text at $a0, then the values from $a1 to null (a leaf)
print_list:
        li    $v0, 4
        syscall
pnext:  beqz  $a1, pdone
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        lw    $a0, 0($a1)
        syscall
        lw    $a1, 4($a1)
        b     pnext
pdone:  li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra
