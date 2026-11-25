# MIPS by Example in eMIPS, episode 25 — A linked list with sbrk.  Each
# node is 8 bytes from the heap — sbrk (service 9) hands out memory and
# never takes it back: the value at 0, the address of the next node at 4,
# and 0 (null) after the last.  push puts a node in front, append walks to
# the end, reverse turns every next pointer around.
        .data
listw:  .asciiz "list:"
lenw:   .asciiz "length "
revw:   .asciiz "reversed:"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20 $s0, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        li    $s0, 0                  # head: the empty list
        move  $a0, $s0                # push 3, 7, 11: each in front
        li    $a1, 3
        jal   push
        move  $a0, $v0
        li    $a1, 7
        jal   push
        move  $a0, $v0
        li    $a1, 11
        jal   push
        move  $s0, $v0
        move  $a0, $s0                # append 20 at the end
        li    $a1, 20
        jal   append
        la    $a0, listw
        move  $a1, $s0
        jal   print_list
        li    $v0, 4                  # its length
        la    $a0, lenw
        syscall
        move  $a0, $s0
        jal   length
        move  $a0, $v0
        li    $v0, 1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $a0, $s0                # reversed in place
        jal   reverse
        move  $s0, $v0
        la    $a0, revw
        move  $a1, $s0
        jal   print_list
        move  $sp, $fp                # main's epilogue
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# push: $a0 = head, $a1 = a value -> $v0 = the new head: a node from sbrk
# in front (a leaf: a system call is not a call)
push:   move  $t0, $a0
        li    $a0, 8                  # sbrk(8): a node
        li    $v0, 9
        syscall                       # $v0 = its address
        sw    $a1, 0($v0)             # value
        sw    $t0, 4($v0)             # next: the old head
        jr    $ra

# append: $a0 = head (not empty), $a1 = a value: a new last node (a leaf)
append: lw    $t0, 4($a0)             # walk until next is null
        beqz  $t0, alast
        move  $a0, $t0
        b     append
alast:  move  $t1, $a0
        li    $a0, 8
        li    $v0, 9
        syscall
        sw    $a1, 0($v0)
        sw    $zero, 4($v0)           # the new last: next = null
        sw    $v0, 4($t1)             # the old last points at it
        jr    $ra

# length: $a0 = head -> $v0 = the number of nodes (a leaf)
length: li    $v0, 0
lnext:  beqz  $a0, ldone
        addiu $v0, $v0, 1
        lw    $a0, 4($a0)             # follow next
        b     lnext
ldone:  jr    $ra

# reverse: $a0 = head -> $v0 = the new head: each next pointer turned to
# the node before (a leaf)
reverse:
        li    $v0, 0                  # prev: none yet
rnext:  beqz  $a0, rdone
        lw    $t0, 4($a0)             # save next first ...
        sw    $v0, 4($a0)             # ... then point back at prev
        move  $v0, $a0                # prev = this node
        move  $a0, $t0                # on to the saved next
        b     rnext
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
