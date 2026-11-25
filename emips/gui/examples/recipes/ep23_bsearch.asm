# MIPS by Example in eMIPS, episode 23 — Binary search, loop and recursive.
# In a sorted array, look at the middle: smaller than x, the answer is to
# its right; larger, to its left.  Each probe halves what is left, so 15
# numbers need at most 4 probes.  bsearch loops; bsearch_r calls itself on
# the half — a frame per call.
        .data
nums:   .word 2, 5, 8, 12, 16, 23, 38, 45, 56, 72, 91, 99, 104, 120, 150
wanted: .word 23, 91, 7
indexw: .asciiz ": index "
probew: .asciiz ", probes "
recw:   .asciiz "recursively:"
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        la    $s0, wanted             # each wanted number, by the loop
        li    $s1, 3
lnext:  la    $a0, nums
        li    $a1, 15
        lw    $a2, 0($s0)
        jal   bsearch
        move  $t0, $v0
        move  $t1, $v1
        li    $v0, 1                  # "x: index i, probes p"
        lw    $a0, 0($s0)
        syscall
        li    $v0, 4
        la    $a0, indexw
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 4
        la    $a0, probew
        syscall
        li    $v0, 1
        move  $a0, $t1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, lnext
        li    $v0, 4                  # and recursively
        la    $a0, recw
        syscall
        la    $s0, wanted
        li    $s1, 3
rnext:  la    $a0, nums
        li    $a1, 0                  # lo
        li    $a2, 14                 # hi
        lw    $a3, 0($s0)
        jal   bsearch_r
        move  $t0, $v0
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, rnext
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# bsearch: $a0 = sorted words, $a1 = n, $a2 = x -> $v0 = x's index or -1,
# $v1 = the probes (a leaf)
bsearch:
        li    $t0, 0                  # lo
        addiu $t1, $a1, -1            # hi
        li    $v1, 0
bloop:  bgt   $t0, $t1, bnone         # nothing left between lo and hi
        addu  $t2, $t0, $t1
        srl   $t2, $t2, 1             # mid = (lo + hi) / 2
        sll   $t3, $t2, 2
        addu  $t3, $a0, $t3
        lw    $t3, 0($t3)             # a[mid]
        addiu $v1, $v1, 1             # one more probe
        beq   $t3, $a2, bfound
        blt   $t3, $a2, bright
        addiu $t1, $t2, -1            # x is smaller: hi = mid - 1
        b     bloop
bright: addiu $t0, $t2, 1             # x is larger: lo = mid + 1
        b     bloop
bfound: move  $v0, $t2
        jr    $ra
bnone:  li    $v0, -1
        jr    $ra

# bsearch_r: $a0 = sorted words, $a1 = lo, $a2 = hi, $a3 = x -> $v0 = x's
# index or -1; it calls itself on the half x can be in (a frame each call)
bsearch_r:
        addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        li    $v0, -1
        bgt   $a1, $a2, rdone         # nothing left: -1
        addu  $t2, $a1, $a2
        srl   $t2, $t2, 1             # mid
        sll   $t3, $t2, 2
        addu  $t3, $a0, $t3
        lw    $t3, 0($t3)
        move  $v0, $t2
        beq   $t3, $a3, rdone         # found: mid
        blt   $t3, $a3, rright
        addiu $a2, $t2, -1            # the left half: lo .. mid - 1
        jal   bsearch_r
        b     rdone
rright: addiu $a1, $t2, 1             # the right half: mid + 1 .. hi
        jal   bsearch_r
rdone:  move  $sp, $fp
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        jr    $ra
