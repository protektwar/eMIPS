# MIPS by Example in eMIPS, episode 42, the RISC-V ending: merge sort on
# RV32.  mergesort keeps a, lo, hi and mid in s1-s4 (s0: the frame pointer)
# across its two calls; merge is a leaf with five pointers, as on MIPS.
# eMIPS: --arch rv32
        .data
array:  .word 38, 27, 43, 3, 9, 82, 10, 3
        .eqv  N, 8
tmp:    .space 4 * N                  # merge's room
before: .asciz "before: "
after:  .asciz "after:  "

        .text
main:   addi  sp, sp, -16
        sw    ra, 12(sp)
        sw    s0, 8(sp)
        addi  s0, sp, 16              # s0 = the frame pointer
        la    a0, before
        li    a7, 4
        ecall
        la    a0, array
        li    a1, N
        call  print_array
        la    a0, array               # mergesort(array, 0, N)
        li    a1, 0
        li    a2, N
        call  mergesort
        la    a0, after
        li    a7, 4
        ecall
        la    a0, array
        li    a1, N
        call  print_array
        lw    s0, 8(sp)
        lw    ra, 12(sp)
        addi  sp, sp, 16
        li    a7, 10
        ecall

# mergesort(a, lo, hi): sorts a[lo..hi-1]
mergesort:
        addi  sp, sp, -32             # 28 ra, 24 s0, 20 s1, 16 s2, 12 s3, 8 s4
        sw    ra, 28(sp)
        sw    s0, 24(sp)
        sw    s1, 20(sp)
        sw    s2, 16(sp)
        sw    s3, 12(sp)
        sw    s4, 8(sp)
        addi  s0, sp, 32
        mv    s1, a0                  # a
        mv    s2, a1                  # lo
        mv    s3, a2                  # hi
        sub   t0, s3, s2
        li    t1, 2
        blt   t0, t1, mdone           # 0 or 1 element: sorted already
        add   s4, s2, s3
        srli  s4, s4, 1               # mid = (lo + hi) / 2
        mv    a2, s4                  # mergesort(a, lo, mid)
        call  mergesort
        mv    a0, s1                  # mergesort(a, mid, hi)
        mv    a1, s4
        mv    a2, s3
        call  mergesort
        mv    a0, s1                  # merge(a, lo, mid, hi)
        mv    a1, s2
        mv    a2, s4
        mv    a3, s3
        call  merge
mdone:  lw    s4, 8(sp)
        lw    s3, 12(sp)
        lw    s2, 16(sp)
        lw    s1, 20(sp)
        lw    s0, 24(sp)
        lw    ra, 28(sp)
        addi  sp, sp, 32
        ret

# merge(a, lo, mid, hi) (a leaf)
merge:
        slli  t0, a1, 2
        add   t0, a0, t0              # i: the left half's front
        slli  t1, a2, 2
        add   t1, a0, t1              # j: the right half's front
        mv    t6, t1                  # the left half's end
        slli  t2, a3, 2
        add   t2, a0, t2              # the right half's end
        la    t3, tmp                 # k: the next place in tmp
both:   beq   t0, t6, rest            # one half empty: copy the other
        beq   t1, t2, rest
        lw    t4, 0(t0)
        lw    t5, 0(t1)
        bgt   t4, t5, right           # a[i] <= a[j]: the left one (stable)
        sw    t4, 0(t3)
        addi  t0, t0, 4
        j     moved
right:  sw    t5, 0(t3)
        addi  t1, t1, 4
moved:  addi  t3, t3, 4
        j     both
rest:   beq   t0, t6, rright          # what is left of the left half
        lw    t4, 0(t0)
        sw    t4, 0(t3)
        addi  t0, t0, 4
        addi  t3, t3, 4
        j     rest
rright: beq   t1, t2, back            # what is left of the right half
        lw    t4, 0(t1)
        sw    t4, 0(t3)
        addi  t1, t1, 4
        addi  t3, t3, 4
        j     rright
back:   slli  t0, a1, 2               # tmp back into a[lo..hi-1]
        add   t0, a0, t0
        la    t1, tmp
copy:   beq   t0, t2, mend
        lw    t4, 0(t1)
        sw    t4, 0(t0)
        addi  t0, t0, 4
        addi  t1, t1, 4
        j     copy
mend:   ret

# print_array(a, n) (a leaf)
print_array:
        mv    t0, a0
        slli  t1, a1, 2
        add   t1, t0, t1              # the end
pa:     lw    a0, 0(t0)
        li    a7, 1
        ecall
        addi  t0, t0, 4
        li    a0, ' '                 # a space between two, a new line after the last
        bne   t0, t1, pspace
        li    a0, '\n'
pspace: li    a7, 11
        ecall
        bne   t0, t1, pa
        ret
