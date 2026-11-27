# MIPS by Example in eMIPS, episode 41, the RISC-V ending: quicksort on RV32.
# The same functions; quicksort keeps a, lo, hi and p in s1-s4 (s0 is the
# frame pointer), saved in a frame of 32 bytes — a multiple of 16; the
# compare-and-branch instructions (bge, blt) are real instructions here, not
# macros, and there are no delay slots.
# eMIPS: --arch rv32
        .data
array:  .word 34, 7, 23, 32, 5, 62, 32, 7, -1, 18
        .eqv  N, 10
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
        la    a0, array               # quicksort(array, 0, N - 1)
        li    a1, 0
        li    a2, N - 1
        call  quicksort
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

# quicksort(a, lo, hi): sorts a[lo..hi]
quicksort:
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
        bge   s2, s3, qdone           # 0 or 1 element: sorted already
        call  partition               # p = partition(a, lo, hi)
        mv    s4, a0
        mv    a0, s1                  # quicksort(a, lo, p - 1)
        mv    a1, s2
        addi  a2, s4, -1
        call  quicksort
        mv    a0, s1                  # quicksort(a, p + 1, hi)
        addi  a1, s4, 1
        mv    a2, s3
        call  quicksort
qdone:  lw    s4, 8(sp)
        lw    s3, 12(sp)
        lw    s2, 16(sp)
        lw    s1, 20(sp)
        lw    s0, 24(sp)
        lw    ra, 28(sp)
        addi  sp, sp, 32
        ret

# partition(a, lo, hi) -> p (a leaf: Lomuto's scheme, with pointers)
partition:
        slli  t0, a2, 2
        add   t0, a0, t0              # &a[hi]
        lw    t1, 0(t0)               # the pivot
        slli  t2, a1, 2
        add   t2, a0, t2              # &a[i], i = lo
        mv    t3, t2                  # &a[j], j = lo
ploop:  beq   t3, t0, pend            # j < hi
        lw    t4, 0(t3)               # a[j]
        bge   t4, t1, pnext           # a[j] < pivot: swap a[i], a[j]
        lw    t5, 0(t2)
        sw    t4, 0(t2)
        sw    t5, 0(t3)
        addi  t2, t2, 4               # i++
pnext:  addi  t3, t3, 4               # j++
        j     ploop
pend:   lw    t5, 0(t2)               # the pivot to a[i]
        sw    t1, 0(t2)
        sw    t5, 0(t0)
        sub   a0, t2, a0              # p = i
        srai  a0, a0, 2
        ret

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
