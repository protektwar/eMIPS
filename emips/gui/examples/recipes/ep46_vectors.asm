# MIPS by Example in eMIPS, episode 46 — Vectors: MSA from assembly and C.
# MSA, the MIPS SIMD Architecture (Release 5 and 6), adds 32 registers of
# 128 bits, $w0-$w31: four words side by side.  ld.w loads four words, addv.w
# adds four pairs at once, st.w stores four: c = a + b in two steps of four
# instead of eight of one.  A sum is four sums in the four lanes, added
# together at the end with copy_s.w.
        .data
        .eqv  N, 8                    # words in each array (a multiple of 4)
        .align 4                      # (16 bytes: a vector's own alignment)
a:      .word 1, 2, 3, 4, 5, 6, 7, 8
b:      .word 10, 20, 30, 40, 50, 60, 70, 80
c:      .space 4 * N
lc:     .asciiz "c = a + b: "
lsum:   .asciiz "sum of c: "

        .text
main:   addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        la    $t0, a                  # c = a + b, four words a step
        la    $t1, b
        la    $t2, c
        li    $t3, N / 4              # steps
add4:   ld.w  $w0, 0($t0)             # a[i..i+3]
        ld.w  $w1, 0($t1)             # b[i..i+3]
        addv.w $w2, $w0, $w1          # four additions at once
        st.w  $w2, 0($t2)             # c[i..i+3]
        addiu $t0, $t0, 16
        addiu $t1, $t1, 16
        addiu $t2, $t2, 16
        addiu $t3, $t3, -1
        bnez  $t3, add4
        li    $v0, 4                  # "c = a + b: 11 22 ..."
        la    $a0, lc
        syscall
        la    $a0, c
        li    $a1, N
        jal   print_array
        la    $a0, c                  # sum(c, N)
        li    $a1, N
        jal   sum
        move  $t0, $v0
        li    $v0, 4
        la    $a0, lsum
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# sum(v, n) -> $v0: four sums in the lanes of $w3, then the lanes added
# (a leaf; n a multiple of 4)
sum:    ldi.w $w3, 0                  # four sums, all 0
sloop:  ld.w  $w4, 0($a0)
        addv.w $w3, $w3, $w4          # each lane adds its own words
        addiu $a0, $a0, 16
        addiu $a1, $a1, -4
        bnez  $a1, sloop
        copy_s.w $v0, $w3[0]          # the four lanes into one sum
        copy_s.w $t0, $w3[1]
        addu  $v0, $v0, $t0
        copy_s.w $t0, $w3[2]
        addu  $v0, $v0, $t0
        copy_s.w $t0, $w3[3]
        addu  $v0, $v0, $t0
        jr    $ra

# print_array(a, n): "a0 a1 ..." and a new line (a leaf)
print_array:
        move  $t0, $a0
        sll   $t1, $a1, 2
        addu  $t1, $t0, $t1           # the end
pa:     lw    $a0, 0($t0)
        li    $v0, 1
        syscall
        addiu $t0, $t0, 4
        li    $v0, 11
        li    $a0, ' '                # a space between two, a new line after the last
        bne   $t0, $t1, pspace
        li    $a0, '\n'
pspace: syscall
        bne   $t0, $t1, pa
        jr    $ra
# eMIPS: --release R5
