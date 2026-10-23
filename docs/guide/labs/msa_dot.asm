# Q15 fixed point with MSA: eight fractions x[i] and y[i] (-1 <= q < 1, one
# halfword each: 0x4000 = 0.5, 0x8000 = -1).  Prints the eight products
# truncated (mul_q.h) and rounded (mulr_q.h), then the dot product
# x[0] y[0] + ... + x[7] y[7] as a Q15 fraction (dotp_s.w, hadd_s.d).
# Each line shows the halfwords of one vector, element 0 first.
        .data
        .align  4
x:      .half   0x4000, 0xe000, 0x1234, 0x7fff, 0x8000, 0x0001, 0x2aaa, 0x5555
y:      .half   0x6000, 0x4000, 0x0003, 0x8000, 0x8000, 0x7fff, 0x3000, 0x0007
buf:    .space  16
        .text
main:   la      $t0, x
        ld.h    $w0, 0($t0)         # x[0..7]
        ld.h    $w1, 16($t0)        # y[0..7]
        mul_q.h $w7, $w0, $w1       # x y, truncated to Q15
        jal     show
        mulr_q.h $w7, $w0, $w1      # x y, rounded to Q15
        jal     show
        dotp_s.w $w2, $w0, $w1      # 4 words: x0 y0 + x1 y1, ... (integers: Q30)
        hadd_s.d $w3, $w2, $w2      # 2 doublewords: the odd + even words
        copy_s.w $t1, $w3[0]        # (low halves: the sums fit 32 bits)
        copy_s.w $t2, $w3[2]
        addu    $t1, $t1, $t2       # the dot product, Q30
        addiu   $t1, $t1, 0x4000    # rounded ...
        sra     $a0, $t1, 15        # ... back to Q15
        li      $v0, 34
        syscall
        li      $v0, 10
        syscall

# show: the eight halfwords of $w7, element 0 first, then a newline
show:   la      $t3, buf
        st.h    $w7, 0($t3)         # element 0 at the lowest address
        li      $t4, 8
next:   lhu     $a0, 0($t3)
        li      $v0, 34
        syscall
        li      $a0, ' '
        li      $v0, 11
        syscall
        addiu   $t3, $t3, 2
        addiu   $t4, $t4, -1
        bnez    $t4, next
        li      $a0, '\n'
        li      $v0, 11
        syscall
        jr      $ra
# eMIPS: --release R5
