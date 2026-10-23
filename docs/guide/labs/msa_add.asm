# Add two arrays of 64 words, c[i] = a[i] + b[i], then print the sum of c:
# add_msa does four words per instruction (ld.w, addv.w, st.w), add32 one
# word at a time.  .eqv MSA chooses which one runs; both print the same sum.
        .eqv    MSA, 1              # 1: add_msa, 0: add32
        .eqv    N, 64
        .data
        .align  4                   # (16 bytes: one vector)
a:      .space  4 * N
b:      .space  4 * N
c:      .space  4 * N
        .text
main:   la      $s0, a              # a[i] = i, b[i] = 1000 * i
        la      $s1, b
        li      $t0, 0
init:   sll     $t1, $t0, 2
        addu    $t2, $s0, $t1
        sw      $t0, 0($t2)
        mul     $t3, $t0, 1000
        addu    $t2, $s1, $t1
        sw      $t3, 0($t2)
        addiu   $t0, $t0, 1
        slti    $t1, $t0, N
        bnez    $t1, init
        la      $a0, a
        la      $a1, b
        la      $a2, c
        li      $a3, N
        li      $t0, MSA
        beqz    $t0, plain
        jal     add_msa
        b       show
plain:  jal     add32
show:   la      $t0, c              # the sum of c, printed
        li      $t1, N
        li      $a0, 0
sum:    lw      $t2, 0($t0)
        addu    $a0, $a0, $t2
        addiu   $t0, $t0, 4
        addiu   $t1, $t1, -1
        bnez    $t1, sum
        li      $v0, 1
        syscall
        li      $v0, 10
        syscall

# add_msa($a0 = a, $a1 = b, $a2 = c, $a3 = words, a multiple of 4)
add_msa:
vloop:  ld.w    $w0, 0($a0)         # a[i .. i+3]
        ld.w    $w1, 0($a1)         # b[i .. i+3]
        addv.w  $w2, $w0, $w1       # four additions
        st.w    $w2, 0($a2)
        addiu   $a0, $a0, 16
        addiu   $a1, $a1, 16
        addiu   $a2, $a2, 16
        addiu   $a3, $a3, -4
        bnez    $a3, vloop
        jr      $ra

# add32($a0 = a, $a1 = b, $a2 = c, $a3 = words)
add32:
sloop:  lw      $t0, 0($a0)
        lw      $t1, 0($a1)
        addu    $t2, $t0, $t1
        sw      $t2, 0($a2)
        addiu   $a0, $a0, 4
        addiu   $a1, $a1, 4
        addiu   $a2, $a2, 4
        addiu   $a3, $a3, -1
        bnez    $a3, sloop
        jr      $ra
# eMIPS: --release R5
