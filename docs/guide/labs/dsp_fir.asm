# One output of a 16-tap FIR filter: y = sum of h[k] * x[k], all Q15
# fractions (0x4000 = 0.5).  fir_dsp does two taps per instruction
# (dpaq_s.w.ph into accumulator 1, then extr_r.w back to Q15); fir32 one
# tap per madd and the shifts by hand.  .eqv DSP chooses; both print y.
# Compare them on the 24K pipeline:  python3 -m emips pipe --model 24k
        .eqv    DSP, 1              # 1: fir_dsp, 0: fir32
        .data
        .align  2
x:      .half   0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000, 0x7fff
        .half  -0x1000,-0x2000,-0x3000,-0x4000,-0x5000,-0x6000,-0x7000,-0x8000
h:      .half   0x0400, 0x0800, 0x0c00, 0x1000, 0x1400, 0x1800, 0x1c00, 0x2000
        .half   0x2000, 0x1c00, 0x1800, 0x1400, 0x1000, 0x0c00, 0x0800, 0x0400
        .text
main:   la      $a0, x
        la      $a1, h
        li      $a2, 16
        li      $t0, DSP
        beqz    $t0, plain
        jal     fir_dsp
        b       show
plain:  jal     fir32
show:   move    $a0, $v0
        li      $v0, 34             # y in hex (Q15 in the low halfword)
        syscall
        li      $v0, 10
        syscall

# fir_dsp($a0 = x, $a1 = h, $a2 = taps, even): $v0 = y (Q15, rounded)
fir_dsp:
        mult    $ac1, $zero, $zero  # ac1 = 0
dloop:  lw      $t0, 0($a0)         # two samples
        lw      $t1, 0($a1)         # two coefficients
        dpaq_s.w.ph $ac1, $t0, $t1  # ac1 += x0 h0 + x1 h1 (Q31)
        addiu   $a0, $a0, 4
        addiu   $a1, $a1, 4
        addiu   $a2, $a2, -2
        bnez    $a2, dloop
        extr_r.w $v0, $ac1, 16      # Q31 -> Q15, rounded
        jr      $ra

# fir32($a0 = x, $a1 = h, $a2 = taps): $v0 = y (Q15, rounded)
fir32:  mtlo    $zero
        mthi    $zero
floop:  lh      $t0, 0($a0)         # one sample
        lh      $t1, 0($a1)         # one coefficient
        madd    $t0, $t1            # HI:LO += x h (Q30)
        addiu   $a0, $a0, 2
        addiu   $a1, $a1, 2
        addiu   $a2, $a2, -1
        bnez    $a2, floop
        mflo    $t0                 # Q30 -> Q15: (HI:LO + 0x4000) >> 15
        mfhi    $t1
        addiu   $t2, $t0, 0x4000
        sltu    $t3, $t2, $t0       # the carry out of LO
        addu    $t1, $t1, $t3
        srl     $t2, $t2, 15
        sll     $t1, $t1, 17
        or      $v0, $t1, $t2
        jr      $ra
