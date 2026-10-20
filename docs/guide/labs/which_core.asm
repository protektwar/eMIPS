# Which core is this?  The program asks the hardware, as an operating
# system's CPU probe does: PRId names the core, Config / Config1 / Config3
# say what it was built with.  Run it with --core NAME (python3 -m emips
# cores lists them) and compare.
        .data
s_prid: .asciiz "PRId "
s_rel:  .asciiz "\nrelease "
s_mmu:  .asciiz "\nMMU type "
s_tlb:  .asciiz ", TLB entries "
s_fpu:  .asciiz "\nFPU "
s_ic:   .asciiz "\nI-cache KB "
s_dc:   .asciiz "\nD-cache KB "
s_ways: .asciiz ", ways "
s_line: .asciiz ", line "
s_m16:  .asciiz "\nMIPS16e "
s_mm:   .asciiz "\nConfig3.ISA "
s_dsp:  .asciiz "\nDSP Module "
s_nl:   .asciiz "\n"

        .text
main:   mfc0    $t0, $16
        srl     $t0, $t0, 10        # Release 2+ (Config.AR > 0) has EBase:
        andi    $t0, $t0, 7         # only core 0 (EBase.CPUNum = 0) reports
        beq     $t0, $zero, report
        mfc0    $t0, $15, 1
        andi    $t0, $t0, 0x3ff
        beq     $t0, $zero, report
idle:   wait                        # the other cores of a multi-core system
        b       idle
report: la      $a0, s_prid
        jal     puts
        mfc0    $a0, $15            # PRId: company 23:16, processor 15:8
        li      $v0, 34
        syscall
        mfc0    $s0, $16            # Config
        la      $a0, s_rel
        jal     puts
        srl     $a0, $s0, 10        # AR 12:10: 0 = Release 1, 1 = 2 to 5, 2 = 6
        andi    $a0, $a0, 7
        jal     putn
        la      $a0, s_mmu
        jal     puts
        srl     $a0, $s0, 7         # MT 9:7: 1 TLB, 3 fixed, 4 VTLB + FTLB
        andi    $a0, $a0, 7
        jal     putn
        mfc0    $s1, $16, 1         # Config1
        srl     $t0, $s0, 7
        andi    $t0, $t0, 7
        li      $t1, 3
        beq     $t0, $t1, nontlb    # the fixed mapping has no TLB
        la      $a0, s_tlb
        jal     puts
        srl     $a0, $s1, 25        # MMUSize 30:25 = entries - 1
        andi    $a0, $a0, 63
        addiu   $a0, $a0, 1
        jal     putn
nontlb: la      $a0, s_fpu
        jal     puts
        andi    $a0, $s1, 1         # FP (bit 0)
        jal     putn
        la      $a0, s_ic
        jal     puts
        srl     $a0, $s1, 16        # IS IL IA at 24:22, 21:19, 18:16
        jal     cache
        la      $a0, s_dc
        jal     puts
        srl     $a0, $s1, 7         # DS DL DA at 15:13, 12:10, 9:7
        jal     cache
        la      $a0, s_m16
        jal     puts
        srl     $a0, $s1, 2         # CA (bit 2): MIPS16e
        andi    $a0, $a0, 1
        jal     putn
        la      $a0, s_mm
        jal     puts
        mfc0    $a0, $16, 3         # Config3.ISA 15:14: 0 MIPS32, 1 microMIPS only,
        srl     $a0, $a0, 14        # 2 both (MIPS32 at reset)
        andi    $a0, $a0, 3
        jal     putn
        la      $a0, s_dsp
        jal     puts
        mfc0    $t0, $16, 3         # Config3.DSPP (10) + DSP2P (11): 0 none,
        srl     $a0, $t0, 10        # 1 Revision 1, 2 Revision 2
        andi    $a0, $a0, 1
        srl     $t1, $t0, 11
        andi    $t1, $t1, 1
        addu    $a0, $a0, $t1
        jal     putn
        la      $a0, s_nl
        jal     puts
        li      $v0, 10
        syscall

# cache: $a0 = S (bits 8:6) L (5:3) A (2:0) of one cache; prints size, ways, line
cache:  move    $t9, $ra
        srl     $t0, $a0, 3
        andi    $t0, $t0, 7         # L: line = 2 << L, 0 = no cache
        beq     $t0, $zero, none
        li      $t3, 2
        sllv    $t3, $t3, $t0       # line bytes
        srl     $t1, $a0, 6
        andi    $t1, $t1, 7         # S: sets = 64 << S (7: 32)
        li      $t4, 64
        sllv    $t4, $t4, $t1
        li      $t5, 7
        bne     $t1, $t5, 1f
        li      $t4, 32
1:      andi    $t2, $a0, 7
        addiu   $t2, $t2, 1         # A: ways = A + 1
        mul     $a0, $t4, $t3
        mul     $a0, $a0, $t2
        srl     $a0, $a0, 10
        jal     putn
        la      $a0, s_ways
        jal     puts
        move    $a0, $t2
        jal     putn
        la      $a0, s_line
        jal     puts
        move    $a0, $t3
        jal     putn
        jr      $t9
none:   move    $a0, $zero
        jal     putn
        jr      $t9

puts:   li      $v0, 4
        syscall
        jr      $ra
putn:   li      $v0, 1
        syscall
        jr      $ra
