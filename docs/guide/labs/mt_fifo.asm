# Lab: a FIFO between threads.  Two threads pass ten numbers through a
# FIFO cell of the ITC (gating storage): the producer (TC 1) stores, the
# consumer (TC 0) loads, and neither ever polls.
        .eqv    ITC, 0xb7000000         # the ITC in kseg1 (PA 0x17000000)
        .eqv    CONTROL, 0x08           # view 1: the cell's tag (E, F, FIFO depth)
        .eqv    SYNC, 0x10              # view 2: E/F synchronized
        .eqv    WORK, 20                # the producer's work per number
        .text
main:   li      $t0, 1 << 26
        mtc0    $t0, $26                # ErrCtl.ITC: CACHE tag ops reach the ITC maps
        li      $t0, 0x17000001         # ITCAddressMap0: base 0x17000000, En
        mtc0    $t0, $28, 2             # (through DTagLo)
        ehb                             # CACHE must see ErrCtl and DTagLo
        cache   9, 0($zero)             # Index Store Tag D at 0: ITCAddressMap0
        li      $t0, 1 << 10            # ITCAddressMap1: AddrMask 1 (2 KB), cells 128 bytes apart
        mtc0    $t0, $28, 2
        ehb
        cache   9, 8($zero)             # Index Store Tag D at 8: ITCAddressMap1
        mtc0    $zero, $26
        li      $s6, ITC
        li      $t0, 1
        sw      $t0, CONTROL($s6)       # cell 0: empty (E = 1)

        li      $t0, 1
        mtc0    $t0, $1, 1              # VPEControl.TargTC = 1
        la      $t1, producer
        mttc0   $t1, $2, 3              # TC 1: TCRestart = producer
        mttgpr  $s6, $s6                # ... its $s6
        li      $t1, 0xa000
        mttc0   $t1, $2, 1              # TC 1: TCStatus.A (activated), DA
        mttc0   $zero, $2, 4            # TC 1: TCHalt = 0
        mfc0    $t1, $2, 1
        ori     $t1, $t1, 0x8000
        mtc0    $t1, $2, 1              # TC 0: DA too (TC 1 may YIELD away)
        emt
        evpe

        li      $s0, 0                  # the consumer: ten loads from the FIFO
        li      $s1, 10
get:    lw      $t2, SYNC($s6)          # waits while the cell is empty
        addu    $s0, $s0, $t2
        addiu   $s1, $s1, -1
        bnez    $s1, get
        move    $a0, $s0
        li      $v0, 1
        syscall
        li      $v0, 10
        syscall

producer:
        li      $t0, 1
make:   li      $t3, WORK               # "compute" the next number
1:      addiu   $t3, $t3, -1
        bnez    $t3, 1b
put:    sw      $t0, SYNC($s6)          # waits while the FIFO is full (4 entries)
        addiu   $t0, $t0, 1
        slti    $t1, $t0, 11
        bnez    $t1, make
        yield   $zero                   # done: free this TC
# eMIPS: --mt 1:2
