# The eMIPS boot monitor for the MIPS Malta (PLAN M15 phase 5).
#
# A small YAMON-compatible monitor in the Malta's monitor flash
# (0x1fc00000 = 0x1e000000, the boot alias): what YAMON does before it
# runs an application (MD00008 "YAMON User's Manual" 2.19: Table 3.1 the
# map it programs, chapter 5 the application interface) — and what eMIPS's
# board firmware does in Python for the kernel boot, in real MIPS code the
# core executes from the reset vector:
#
#   * the GT-64120: its registers to 0x1be00000, the SDRAM, the device bus
#     (flash and board registers), the PCI windows, its PCI target over
#     the SDRAM (DMA: PCI address = physical address);
#   * the PIIX4: PCI interrupt routing, the IDE function (I/O, bus master,
#     both channels, the bus master block); the disks spin up;
#   * the Super I/O's serial ports tty0 / tty1 and the CBUS UART tty2:
#     38400 baud, 8N1, FIFOs;
#   * the caches (Index Store Tag), the TLB (unique, never-matching
#     entries), kseg0 cacheable; on a multi-core Malta the GIC (the 8259s
#     on its source 3, to core 0's pin 0) and core 0 in the coherence domain;
#   * the exception vectors in RAM (EBase 0x80000000) — unless the
#     application brings its own — and the YAMON function table at
#     0x1fc00500 (print_count, exit, flush_cache, print, register /
#     deregister CPU ISR, IC ISR and ESR, getchar, syscon_read: the
#     offsets of yamon_api.h);
#   * then the application, in YAMON's application context (Table 5.1):
#     a0 = argc, a1 = argv, a2 = the environment, a3 = the memory size,
#     sp = 4 words below the top of the SDRAM, ra = back to the monitor,
#     EPC = the entry, interrupts off.
#
# The application is in memory already: eMIPS's loader plays YAMON's
# "load" (and fills the application descriptor at MON_DESC, what "go"
# knows).  When the application returns (jr ra, or exit), the monitor
# ends the run (the eMIPS exit service) — YAMON would show its prompt.
#
# @NAME@ are board values filled in by emips.monitor; lines between
# "#@if NAME" / "#@else" / "#@endif" are kept or dropped by it.

        .set reorder
        .eqv  MON_DESC, 0x8000c000      # the application descriptor (the loader)
        .eqv  CPU_ISR, 0x8000c100       # 8 x (isr, data) + the default
        .eqv  IC_ISR, 0x8000c180        # 16 x (isr, data) + the default
        .eqv  ESR, 0x8000c210           # 32 x esr + the default
        .eqv  MON_FLAGS, 0x8000c2a0     # bit 0: the 8259s are initialised
        .eqv  FRAME, 0x8000c300         # an exception's saved registers
        .eqv  EXC_STACK, 0x8000fff0     # the exception handler's stack
        .eqv  DEFAULT_ID, 0xfffffff0    # YAMON_DEFAULT_HANDLER

# ------------------------------------------------------------------ the reset vector

        .text 0xbfc00000
reset:  b     start                     # 0xbfc00010 is the REVISION register

# ------------------------------------------------------------------ BEV vectors

        .text 0xbfc00200                # TLB refill (BEV = 1)
        j     bev_exception
        .text 0xbfc00300                # cache error (BEV = 1)
        j     bev_exception
        .text 0xbfc00380                # general exception (BEV = 1)
        j     bev_exception
        .text 0xbfc00400                # interrupt, Cause.IV (BEV = 1)
        j     bev_exception
        .text 0xbfc00480                # EJTAG debug exception (no probe)
        j     debug_exception

# ------------------------------------------------------------------ the function table

        .text 0xbfc00500                # 0x1fc00500 (yamon_api.h): kseg0 addresses
        .word 0                                  # 0x00
        .word f_print_count - 0x20000000         # 0x04 print_count
        .word 0, 0, 0, 0, 0, 0                   # 0x08 - 0x1c
        .word f_exit - 0x20000000                # 0x20 exit
        .word 0, 0                               # 0x24, 0x28
        .word f_flush_cache - 0x20000000         # 0x2c flush_cache
        .word 0                                  # 0x30
        .word f_print - 0x20000000               # 0x34 print
        .word f_register_cpu_isr - 0x20000000    # 0x38
        .word f_deregister_cpu_isr - 0x20000000  # 0x3c
        .word f_register_ic_isr - 0x20000000     # 0x40
        .word f_deregister_ic_isr - 0x20000000   # 0x44
        .word f_register_esr - 0x20000000        # 0x48
        .word f_deregister_esr - 0x20000000      # 0x4c
        .word f_getchar - 0x20000000             # 0x50
        .word f_syscon_read - 0x20000000         # 0x54

# ------------------------------------------------------------------ the initialisation

        .text 0xbfc00600
start:  lui   $t0, 0xb400               # 1. the GT-64120's registers: at reset at
        li    $t1, @ISD@                #    0x14000000, moved to 0x1be00000 (ISD;
        sw    $t1, 0x68($t0)            #    a little-endian register)
        lui   $s7, 0xbbe0               # the GT-64120's registers from now on
@GT_INIT@
        lui   $s6, 0xb800               # ISA / PCI I/O ports (PCI I/O window)

        li    $a0, @CFG_PIRQ@           # 2. the PIIX4: PIRQA-D to IRQ 10, 10, 11, 11
        li    $a1, 0x0b0b0a0a
        bal   cfg_write
        li    $a0, @CFG_IDE_CMD@        # the IDE function: I/O space, bus master
        li    $a1, 0x00000005
        bal   cfg_write
        li    $a0, @CFG_IDE_TIM@        # both channels decoded, compatible timing
        li    $a1, 0x80008000
        bal   cfg_write
        li    $a0, @CFG_IDE_BMIBA@      # the bus master block
        li    $a1, @BMIBA@
        bal   cfg_write
        li    $a0, @CFG_GT_CMD@         # 3. the GT-64120 as a PCI target: Memory /
        li    $a1, @GT_CMD@             #    Master Enable, BAR0 at PCI 0 (its own
        bal   cfg_write                 #    header: little-endian)
        li    $a0, @CFG_GT_BAR0@
        li    $a1, 0
        bal   cfg_write

        li    $t0, 0x55                 # 4. the Super I/O (configuration mode)
        sb    $t0, 0x3f0($s6)
        li    $a0, 4                    # logical device 4: tty0 at 0x3f8, IRQ 4
        li    $a1, 0x3f8
        li    $a2, 4
        bal   superio_serial
        li    $a0, 5                    # logical device 5: tty1 at 0x2f8, IRQ 3
        li    $a1, 0x2f8
        li    $a2, 3
        bal   superio_serial
        li    $t0, 0xaa                 # exit the configuration mode
        sb    $t0, 0x3f0($s6)

        addiu $a0, $s6, 0x3f8           # 5. the UARTs: 38400 8N1, FIFOs, OUT2 off
        li    $a1, 3                    #    (1.8432 MHz / 16 / 3)
        li    $a2, 1
        bal   uart_init
        addiu $a0, $s6, 0x2f8
        bal   uart_init
        lui   $a0, 0xbf00               # tty2: the CBUS UART, bits 7:0 of words 8
        li    $t0, 0x80                 # bytes apart (32-bit accesses, as Linux's
        sw    $t0, 0x918($a0)           # regshift 3 / mem32): 3.6864 MHz / 16 / 6
        li    $t0, 6
        sw    $t0, 0x900($a0)           # DLL
        sw    $zero, 0x908($a0)         # DLM
        li    $t0, 0x03
        sw    $t0, 0x918($a0)           # LCR: 8N1
        li    $t0, 0x07
        sw    $t0, 0x910($a0)           # FCR: FIFOs
        li    $t0, 0x03
        sw    $t0, 0x920($a0)           # MCR: DTR, RTS

        addiu $a0, $s6, 0x1f7           # 6. the disks spin up: wait while BSY
        bal   disk_wait
        addiu $a0, $s6, 0x177
        bal   disk_wait

        bal   cache_init                # 7. the caches (still running uncached)
        bal   tlb_init                  # 8. the TLB
        bal   cps_init                  # 9. a multi-core Malta: the GIC, coherence

        la    $t0, go                   # 10. on through kseg0 (the caches are clean)
        lui   $t1, 0x2000
        subu  $t0, $t0, $t1
        jr    $t0

# ------------------------------------------------------------------ init helpers

cfg_write:                              # a0: PCI configuration address, a1: data
        sw    $a0, 0xcf8($s7)           # (both as the CPU stores them)
        sw    $a1, 0xcfc($s7)
        jr    $ra

superio_serial:                         # a0: logical device, a1: base, a2: IRQ
        li    $t0, 0x07                 # select the logical device
        sb    $t0, 0x3f0($s6)
        sb    $a0, 0x3f1($s6)
        li    $t0, 0x60                 # base address, high byte
        sb    $t0, 0x3f0($s6)
        srl   $t1, $a1, 8
        sb    $t1, 0x3f1($s6)
        li    $t0, 0x61                 # base address, low byte
        sb    $t0, 0x3f0($s6)
        sb    $a1, 0x3f1($s6)
        li    $t0, 0x70                 # the IRQ
        sb    $t0, 0x3f0($s6)
        sb    $a2, 0x3f1($s6)
        li    $t0, 0x30                 # activate
        sb    $t0, 0x3f0($s6)
        li    $t1, 1
        sb    $t1, 0x3f1($s6)
        jr    $ra

uart_init:                              # a0: base, a1: divisor, a2: 1 (byte registers)
        sll   $t1, $a2, 1
        addu  $t1, $t1, $a2
        addu  $t1, $t1, $a0             # LCR (register 3)
        li    $t0, 0x80                 # DLAB
        sb    $t0, 0($t1)
        sb    $a1, 0($a0)               # DLL
        addu  $t2, $a0, $a2
        sb    $zero, 0($t2)             # DLM
        li    $t0, 0x03                 # 8 data bits, no parity, 1 stop bit
        sb    $t0, 0($t1)
        sll   $t2, $a2, 1
        addu  $t2, $t2, $a0             # FCR (register 2)
        li    $t0, 0x07                 # FIFOs on, both emptied
        sb    $t0, 0($t2)
        sll   $t2, $a2, 2
        addu  $t2, $t2, $a0             # MCR (register 4)
        li    $t0, 0x03                 # DTR, RTS
        sb    $t0, 0($t2)
        jr    $ra

disk_wait:                              # a0: a channel's Status port
        lbu   $t0, 0($a0)
        li    $t1, 0x7f                 # nothing on the cable (DD7 pulled down)
        beq   $t0, $t1, 2f
        li    $t1, 0xff                 # the channel not decoded
        beq   $t0, $t1, 2f
        li    $t2, @DISK_TIMEOUT@
1:      lbu   $t0, 0($a0)
        andi  $t0, $t0, 0x80            # BSY
        beqz  $t0, 2f
        addiu $t2, $t2, -1
        bnez  $t2, 1b
2:      jr    $ra

cache_init:                             # Index Store Tag with zero tags: every
        move  $t8, $ra                  # line invalid (ITagLo, DTagLo, L23TagLo)
        mtc0  $zero, $28, 0
        mtc0  $zero, $28, 2
        mtc0  $zero, $28, 4
        mtc0  $zero, $29, 0
        mfc0  $t9, $16, 1               # Config1
        srl   $t0, $t9, 19              # IL: the I-cache line (2 << IL), 0: none
        andi  $t0, $t0, 7
        beqz  $t0, 2f
        li    $t1, 2
        sllv  $t1, $t1, $t0             # line
        srl   $t2, $t9, 22              # IS: sets = 64 << IS (7: 32)
        andi  $t2, $t2, 7
        bal   sets_of
        srl   $t3, $t9, 16              # IA: ways - 1
        andi  $t3, $t3, 7
        addiu $t3, $t3, 1
        move  $t4, $zero                # sets x ways lines (no multiply:
7:      addu  $t4, $t4, $v0             # Release 6 has no HI / LO)
        addiu $t3, $t3, -1
        bnez  $t3, 7b
        lui   $t5, 0x8000
1:      cache 8, 0($t5)                 # Index Store Tag I
        addu  $t5, $t5, $t1
        addiu $t4, $t4, -1
        bnez  $t4, 1b
2:      srl   $t0, $t9, 10              # DL
        andi  $t0, $t0, 7
        beqz  $t0, 4f
        li    $t1, 2
        sllv  $t1, $t1, $t0
        srl   $t2, $t9, 13              # DS
        andi  $t2, $t2, 7
        bal   sets_of
        srl   $t3, $t9, 7               # DA
        andi  $t3, $t3, 7
        addiu $t3, $t3, 1
        move  $t4, $zero                # sets x ways lines (no multiply:
7:      addu  $t4, $t4, $v0             # Release 6 has no HI / LO)
        addiu $t3, $t3, -1
        bnez  $t3, 7b
        lui   $t5, 0x8000
3:      cache 9, 0($t5)                 # Index Store Tag D
        addu  $t5, $t5, $t1
        addiu $t4, $t4, -1
        bnez  $t4, 3b
4:      mfc0  $t0, $16, 1               # Config2 is there (Config1.M)
        bgez  $t0, 6f
        mfc0  $t9, $16, 2               # Config2: the L2 (SS 11:8, SL 7:4, SA 3:0)
        srl   $t0, $t9, 4
        andi  $t0, $t0, 0xf
        beqz  $t0, 6f
        li    $t1, 2
        sllv  $t1, $t1, $t0             # line
        srl   $t2, $t9, 8
        andi  $t2, $t2, 0xf
        li    $v0, 64
        sllv  $v0, $v0, $t2             # sets = 64 << SS
        andi  $t3, $t9, 0xf
        addiu $t3, $t3, 1
        move  $t4, $zero                # sets x ways lines (no multiply:
7:      addu  $t4, $t4, $v0             # Release 6 has no HI / LO)
        addiu $t3, $t3, -1
        bnez  $t3, 7b
        lui   $t5, 0x8000
5:      cache 0xb, 0($t5)               # Index Store Tag S
        addu  $t5, $t5, $t1
        addiu $t4, $t4, -1
        bnez  $t4, 5b
6:      jr    $t8

sets_of:                                # t2: IS / DS -> v0 = sets
        li    $v0, 32
        li    $t6, 7
        beq   $t2, $t6, 1f
        li    $v0, 64
        sllv  $v0, $v0, $t2
1:      jr    $ra

tlb_init:                               # a TLB (Config.MT 1 or 4): every entry a
        mfc0  $t0, $16, 0               # unique kseg0 VPN2 (never matches) and
        srl   $t1, $t0, 7               # invalid
        andi  $t1, $t1, 7
        li    $t2, 1
        beq   $t1, $t2, 1f
        li    $t2, 4
        bne   $t1, $t2, 3f
1:      mfc0  $t0, $16, 1
        srl   $t3, $t0, 25              # Config1.MMUSize - 1
        andi  $t3, $t3, 0x3f
        mtc0  $zero, $2                 # EntryLo0
        mtc0  $zero, $3                 # EntryLo1
        mtc0  $zero, $5                 # PageMask: 4 KB
        mtc0  $zero, $6                 # Wired
        lui   $t4, 0x8000
2:      mtc0  $t4, $10                  # EntryHi
        mtc0  $t3, $0                   # Index
        sll   $zero, $zero, 3           # ehb
        tlbwi
        addiu $t4, $t4, 0x2000
        addiu $t3, $t3, -1
        bgez  $t3, 2b
3:      jr    $ra

cps_init:                               # Config3.CMGCR: a CPS — the GIC and the
        mfc0  $t0, $16, 1               # coherence domain
        bgez  $t0, 2f
        mfc0  $t0, $16, 2
        bgez  $t0, 2f
        mfc0  $t0, $16, 3
        sll   $t1, $t0, 2               # CMGCR is bit 29
        bgez  $t1, 2f
        mfc0  $t0, $15, 3               # CMGCRBase: PA 35:15 in bits 27:11
        sll   $t0, $t0, 4
        lui   $t1, 0xa000
        or    $t0, $t0, $t1             # the GCRs through kseg1
        li    $t1, 0x1bdc0001           # GCR_GIC_BASE: the GIC at 0x1bdc0000
        sw    $t1, 0x80($t0)
        li    $t1, 0xff                 # GCR_CL_COHERENCE: core 0 coherent
        sw    $t1, 0x2008($t0)
        lui   $t0, 0xbbdc               # the GIC: source 3 level, active high,
        li    $t1, 0x08                 # enabled, to pin 0 of VP 0
        sw    $t1, 0x100($t0)           # SH_POL
        sw    $zero, 0x180($t0)         # SH_TRIG: level
        lui   $t1, 0x8000
        sw    $t1, 0x50c($t0)           # SH_MAP3_PIN: to pin 0
        li    $t1, 1
        sw    $t1, 0x2060($t0)          # SH_MAP3_VP: VP 0
        li    $t1, 0x08
        sw    $t1, 0x380($t0)           # SH_SMASK: source 3 enabled
2:      jr    $ra

bev_exception:                          # an exception before the vectors are in RAM
        lui   $sp, %hi(EXC_STACK)
        addiu $sp, $sp, %lo(EXC_STACK)
        b     unhandled

debug_exception:                        # EJTAG: no probe, no debug monitor
        la    $a0, msg_debug
        bal   mon_puts
        li    $a0, 0x200
        b     mon_exit

# ------------------------------------------------------------------ on through kseg0

go:     mfc0  $t0, $16, 0
        li    $t1, @CONFIG_CLEAR@       # 11. kseg0 (and kuseg / kseg2-3 with the
        and   $t0, $t0, $t1             #     fixed mapping) cacheable
        li    $t1, @CONFIG_SET@
        or    $t0, $t0, $t1
        mtc0  $t0, $16, 0
        sll   $zero, $zero, 3           # ehb

        lui   $s0, %hi(MON_DESC)        # 12. the vectors in RAM (EBase 0x80000000)
        addiu $s0, $s0, %lo(MON_DESC)
        lw    $t0, 8($s0)               # flags: 1 the application has its handler,
        andi  $t1, $t0, 1               # 2 its TLB refill handler
        bnez  $t1, 1f
        lui   $a0, 0xa000               # (through kseg1: straight to memory)
        addiu $a0, $a0, 0x180
        bal   put_stub
        lui   $a0, 0xa000
        addiu $a0, $a0, 0x200
        bal   put_stub
1:      lw    $t0, 8($s0)
        andi  $t1, $t0, 2
        bnez  $t1, 2f
        mfc0  $t0, $16, 0               # a TLB: the firmware's refill handler
        srl   $t1, $t0, 7
        andi  $t1, $t1, 7
        li    $t2, 1
        beq   $t1, $t2, 3f
        li    $t2, 4
        bne   $t1, $t2, 2f
3:      la    $t0, refill_start         # (copied word by word to 0x80000000)
        la    $t1, refill_end
        lui   $t2, 0xa000
4:      lw    $t3, 0($t0)
        sw    $t3, 0($t2)
        addiu $t0, $t0, 4
        addiu $t2, $t2, 4
        bne   $t0, $t1, 4b
2:
        la    $a0, msg_banner           # 13. the sign-on message on tty0
        bal   mon_puts
        mtc0  $zero, $15, 1             # 14. the application (Table 5.1): EBase 0x80000000
        li    $t0, @STATUS@             # kernel mode, BEV = ERL = 0, EXL = 1 (for
        mtc0  $t0, $12                  # the eret), interrupts off
        lw    $t0, 4($s0)               # the entry
        mtc0  $t0, $14                  # EPC
        lw    $a0, 12($s0)              # argc
        lw    $a1, 16($s0)              # argv
        lw    $a2, 20($s0)              # the environment
        lw    $a3, 24($s0)              # the memory size
        lw    $sp, 28($s0)              # 4 words below the top of the SDRAM
        lw    $gp, 32($s0)              # (eMIPS: the program's $gp)
        la    $ra, appl_return
        lui   $t0, 0x2000
        subu  $ra, $ra, $t0             # kseg0
        move  $s0, $zero
        move  $s6, $zero
        move  $s7, $zero
        move  $t0, $zero
        move  $t1, $zero
        move  $t2, $zero
        move  $t3, $zero
        move  $t9, $zero
        move  $v0, $zero
        sll   $zero, $zero, 3           # ehb
        eret

put_stub:                               # a0: a vector in RAM: jump to the handler
        la    $t0, exc_stub
        lw    $t1, 0($t0)
        sw    $t1, 0($a0)
        lw    $t1, 4($t0)
        sw    $t1, 4($a0)
        lw    $t1, 8($t0)
        sw    $t1, 8($a0)
        lw    $t1, 12($t0)
        sw    $t1, 12($a0)
        jr    $ra

exc_stub:                               # (copied to the vectors)
        .set noreorder
        lui   $k0, %hi(mon_exception - 0x20000000)
        addiu $k0, $k0, %lo(mon_exception - 0x20000000)
        jr    $k0
        nop
        .set reorder

appl_return:                            # the application returned (jr ra): v0
        move  $a0, $v0
        b     f_exit

# ------------------------------------------------------------------ the TLB refill handler

refill_start:
@REFILL@
refill_end:

# ------------------------------------------------------------------ the exception handler

mon_exception:                          # every exception and interrupt (BEV = 0)
        .set noat
        lui   $k0, %hi(FRAME)
        addiu $k0, $k0, %lo(FRAME)
        sw    $1, 4($k0)
        sw    $2, 8($k0)
        sw    $3, 12($k0)
        sw    $4, 16($k0)
        sw    $5, 20($k0)
        sw    $6, 24($k0)
        sw    $7, 28($k0)
        sw    $8, 32($k0)
        sw    $9, 36($k0)
        sw    $10, 40($k0)
        sw    $11, 44($k0)
        sw    $12, 48($k0)
        sw    $13, 52($k0)
        sw    $14, 56($k0)
        sw    $15, 60($k0)
        sw    $16, 64($k0)
        sw    $17, 68($k0)
        sw    $18, 72($k0)
        sw    $19, 76($k0)
        sw    $20, 80($k0)
        sw    $21, 84($k0)
        sw    $22, 88($k0)
        sw    $23, 92($k0)
        sw    $24, 96($k0)
        sw    $25, 100($k0)
        sw    $28, 112($k0)
        sw    $29, 116($k0)
        sw    $30, 120($k0)
        sw    $31, 124($k0)
#@if HILO
        mfhi  $t0
        sw    $t0, 128($k0)
        mflo  $t0
        sw    $t0, 132($k0)
#@endif
        .set at
        lui   $sp, %hi(EXC_STACK)
        addiu $sp, $sp, %lo(EXC_STACK)
        mfc0  $s0, $13                  # Cause
        andi  $t0, $s0, 0x7c
        srl   $s1, $t0, 2               # ExcCode
        bnez  $s1, exc_esr

        mfc0  $t1, $12                  # an interrupt: the highest pending and
        and   $t2, $s0, $t1             # enabled, IP7 first
        srl   $t2, $t2, 8
        andi  $t2, $t2, 0xff
        beqz  $t2, exc_done             # gone already
        li    $s2, 7
1:      srlv  $t3, $t2, $s2
        andi  $t3, $t3, 1
        bnez  $t3, 2f
        addiu $s2, $s2, -1
        bgez  $s2, 1b
        b     exc_done
2:      sll   $t4, $s2, 3               # s2: the CPU interrupt
        li    $t5, CPU_ISR
        addu  $t5, $t5, $t4
        lw    $t6, 0($t5)
        bnez  $t6, call_isr
        li    $t7, 2                    # INT0 (IP2): the 8259s, if an IC ISR is
        bne   $s2, $t7, cpu_default     # registered for the line
        li    $t0, MON_FLAGS
        lw    $t0, 0($t0)
        andi  $t0, $t0, 1
        beqz  $t0, cpu_default
        lui   $t0, 0xbbe0               # a PCI interrupt acknowledge: the vector
        lbu   $s3, 0xc34($t0)           # (ICW2 bases 0 and 8: the IRQ)
        andi  $s3, $s3, 0x0f
        sll   $t4, $s3, 3
        li    $t5, IC_ISR
        addu  $t5, $t5, $t4
        lw    $t6, 0($t5)
        bnez  $t6, 3f
        li    $t5, IC_ISR + 128         # the default IC ISR
        lw    $t6, 0($t5)
        bnez  $t6, 3f
        move  $a0, $s3                  # none: mask the line
        bal   ic_mask
        la    $a0, msg_ic
        bal   mon_puts
        move  $a0, $s3
        bal   mon_puthex
        la    $a0, msg_masked
        bal   mon_puts
        b     4f
3:      lw    $a0, 4($t5)
        jalr  $t6
4:      lui   $t1, 0xb800               # end of interrupt: the slave, the master
        li    $t0, 0x20
        sltiu $t2, $s3, 8
        bnez  $t2, 5f
        sb    $t0, 0xa0($t1)
5:      sb    $t0, 0x20($t1)
        b     exc_done
cpu_default:
        li    $t5, CPU_ISR + 64         # the default CPU ISR
        lw    $t6, 0($t5)
        bnez  $t6, call_isr
        mfc0  $t0, $12                  # none: disable the interrupt
        li    $t1, 0x100
        sllv  $t1, $t1, $s2
        nor   $t1, $t1, $zero
        and   $t0, $t0, $t1
        mtc0  $t0, $12
        la    $a0, msg_int
        bal   mon_puts
        move  $a0, $s2
        bal   mon_puthex
        la    $a0, msg_masked
        bal   mon_puts
        b     exc_done
call_isr:
        lw    $a0, 4($t5)               # isr(data)
        jalr  $t6
        b     exc_done

exc_esr:                                # an exception: its ESR, the default ESR
        sll   $t0, $s1, 2
        li    $t1, ESR
        addu  $t1, $t1, $t0
        lw    $t6, 0($t1)
        bnez  $t6, 1f
        li    $t1, ESR + 128
        lw    $t6, 0($t1)
        beqz  $t6, unhandled
1:      jalr  $t6                       # esr(): EPC / Status as it leaves them
        b     exc_done

unhandled:                              # no ESR: say so, end the application
        la    $a0, msg_exc
        bal   mon_puts
        mfc0  $a0, $13
        srl   $a0, $a0, 2
        andi  $a0, $a0, 0x1f
        move  $s1, $a0
        bal   mon_puthex
        la    $a0, msg_epc
        bal   mon_puts
        mfc0  $a0, $14
        bal   mon_puthex
        la    $a0, msg_bad
        bal   mon_puts
        mfc0  $a0, $8
        bal   mon_puthex
        la    $a0, msg_nl
        bal   mon_puts
        addiu $a0, $s1, 0x100           # the exit code: 0x100 + ExcCode
        b     mon_exit

exc_done:                               # back to the interrupted code
        .set noat
        lui   $k0, %hi(FRAME)
        addiu $k0, $k0, %lo(FRAME)
#@if HILO
        lw    $t0, 128($k0)
        mthi  $t0
        lw    $t0, 132($k0)
        mtlo  $t0
#@endif
        lw    $1, 4($k0)
        lw    $2, 8($k0)
        lw    $3, 12($k0)
        lw    $4, 16($k0)
        lw    $5, 20($k0)
        lw    $6, 24($k0)
        lw    $7, 28($k0)
        lw    $8, 32($k0)
        lw    $9, 36($k0)
        lw    $10, 40($k0)
        lw    $11, 44($k0)
        lw    $12, 48($k0)
        lw    $13, 52($k0)
        lw    $14, 56($k0)
        lw    $15, 60($k0)
        lw    $16, 64($k0)
        lw    $17, 68($k0)
        lw    $18, 72($k0)
        lw    $19, 76($k0)
        lw    $20, 80($k0)
        lw    $21, 84($k0)
        lw    $22, 88($k0)
        lw    $23, 92($k0)
        lw    $24, 96($k0)
        lw    $25, 100($k0)
        lw    $28, 112($k0)
        lw    $29, 116($k0)
        lw    $30, 120($k0)
        lw    $31, 124($k0)
        sll   $zero, $zero, 3           # ehb
        eret
        .set at

# ------------------------------------------------------------------ the functions
# (called through the table with the o32 convention; they use the
# caller's stack: none of them needs one)

f_print_count:                          # print_count(port, s, count): tty0
        lui   $t9, 0xb800
1:      blez  $a2, 3f
2:      lbu   $t0, 0x3fd($t9)           # LSR.THRE
        andi  $t0, $t0, 0x20
        beqz  $t0, 2b
        lbu   $t1, 0($a1)
        sb    $t1, 0x3f8($t9)
        addiu $a1, $a1, 1
        addiu $a2, $a2, -1
        b     1b
3:      jr    $ra

f_print:                                # print(port, s): a zero-terminated string
        move  $a0, $a1
mon_puts:                               # a0: a string (t0, t1, t9)
        lui   $t9, 0xb800
1:      lbu   $t1, 0($a0)
        beqz  $t1, 3f
2:      lbu   $t0, 0x3fd($t9)
        andi  $t0, $t0, 0x20
        beqz  $t0, 2b
        sb    $t1, 0x3f8($t9)
        addiu $a0, $a0, 1
        b     1b
3:      jr    $ra

mon_puthex:                             # a0: a word in hexadecimal (t0-t3, t9)
        lui   $t9, 0xb800
        li    $t2, 28
1:      srlv  $t1, $a0, $t2
        andi  $t1, $t1, 0xf
        sltiu $t3, $t1, 10
        addiu $t1, $t1, '0'
        bnez  $t3, 2f
        addiu $t1, $t1, 'a' - '0' - 10
2:      lbu   $t0, 0x3fd($t9)
        andi  $t0, $t0, 0x20
        beqz  $t0, 2b
        sb    $t1, 0x3f8($t9)
        addiu $t2, $t2, -4
        bgez  $t2, 1b
        jr    $ra

f_getchar:                              # getchar(port, *ch): 1 = a character
        lui   $t9, 0xb800
        lbu   $t0, 0x3fd($t9)           # LSR.DR
        andi  $t0, $t0, 0x01
        beqz  $t0, 1f
        lbu   $t1, 0x3f8($t9)
        sb    $t1, 0($a1)
        li    $v0, 1
        jr    $ra
1:      move  $v0, $zero
        jr    $ra

f_exit:                                 # exit(rc): the application is done
mon_exit:
        lui   $t9, 0xb800               # tty0 sends what it still has (LSR.TEMT)
1:      lbu   $t0, 0x3fd($t9)
        andi  $t0, $t0, 0x40
        beqz  $t0, 1b
        li    $v0, 17                   # the eMIPS exit service (YAMON: its prompt)
        syscall
1:      wait
        b     1b

f_syscon_read:                          # syscon_read(id, *param, size): 0 = done
        sltiu $t0, $a2, 4
        bnez  $t0, 9f
        li    $t1, 34                   # SYSCON_BOARD_CPU_CLOCK_FREQ_ID
        li    $v0, @CPU_HZ@
        beq   $a0, $t1, 8f
        li    $t1, 35                   # SYSCON_BOARD_BUS_CLOCK_FREQ_ID
        li    $v0, @BUS_HZ@
        beq   $a0, $t1, 8f
        li    $t1, 36                   # SYSCON_BOARD_PCI_FREQ_KHZ_ID
        li    $v0, @PCI_KHZ@
        beq   $a0, $t1, 8f
9:      li    $v0, -1
        jr    $ra
8:      sw    $v0, 0($a1)
        move  $v0, $zero
        jr    $ra

f_flush_cache:                          # flush_cache(type): 0 the I-cache
        mfc0  $t9, $16, 1               # (invalidate), 1 the D-cache (write back,
        bnez  $a0, 3f                   # invalidate)
        srl   $t0, $t9, 19
        andi  $t0, $t0, 7
        beqz  $t0, 9f
        li    $t1, 2
        sllv  $t1, $t1, $t0
        srl   $t2, $t9, 22
        andi  $t2, $t2, 7
        move  $t8, $ra
        bal   sets_of
        move  $ra, $t8
        srl   $t3, $t9, 16
        andi  $t3, $t3, 7
        addiu $t3, $t3, 1
        move  $t4, $zero                # sets x ways lines (no multiply:
7:      addu  $t4, $t4, $v0             # Release 6 has no HI / LO)
        addiu $t3, $t3, -1
        bnez  $t3, 7b
        lui   $t5, 0x8000
1:      cache 0, 0($t5)                 # Index Invalidate I
        addu  $t5, $t5, $t1
        addiu $t4, $t4, -1
        bnez  $t4, 1b
        b     9f
3:      srl   $t0, $t9, 10
        andi  $t0, $t0, 7
        beqz  $t0, 9f
        li    $t1, 2
        sllv  $t1, $t1, $t0
        srl   $t2, $t9, 13
        andi  $t2, $t2, 7
        move  $t8, $ra
        bal   sets_of
        move  $ra, $t8
        srl   $t3, $t9, 7
        andi  $t3, $t3, 7
        addiu $t3, $t3, 1
        move  $t4, $zero                # sets x ways lines (no multiply:
7:      addu  $t4, $t4, $v0             # Release 6 has no HI / LO)
        addiu $t3, $t3, -1
        bnez  $t3, 7b
        lui   $t5, 0x8000
4:      cache 1, 0($t5)                 # Index Writeback Invalidate D
        addu  $t5, $t5, $t1
        addiu $t4, $t4, -1
        bnez  $t4, 4b
        sync
9:      jr    $ra

f_register_cpu_isr:                     # (cpu_int, isr, data, *ref): 0 = done
        li    $t0, DEFAULT_ID
        li    $t2, CPU_ISR + 64
        beq   $a0, $t0, 1f
        sltiu $t0, $a0, 8
        beqz  $t0, 9f
        sll   $t1, $a0, 3
        li    $t2, CPU_ISR
        addu  $t2, $t2, $t1
        mfc0  $t3, $12                  # enable the interrupt (Status.IM)
        li    $t4, 0x100
        sllv  $t4, $t4, $a0
        or    $t3, $t3, $t4
        mtc0  $t3, $12
        sll   $zero, $zero, 3           # ehb
1:      sw    $a1, 0($t2)
        sw    $a2, 4($t2)
        beqz  $a3, 2f
        sw    $t2, 0($a3)
2:      move  $v0, $zero
        jr    $ra
9:      li    $v0, -1
        jr    $ra

f_deregister_cpu_isr:                   # (ref)
        sw    $zero, 0($a0)
        sw    $zero, 4($a0)
        li    $t0, CPU_ISR
        subu  $t1, $a0, $t0
        sltiu $t2, $t1, 64
        beqz  $t2, 1f
        srl   $t1, $t1, 3               # disable the interrupt
        li    $t4, 0x100
        sllv  $t4, $t4, $t1
        nor   $t4, $t4, $zero
        mfc0  $t3, $12
        and   $t3, $t3, $t4
        mtc0  $t3, $12
        sll   $zero, $zero, 3           # ehb
1:      move  $v0, $zero
        jr    $ra

f_register_ic_isr:                      # (ic_line, isr, data, *ref): the 8259s
        li    $t0, DEFAULT_ID
        li    $t2, IC_ISR + 128
        beq   $a0, $t0, 1f
        sltiu $t0, $a0, 16
        beqz  $t0, 9f
        sll   $t1, $a0, 3
        li    $t2, IC_ISR
        addu  $t2, $t2, $t1
1:      sw    $a1, 0($t2)
        sw    $a2, 4($t2)
        beqz  $a3, 2f
        sw    $t2, 0($a3)
2:      move  $t8, $ra
        bal   ic_init                   # the first time: ICW1-4, all masked
        li    $t0, DEFAULT_ID
        beq   $a0, $t0, 3f
        bal   ic_unmask                 # a0: the line
3:      mfc0  $t3, $12                  # INT0 (Status.IM2): the 8259s
        ori   $t3, $t3, 0x400
        mtc0  $t3, $12
        sll   $zero, $zero, 3           # ehb
        move  $ra, $t8
        move  $v0, $zero
        jr    $ra
9:      li    $v0, -1
        jr    $ra

f_deregister_ic_isr:                    # (ref)
        sw    $zero, 0($a0)
        sw    $zero, 4($a0)
        li    $t0, IC_ISR
        subu  $t1, $a0, $t0
        sltiu $t2, $t1, 128
        beqz  $t2, 1f
        srl   $a0, $t1, 3
        move  $t8, $ra
        bal   ic_mask
        move  $ra, $t8
1:      move  $v0, $zero
        jr    $ra

ic_init:                                # the 8259s once: edge, cascade, vectors
        li    $t0, MON_FLAGS            # 0x00 / 0x08 (= the IRQ), all masked
        lw    $t1, 0($t0)
        andi  $t2, $t1, 1
        bnez  $t2, 1f
        ori   $t1, $t1, 1
        sw    $t1, 0($t0)
        lui   $t9, 0xb800
        li    $t0, 0x11                 # ICW1
        sb    $t0, 0x20($t9)
        sb    $t0, 0xa0($t9)
        sb    $zero, 0x21($t9)          # ICW2: master vectors 0x00-0x07
        li    $t0, 0x08                 #       slave 0x08-0x0f
        sb    $t0, 0xa1($t9)
        li    $t0, 0x04                 # ICW3: the slave on IRQ 2
        sb    $t0, 0x21($t9)
        li    $t0, 0x02
        sb    $t0, 0xa1($t9)
        li    $t0, 0x01                 # ICW4: 8086 mode
        sb    $t0, 0x21($t9)
        sb    $t0, 0xa1($t9)
        li    $t0, 0xff                 # OCW1: everything masked
        sb    $t0, 0x21($t9)
        sb    $t0, 0xa1($t9)
1:      jr    $ra

ic_unmask:                              # a0: an IRQ 0-15 (a slave's: + IRQ 2)
        lui   $t9, 0xb800
        sltiu $t0, $a0, 8
        beqz  $t0, 1f
        lbu   $t1, 0x21($t9)
        li    $t2, 1
        sllv  $t2, $t2, $a0
        nor   $t2, $t2, $zero
        and   $t1, $t1, $t2
        sb    $t1, 0x21($t9)
        jr    $ra
1:      lbu   $t1, 0xa1($t9)
        addiu $t3, $a0, -8
        li    $t2, 1
        sllv  $t2, $t2, $t3
        nor   $t2, $t2, $zero
        and   $t1, $t1, $t2
        sb    $t1, 0xa1($t9)
        lbu   $t1, 0x21($t9)
        andi  $t1, $t1, 0xfb
        sb    $t1, 0x21($t9)
        jr    $ra

ic_mask:                                # a0: an IRQ 0-15
        lui   $t9, 0xb800
        sltiu $t0, $a0, 8
        li    $t2, 1
        beqz  $t0, 1f
        sllv  $t2, $t2, $a0
        lbu   $t1, 0x21($t9)
        or    $t1, $t1, $t2
        sb    $t1, 0x21($t9)
        jr    $ra
1:      addiu $t3, $a0, -8
        sllv  $t2, $t2, $t3
        lbu   $t1, 0xa1($t9)
        or    $t1, $t1, $t2
        sb    $t1, 0xa1($t9)
        jr    $ra

f_register_esr:                         # (exception, esr, *ref, *retfunc)
        li    $t0, DEFAULT_ID
        li    $t2, ESR + 128
        beq   $a0, $t0, 1f
        sltiu $t0, $a0, 32
        beqz  $t0, 9f
        sll   $t1, $a0, 2
        li    $t2, ESR
        addu  $t2, $t2, $t1
1:      sw    $a1, 0($t2)
        beqz  $a2, 2f
        sw    $t2, 0($a2)
2:      beqz  $a3, 3f
        la    $t0, esr_return           # back to the monitor: the application ends
        lui   $t1, 0x2000
        subu  $t0, $t0, $t1
        sw    $t0, 0($a3)
3:      move  $v0, $zero
        jr    $ra
9:      li    $v0, -1
        jr    $ra

f_deregister_esr:                       # (ref)
        sw    $zero, 0($a0)
        move  $v0, $zero
        jr    $ra

esr_return:                             # an ESR gives up: like an unhandled exception
        b     unhandled

# ------------------------------------------------------------------ messages

msg_banner:   .asciiz "@BANNER@"
msg_exc:      .asciiz "\nmonitor: exception, ExcCode 0x"
msg_epc:      .asciiz ", EPC 0x"
msg_bad:      .asciiz ", BadVAddr 0x"
msg_int:      .asciiz "\nmonitor: no ISR for CPU interrupt 0x"
msg_ic:       .asciiz "\nmonitor: no ISR for IRQ 0x"
msg_debug:    .asciiz "\nmonitor: a debug exception (no probe, no debug handler)\n"
msg_masked:   .asciiz " (masked)\n"
msg_nl:       .asciiz "\n"
