# Lab 33: DMA on the Malta.  The PIIX4's bus master moves sector 0 of the
# disk into memory by itself: the program gives it a table of memory
# regions (Physical Region Descriptors), tells the disk READ DMA and
# starts it.  The bus master writes the SDRAM directly — not through the
# CPU's data cache: on the Malta DMA is not coherent.  The buffer was in
# the cache before (filled with dots, write-back): what does the CPU see?
        .text
main:   mfc0  $t0, $15, 1           # EBase.CPUNum: on a multi-core Malta only
        andi  $t0, $t0, 0x3ff       # core 0 goes on, the others sleep
        bnez  $t0, idle
        li    $s0, 0xb8000000       # ISA / PCI I/O port P at 0xb8000000 + P
        li    $s7, 0x1fffffff       # kseg0 / kseg1 -> physical: the low 29 bits
        lui   $s6, 0xa000           # kseg1: the same memory, uncached

        la    $s1, buf              # 1. the buffer, through the cache (kseg0)
        li    $t0, '.'
        li    $t1, 64
fill:   sb    $t0, 0($s1)
        addiu $s1, $s1, 1
        addiu $t1, $t1, -1
        bnez  $t1, fill
        sb    $zero, 0($s1)         # a 64-character string of dots

        la    $t0, buf              # 2. the PRD table: one region, EOT
        and   $t0, $t0, $s7         # the buffer's physical address
        wsbh  $t0, $t0              # PCI is little-endian: swap the
        rotr  $t0, $t0, 16          # four bytes of every word
        la    $t2, prd
        or    $t2, $t2, $s6         # through kseg1: straight to memory
        sw    $t0, 0($t2)           # address
        li    $t0, 0x00020080       # 512 bytes (0x0200), EOT (bit 31): 00 02 00 80
        sw    $t0, 4($t2)

        la    $t0, prd              # 3. the bus master (primary channel at
        and   $t0, $t0, $s7         #    ports 0x1000-0x1007)
        wsbh  $t0, $t0
        rotr  $t0, $t0, 16
        sw    $t0, 0x1004($s0)      # PRD table pointer
        li    $t0, 0x06
        sb    $t0, 0x1002($s0)      # clear Error and Interrupt (write 1)
        li    $t0, 0x08
        sb    $t0, 0x1000($s0)      # direction: write to memory

busy:   lbu   $t0, 0x1f7($s0)       # 4. the disk: READ DMA, LBA 0, 1 sector
        andi  $t0, $t0, 0x80
        bnez  $t0, busy
        li    $t0, 0xe0
        sb    $t0, 0x1f6($s0)
        li    $t0, 1
        sb    $t0, 0x1f2($s0)
        sb    $zero, 0x1f3($s0)
        sb    $zero, 0x1f4($s0)
        sb    $zero, 0x1f5($s0)
        li    $t0, 0xc8
        sb    $t0, 0x1f7($s0)
        li    $t0, 0x09
        sb    $t0, 0x1000($s0)      # 5. start the bus master

done:   lbu   $t0, 0x1002($s0)      # wait for IDE Interrupt Status
        andi  $t0, $t0, 0x04
        beqz  $t0, done
        sb    $zero, 0x1000($s0)    # stop it
        lbu   $t0, 0x1f7($s0)       # the disk's status (acknowledges it)

        la    $a0, cached_txt       # 6. what the CPU sees
        li    $v0, 4
        syscall
        la    $a0, buf              # through the cache
        syscall
        la    $a0, uncached_txt
        syscall
        la    $a0, buf
        or    $a0, $a0, $s6         # through kseg1: the memory itself
        syscall

        la    $t0, buf              # 7. drop the buffer's cached lines
        li    $t1, 3                # (64 bytes + the 0: three 32-byte lines)
inval:  cache 0x11, 0($t0)          # Hit Invalidate D
        addiu $t0, $t0, 32
        addiu $t1, $t1, -1
        bnez  $t1, inval
        la    $a0, after_txt
        li    $v0, 4
        syscall
        la    $a0, buf              # through the cache again
        syscall
        li    $v0, 10
        syscall
idle:   wait
        b     idle

        .data
cached_txt:   .asciiz "cached:   "
uncached_txt: .asciiz "\nuncached: "
after_txt:    .asciiz "\ninvalidated, cached: "
        .align 5                    # a cache line of its own
buf:    .space 512
        .space 32
        .align 3
prd:    .space 8
# eMIPS: --board malta --caches --disk malta_dma.txt    (the settings this program runs with)
