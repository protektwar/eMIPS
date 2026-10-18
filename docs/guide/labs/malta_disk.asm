# Lab 32: an IDE disk on the Malta.  The PIIX4's primary IDE channel is
# at ISA ports 0x1f0-0x1f7 (the command block) and 0x3f6 (Alternate
# Status / Device Control); its disk is malta_disk.txt (a file is a disk
# image: one 512-byte sector here).  The program asks the disk who it is
# (IDENTIFY DEVICE), then reads sector 0 by PIO: 256 reads of the 16-bit
# Data register.
        .text
main:   li    $s0, 0xb8000000       # ISA port P is at 0xb8000000 + P
        li    $a0, 0
        li    $a1, 0xec             # IDENTIFY DEVICE
        jal   command
        la    $a0, ident
        jal   read_block            # 256 words: Table 11 of ATA-4
        bnez  $v0, failed

        la    $a0, model_txt
        li    $v0, 4
        syscall
        la    $t1, ident + 54       # words 27-46: the model, 40 characters,
        li    $t2, 20               # the first of each pair in bits 15:8
model:  lbu   $a0, 1($t1)           # bits 15:8 = the byte at the higher address
        li    $v0, 11
        syscall
        lbu   $a0, 0($t1)
        syscall
        addiu $t1, $t1, 2
        addiu $t2, $t2, -1
        bnez  $t2, model

        la    $a0, sectors_txt
        li    $v0, 4
        syscall
        lw    $a0, ident + 120      # words 60-61: the number of sectors, a
        wsbh  $a0, $a0              # little-endian 32-bit value: on a
        rotr  $a0, $a0, 16          # big-endian CPU swap its four bytes
        li    $v0, 1
        syscall
        li    $a0, '\n'
        li    $v0, 11
        syscall

        li    $a0, 0                # LBA 0
        li    $a1, 0x20             # READ SECTOR(S)
        jal   command
        la    $a0, sector
        jal   read_block
        bnez  $v0, failed
        la    $a0, sector           # the sector's bytes are in order
        li    $v0, 4
        syscall
        li    $v0, 10
        syscall

failed: lbu   $a0, 0x1f1($s0)       # the Error register
        li    $v0, 34
        syscall
        li    $v0, 10
        syscall

# command: wait until the disk is not busy, then one sector at LBA $a0
# (28 bits), command code $a1.
command:
        lbu   $t0, 0x1f7($s0)       # Status: wait for BSY = 0
        andi  $t0, $t0, 0x80
        bnez  $t0, command
        li    $t0, 1
        sb    $t0, 0x1f2($s0)       # Sector Count
        sb    $a0, 0x1f3($s0)       # LBA 7:0
        srl   $t0, $a0, 8
        sb    $t0, 0x1f4($s0)       # LBA 15:8
        srl   $t0, $a0, 16
        sb    $t0, 0x1f5($s0)       # LBA 23:16
        srl   $t0, $a0, 24
        andi  $t0, $t0, 0x0f
        ori   $t0, $t0, 0xe0        # Device/Head: LBA, device 0, LBA 27:24
        sb    $t0, 0x1f6($s0)
        sb    $a1, 0x1f7($s0)       # Command: the disk starts
        jr    $ra

# read_block: wait for DRQ (the block is ready), copy 512 bytes to $a0;
# $v0 = 0, or 1 when the disk reports an error (ERR).
read_block:
        lbu   $t0, 0x1f7($s0)       # Status (reading it clears the interrupt)
        andi  $t1, $t0, 0x80
        bnez  $t1, read_block       # BSY: not yet
        andi  $t1, $t0, 0x01
        bnez  $t1, rb_error         # ERR
        andi  $t1, $t0, 0x08
        beqz  $t1, read_block       # no DRQ yet
        li    $t2, 256
rb_word:
        lhu   $t0, 0x1f0($s0)       # the Data register: 16 bits
        sh    $t0, 0($a0)           # stored back as read: the bytes keep their order
        addiu $a0, $a0, 2
        addiu $t2, $t2, -1
        bnez  $t2, rb_word
        li    $v0, 0
        jr    $ra
rb_error:
        li    $v0, 1
        jr    $ra

        .data
model_txt:   .asciiz "model: "
sectors_txt: .asciiz "\nsectors: "
        .align 2
ident:  .space 512
sector: .space 513                  # + a 0: print it as a string
# eMIPS: --board malta --disk malta_disk.txt    (the settings this program runs with)
