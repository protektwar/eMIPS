# Memory-mapped I/O (MARS "Keyboard and Display MMIO Simulator"):
#   0xffff0000 receiver control (bit 0: a key is ready)
#   0xffff0004 receiver data    (the key)
#   0xffff0008 transmitter control (bit 0: ready to display)
#   0xffff000c transmitter data (the character to display)
# Echoes what you type in the Console "MMIO keyboard" field, until Enter.
        .text
main:   lui   $s0, 0xffff       # $s0 = 0xffff0000
        li    $s1, 10           # '\n'
wait_rx: lw   $t0, 0($s0)       # polling: is a key ready?
        andi  $t0, $t0, 1
        beqz  $t0, wait_rx
        lw    $t1, 4($s0)       # read it (clears "ready")
wait_tx: lw   $t0, 8($s0)       # can the display take a character?
        andi  $t0, $t0, 1
        beqz  $t0, wait_tx
        sw    $t1, 12($s0)      # display it
        bne   $t1, $s1, wait_rx
        li    $v0, 10
        syscall
