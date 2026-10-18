# Load-use hazard: the addu needs $t0 one cycle after the lw loads it.
        .text
main:   lw    $t0, 0($sp)
        addu  $t1, $t0, $t0     # needs $t0: 1 stall, then forwarded from MEM/WB
        li    $v0, 10
        syscall
# eMIPS: --pipeline    (the settings this program runs with)
