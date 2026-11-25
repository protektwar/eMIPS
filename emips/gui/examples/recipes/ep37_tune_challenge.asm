# MIPS by Example in eMIPS, episode 37's challenge, solved: a bass under
# the tune.  At the first beat of every bar (the beats so far a multiple of
# 4) a bass note starts with service 31 — which does not wait, so the
# melody's note, played with 33 right after it, sounds together with it.
        .data
        .eqv  TEMPO, 400              # milliseconds a beat
twinkle:.word 60,1, 60,1, 67,1, 67,1, 69,1, 69,1, 67,2     # C C G G A A G-
        .word 65,1, 65,1, 64,1, 64,1, 62,1, 62,1, 60,2     # F F E E D D C-
        .word 67,1, 67,1, 65,1, 65,1, 64,1, 64,1, 62,2     # G G F F E E D-
        .word 67,1, 67,1, 65,1, 65,1, 64,1, 64,1, 62,2
        .word 60,1, 60,1, 67,1, 67,1, 69,1, 69,1, 67,2
        .word 65,1, 65,1, 64,1, 64,1, 62,1, 62,1, 60,2
        .word 0                       # the end
bass:   .word 48, 53, 53, 48, 48, 55, 48, 55, 48, 53, 53, 48   # a note a bar: C F F C ...
notes:  .asciiz "notes: "
ms:     .asciiz "\nms: "
bars:   .asciiz "\nbass notes: "
        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 0-15 the home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        li    $v0, 30                 # the time now (the simulated clock)
        syscall
        move  $s0, $a0
        la    $a0, twinkle            # play(tune, tempo, instrument)
        li    $a1, TEMPO
        li    $a2, 0                  # piano
        jal   play
        move  $t0, $v0
        li    $v0, 4                  # "notes: n"
        la    $a0, notes
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 4                  # "bass notes: n"
        la    $a0, bars
        syscall
        li    $v0, 1
        move  $a0, $v1
        syscall
        li    $v0, 30                 # "ms: the time it took"
        syscall
        subu  $t0, $a0, $s0
        li    $v0, 4
        la    $a0, ms
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# play(tune, tempo, instrument) -> $v0 = the notes played, $v1 = the bass
# notes: each (pitch, beats) pair for beats * tempo milliseconds, until
# pitch 0; at every bar's first beat a bass note under it (a leaf)
play:   move  $t0, $a0                # the next pair
        move  $t1, $a1                # the tempo
        move  $t2, $a2                # the instrument
        li    $t3, 0                  # notes played
        li    $t4, 0                  # beats so far
        li    $v1, 0                  # bass notes played
pnext:  lw    $a0, 0($t0)             # pitch
        beqz  $a0, pdone
        andi  $t5, $t4, 3             # the first beat of a bar?
        bnez  $t5, pmelody
        sll   $t5, $v1, 2             # the bar's bass note ...
        lw    $a0, bass($t5)
        sll   $a1, $t1, 2             # ... for a whole bar: 4 beats
        li    $a2, 32                 # acoustic bass
        li    $a3, 80
        li    $v0, 31                 # play it, do not wait
        syscall
        addiu $v1, $v1, 1
        lw    $a0, 0($t0)             # (the melody's pitch again)
pmelody:lw    $a1, 4($t0)             # beats ...
        addu  $t4, $t4, $a1
        mul   $a1, $a1, $t1           # ... times the tempo: milliseconds
        move  $a2, $t2
        li    $a3, 100                # volume
        li    $v0, 33                 # play the note and wait for its end
        syscall
        addiu $t3, $t3, 1
        addiu $t0, $t0, 8
        b     pnext
pdone:  move  $v0, $t3
        jr    $ra
