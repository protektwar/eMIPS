# Lab 93 — A tune: a melody from a table (service 33: play and wait), then
# eight notes drawn at random from a pentatonic scale (40 seeds a stream,
# 42 draws), timed with service 30.  It prints the milliseconds each part
# took and the notes drawn.
# Run:  python3 -m emips run docs/guide/labs/tune.asm --midi-wav tune.wav
#       (the window: the Sound pane lists the notes, Save WAV / Save MIDI)
        .data
melody: .word 60, 250,  62, 250,  64, 250,  60, 250     # pitch, ms
        .word 64, 250,  65, 250,  67, 500
        .word 0                                         # the end
scale:  .word 60, 62, 64, 67, 69                        # C D E G A
        .text
main:   li    $v0, 30                 # time: the milliseconds now
        syscall
        move  $s7, $a0
        la    $s0, melody
play:   lw    $a0, 0($s0)             # pitch
        beqz  $a0, tunes
        lw    $a1, 4($s0)             # duration
        li    $a2, 0                  # instrument 0: piano
        li    $a3, 100                # volume
        li    $v0, 33                 # play and wait
        syscall
        addiu $s0, $s0, 8
        b     play
tunes:  li    $v0, 30
        syscall
        subu  $a0, $a0, $s7           # the melody's milliseconds
        li    $v0, 1
        syscall
        li    $a0, '\n'
        li    $v0, 11
        syscall
        li    $v0, 40                 # seed stream 1 with 2026
        li    $a0, 1
        li    $a1, 2026
        syscall
        li    $v0, 30
        syscall
        move  $s7, $a0
        li    $s1, 8
rand:   li    $v0, 42                 # an index 0 .. 4 from stream 1
        li    $a0, 1
        li    $a1, 5
        syscall
        sll   $a0, $a0, 2
        lw    $a0, scale($a0)
        move  $s2, $a0
        li    $a1, 125
        li    $a2, 11                 # instrument 11: vibraphone
        li    $a3, 90
        li    $v0, 33
        syscall
        move  $a0, $s2                # print the note
        li    $v0, 1
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        addiu $s1, $s1, -1
        bnez  $s1, rand
        li    $v0, 30
        syscall
        subu  $a0, $a0, $s7           # their milliseconds
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall
