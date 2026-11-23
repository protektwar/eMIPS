# Sound: a C major scale on the piano, up and down (MARS's service 33: play
# a note and wait for it).  The Sound pane lists the notes, plays them
# when your Qt has QtMultimedia, and saves them as WAV or MIDI.
        .data
scale:  .word 60, 62, 64, 65, 67, 69, 71, 72
        .text
main:   la    $s0, scale
        li    $s1, 8
up:     lw    $a0, 0($s0)             # pitch (60 = middle C)
        jal   note
        addiu $s0, $s0, 4
        addiu $s1, $s1, -1
        bnez  $s1, up
        li    $s1, 7
down:   addiu $s0, $s0, -4
        lw    $a0, -4($s0)
        jal   note
        addiu $s1, $s1, -1
        bnez  $s1, down
        li    $v0, 10
        syscall
note:   li    $a1, 200                # duration (ms)
        li    $a2, 0                  # instrument: 0 = piano
        li    $a3, 100                # volume
        li    $v0, 33
        syscall
        jr    $ra
