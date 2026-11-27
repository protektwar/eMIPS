# MIPS by Example in eMIPS, episode 45 — A tiny text adventure.  The world
# is data: each room a record of six words — its name, its description, and
# the room to the north, south, east and west (-1: a wall).  The program only
# reads a command, looks the exit up in the record, and moves.  One door is
# locked: the garden needs the key from the library.
        .data
        .eqv  ROOM, 24                # a room record: 6 words
        .eqv  NAME, 0
        .eqv  DESC, 4
        .eqv  EXITS, 8                # north, south, east, west: 8, 12, 16, 20
        .eqv  GARDEN, 3
rooms:  .word nhall, dhall, 1, 3, 2, -1      # 0 the hall
        .word nlib, dlib, -1, 0, -1, -1      # 1 the library
        .word nkit, dkit, -1, -1, -1, 0      # 2 the kitchen
        .word ngar, dgar, 0, -1, -1, -1      # 3 the garden
nhall:  .asciiz "hall"
dhall:  .asciiz "Doors lead north, east and south."
nlib:   .asciiz "library"
dlib:   .asciiz "Shelves of old manuals; the door is south."
nkit:   .asciiz "kitchen"
dkit:   .asciiz "It smells of coffee; the door is west."
ngar:   .asciiz "garden"
dgar:   .asciiz "Sunlight at last!"
youare: .asciiz "You are in the "
dot:    .asciiz ". "
keyhere: .asciiz "There is a key here.\n"
prompt: .asciiz "> "
wall:   .asciiz "You can't go that way.\n"
locked: .asciiz "The door to the garden is locked.\n"
opened: .asciiz "You open the door with the key.\n"
took:   .asciiz "You take the key.\n"
nokey:  .asciiz "There is no key here.\n"
what:   .asciiz "I don't understand.\n"
win:    .asciiz "You win!\n"
bye:    .asciiz "Bye.\n"
dirs:   .asciiz "nsew"                # the commands that move, in EXITS order
line:   .space 32                     # the command typed

        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s1, 16 $s0, 0-15 the home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s1, 20($sp)
        sw    $s0, 16($sp)
        move  $fp, $sp
        li    $s0, 0                  # $s0: the room we are in (the hall)
        li    $s1, 1                  # $s1: the room the key lies in (-1: we have it)
        move  $a0, $s0
        move  $a1, $s1
        jal   describe
loop:   li    $v0, 4                  # "> " and a command
        la    $a0, prompt
        syscall
        li    $v0, 8
        la    $a0, line
        li    $a1, 32
        syscall
        lbu   $t0, line               # its first letter decides
        beq   $t0, 'q', quit
        beq   $t0, 'l', look
        beq   $t0, 't', take
        li    $t1, 0                  # n s e w: which exit? (its index in dirs)
find:   lbu   $t2, dirs($t1)
        beqz  $t2, huh                # not a direction either
        beq   $t2, $t0, go
        addiu $t1, $t1, 1
        b     find
go:     li    $t3, ROOM               # the exit: rooms[$s0].exits[$t1]
        mul   $t3, $s0, $t3
        sll   $t4, $t1, 2
        addu  $t3, $t3, $t4
        lw    $t5, rooms+EXITS($t3)
        bltz  $t5, nogo               # -1: a wall
        bne   $t5, GARDEN, enter
        bltz  $s1, unlock             # the garden: do we have the key?
        li    $v0, 4
        la    $a0, locked
        syscall
        b     loop
unlock: li    $v0, 4
        la    $a0, opened
        syscall
enter:  move  $s0, $t5                # we are there
look:   move  $a0, $s0
        move  $a1, $s1
        jal   describe
        bne   $s0, GARDEN, loop
        li    $v0, 4                  # the garden: the end
        la    $a0, win
        syscall
        b     done
nogo:   li    $v0, 4
        la    $a0, wall
        syscall
        b     loop
take:   bne   $s1, $s0, notake        # the key in this room?
        li    $s1, -1                 # now we have it
        li    $v0, 4
        la    $a0, took
        syscall
        b     loop
notake: li    $v0, 4
        la    $a0, nokey
        syscall
        b     loop
huh:    li    $v0, 4
        la    $a0, what
        syscall
        b     loop
quit:   li    $v0, 4
        la    $a0, bye
        syscall
done:   move  $sp, $fp                # main's epilogue
        lw    $s0, 16($sp)
        lw    $s1, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# describe(room, keyroom): "You are in the NAME. DESC" and the key, if it
# lies here (a leaf)
describe:
        li    $t0, ROOM               # the record: rooms + room × 24
        mul   $t0, $a0, $t0
        la    $t1, rooms
        addu  $t0, $t1, $t0
        li    $v0, 4
        move  $t2, $a0
        la    $a0, youare
        syscall
        lw    $a0, NAME($t0)
        syscall
        la    $a0, dot
        syscall
        lw    $a0, DESC($t0)
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        bne   $a1, $t2, dend          # the key here?
        li    $v0, 4
        la    $a0, keyhere
        syscall
dend:   jr    $ra
