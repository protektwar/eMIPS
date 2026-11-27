# MIPS by Example in eMIPS, episode 45 — the challenge, solved: a lamp, a
# dark cellar and an inventory.  Two more exits, up and down, make a room
# record eight words; the cellar under the kitchen is dark without the lamp
# (from the library), and the key now lies in the cellar.  Items are a
# table too: the room each lies in, -1 when we carry it.  i lists them.
# (The episode: a tiny text adventure.)
        .data
        .eqv  ROOM, 32                # a room record: 8 words
        .eqv  NAME, 0
        .eqv  DESC, 4
        .eqv  EXITS, 8                # north, south, east, west, up, down: 8 ... 28
        .eqv  GARDEN, 3
        .eqv  CELLAR, 4
        .eqv  ITEMS, 2
rooms:  .word nhall, dhall, 1, 3, 2, -1, -1, -1      # 0 the hall
        .word nlib, dlib, -1, 0, -1, -1, -1, -1      # 1 the library
        .word nkit, dkit, -1, -1, -1, 0, -1, 4       # 2 the kitchen
        .word ngar, dgar, 0, -1, -1, -1, -1, -1      # 3 the garden
        .word ncel, dcel, -1, -1, -1, -1, 2, -1      # 4 the cellar
items:  .word 1, 4                    # where the lamp and the key lie (-1: carried)
inames: .word nlamp, nkey
nhall:  .asciiz "hall"
dhall:  .asciiz "Doors lead north, east and south."
nlib:   .asciiz "library"
dlib:   .asciiz "Shelves of old manuals; the door is south."
nkit:   .asciiz "kitchen"
dkit:   .asciiz "It smells of coffee; the door is west, stairs go down."
ngar:   .asciiz "garden"
dgar:   .asciiz "Sunlight at last!"
ncel:   .asciiz "cellar"
dcel:   .asciiz "Dusty barrels; the stairs go up."
nlamp:  .asciiz "lamp"
nkey:   .asciiz "key"
youare: .asciiz "You are in the "
dot:    .asciiz ". "
there:  .asciiz "There is a "
here:   .asciiz " here.\n"
dark:   .asciiz "It is too dark to see.\n"
prompt: .asciiz "> "
wall:   .asciiz "You can't go that way.\n"
locked: .asciiz "The door to the garden is locked.\n"
opened: .asciiz "You open the door with the key.\n"
took:   .asciiz "You take the "
tend:   .asciiz ".\n"
nothing: .asciiz "There is nothing here.\n"
carry:  .asciiz "You carry:"
none:   .asciiz " nothing"
what:   .asciiz "I don't understand.\n"
win:    .asciiz "You win!\n"
bye:    .asciiz "Bye.\n"
dirs:   .asciiz "nsewud"              # the commands that move, in EXITS order
line:   .space 32                     # the command typed

        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 16 $s0, 0-15 the home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 16($sp)
        move  $fp, $sp
        li    $s0, 0                  # $s0: the room we are in (the hall)
        move  $a0, $s0
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
        bne   $t0, 't', notake
        move  $a0, $s0                # take(room)
        jal   take
        b     loop
notake: bne   $t0, 'i', noinv
        jal   inventory
        b     loop
noinv:  li    $t1, 0                  # n s e w u d: which exit?
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
        lw    $t6, items+4            # the garden: do we have the key?
        bltz  $t6, unlock
        li    $v0, 4
        la    $a0, locked
        syscall
        b     loop
unlock: li    $v0, 4
        la    $a0, opened
        syscall
enter:  move  $s0, $t5                # we are there
look:   move  $a0, $s0
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
huh:    li    $v0, 4
        la    $a0, what
        syscall
        b     loop
quit:   li    $v0, 4
        la    $a0, bye
        syscall
done:   move  $sp, $fp                # main's epilogue
        lw    $s0, 16($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# dark_here(room) -> $v0: 1 in the cellar without the lamp (a leaf)
dark_here:
        li    $v0, 0
        bne   $a0, CELLAR, dkdone
        lw    $t0, items              # the lamp: carried?
        bltz  $t0, dkdone
        li    $v0, 1
dkdone: jr    $ra

# describe(room): its name and description and the items lying there — or
# the dark
describe:
        addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s1, 16 $s0
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s1, 20($sp)
        sw    $s0, 16($sp)
        move  $fp, $sp
        move  $s0, $a0                # $s0: the room
        jal   dark_here
        beqz  $v0, light
        li    $v0, 4
        la    $a0, dark
        syscall
        b     ddone
light:  li    $t0, ROOM               # the record: rooms + room × 32
        mul   $t0, $s0, $t0
        la    $t1, rooms
        addu  $t0, $t1, $t0
        li    $v0, 4
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
        li    $s1, 0                  # $s1: each item: lying here?
ditem:  sll   $t0, $s1, 2
        lw    $t1, items($t0)
        bne   $t1, $s0, dnext
        li    $v0, 4                  # "There is a lamp here."
        la    $a0, there
        syscall
        lw    $a0, inames($t0)
        syscall
        la    $a0, here
        syscall
dnext:  addiu $s1, $s1, 1
        blt   $s1, ITEMS, ditem
ddone:  move  $sp, $fp
        lw    $s0, 16($sp)
        lw    $s1, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# take(room): every item lying here is carried now — not in the dark
take:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s1, 16 $s0
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s1, 20($sp)
        sw    $s0, 16($sp)
        move  $fp, $sp
        move  $s0, $a0
        jal   dark_here
        bnez  $v0, tnone              # in the dark: nothing to see, nothing taken
        li    $s1, 0                  # each item: here? (-1 marks it carried)
        li    $t2, 0                  # how many we took
titem:  sll   $t0, $s1, 2
        lw    $t1, items($t0)
        bne   $t1, $s0, tnext
        li    $t1, -1
        sw    $t1, items($t0)
        addiu $t2, $t2, 1
        li    $v0, 4                  # "You take the lamp."
        la    $a0, took
        syscall
        lw    $a0, inames($t0)
        syscall
        la    $a0, tend
        syscall
tnext:  addiu $s1, $s1, 1
        blt   $s1, ITEMS, titem
        bnez  $t2, tdone
tnone:  li    $v0, 4
        la    $a0, nothing
        syscall
tdone:  move  $sp, $fp
        lw    $s0, 16($sp)
        lw    $s1, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# inventory(): "You carry: lamp key" (a leaf)
inventory:
        li    $v0, 4
        la    $a0, carry
        syscall
        li    $t0, 0                  # each item's offset
        li    $t2, 0                  # how many we carry
iitem:  lw    $t1, items($t0)
        bgez  $t1, inext
        addiu $t2, $t2, 1
        li    $v0, 11
        li    $a0, ' '
        syscall
        li    $v0, 4
        lw    $a0, inames($t0)
        syscall
inext:  addiu $t0, $t0, 4
        blt   $t0, 4 * ITEMS, iitem
        bnez  $t2, iend
        li    $v0, 4
        la    $a0, none
        syscall
iend:   li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra
